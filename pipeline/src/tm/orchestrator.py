import asyncio
import json
import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from enum import Enum

from rich.console import Console
from .config import settings
from .dedup import SimhashIndex
from .models import CellStatus, ExtractionOutput, PredictionExtraction
from .aggregator import aggregate_predictions, needs_aggregation, aggregate_article_predictions
from .runner import run_article, ArticleInput, PipelineResult
from .progress import update_cell, load_state
from .ingestor import BraveIngestor, GDELTIngestor, DDGIngestor
from .utils import KNOWN_SOURCE_IDS
from .gnews_ingest import _is_stub_page
import httpx
from bs4 import BeautifulSoup

console = Console()

class SearchMode(str, Enum):
    mock = "mock"
    local_file = "local_file"
    api = "api"

class Orchestrator:
    def __init__(self, data_dir: Path, mode: SearchMode = SearchMode.mock, force_reextract: bool = False):
        self.data_dir = data_dir
        self.mode = mode
        self.force_reextract = force_reextract
        _vault_default = str(settings.vault_dir) if settings.vault_dir != Path("") else str(data_dir / "vault2")
        self.vault_dir = Path(os.environ.get("VAULT_DIR", _vault_default))
        self.atlas_dir = data_dir / "atlas"
        self.events_dir = data_dir / "events"
        self.sources_dir = data_dir / "sources"
        self.raw_ingest_dir = data_dir / "raw_ingest"
        
        for d in [self.vault_dir / "articles", self.vault_dir / "extractions", self.raw_ingest_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._simhash_idx = SimhashIndex(self.vault_dir)

        # Use DDG as primary ingestor; Brave demoted to fallback (quota exhausted)
        self.ingestor = DDGIngestor()
        brave_key = os.environ.get("BRAVE_API_KEY", settings.brave_api_key)
        self.brave = BraveIngestor(brave_key) if brave_key else None
        self.ddg = DDGIngestor()  # always available as fallback

    def get_article_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def get_full_text(self, url: str) -> str:
        console.print(f"    [dim]Scraping full text from: {url}[/dim]")
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, follow_redirects=True)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    for s in soup(["script", "style", "nav", "footer", "header"]):
                        s.extract()
                    return soup.get_text(separator=" ", strip=True)
        except Exception as e:
            console.print(f"    [dim red]Scrape failed: {str(e)}[/dim red]")
        return ""

    async def run_event(self, event_id: str):
        event_path = self.events_dir / f"{event_id}.json"
        if not event_path.exists():
            return
        
        with open(event_path, "r") as f:
            event = json.load(f)
        
        outcome_date = datetime.strptime(event["outcome_date"], "%Y-%m-%d")
        window_days = event.get("predictive_window_days", 14)
        start_date = outcome_date - timedelta(days=window_days)
        
        console.print(f"[bold cyan]Processing Event:[/bold cyan] {event['name']} ({event_id})")

        source_files = sorted(list(self.sources_dir.glob("*.json")))
        for source_file in source_files:
            with open(source_file, "r") as f:
                source = json.load(f)
            
            if source["id"] not in KNOWN_SOURCE_IDS:
                continue

            state = load_state()
            cell = state.get(event_id, source["id"])
            skip_statuses = (CellStatus.done, CellStatus.no_predictions) if not self.force_reextract else (CellStatus.done,)
            if cell.status in skip_statuses:
                continue

            console.print(f"  [bold]Source:[/bold] {source['name']}")

            articles = await self.search_articles(source, event, start_date, outcome_date)

            if not articles:
                if cell.status != CellStatus.failed:
                    update_cell(event_id, source["id"], CellStatus.no_predictions)
                continue

            # Track whether any article hit an infra-level error (timeout,
            # exception, runner-set error). The final cell-status branch
            # below uses this to distinguish "genuinely no predictions found"
            # from "we don't know — something broke". Without this, any
            # non-TimeoutError exception bubbling out of process_article
            # would abort the whole cell loop, skipping the final status
            # write and leaving the cell stuck at whatever in-progress state
            # runner.py last set.
            had_errors = False
            for raw_art in articles:
                try:
                    result = await asyncio.wait_for(
                        self.process_article(raw_art, event, source),
                        timeout=300,
                    )
                    if result is not None and result.error:
                        had_errors = True
                except asyncio.TimeoutError:
                    console.print(f"    [bold red]Article timeout (300s), skipping[/bold red]")
                    had_errors = True
                except Exception as e:
                    console.print(
                        f"    [bold red]Article exception ({type(e).__name__}): {e}[/bold red]"
                    )
                    had_errors = True
                await asyncio.sleep(3)

            self._write_cell_signal(event["id"], source["id"])
            # Final cell status is derived from two signals:
            #   1. Did any article actually write predictions to disk? (has_predictions)
            #   2. Did any article hit an infra-level error? (had_errors)
            # Precedence: predictions-on-disk wins — partial success is still
            # success, even if other articles in the cell errored. Without
            # predictions, had_errors routes to `failed` (visible in the
            # atlas as a real problem) vs `no_predictions` (genuine empty
            # result — e.g. gatekeeper rejected everything).
            cell_dir = self.atlas_dir / event_id / source["id"]
            has_predictions = any(
                json.loads(f.read_text()).get("predictions")
                for f in cell_dir.glob("entry_*.json")
                if f.exists()
            ) if cell_dir.exists() else False
            if has_predictions:
                total_preds = sum(
                    len(json.loads(f.read_text()).get("predictions", []))
                    for f in cell_dir.glob("entry_*.json")
                    if f.exists()
                )
                update_cell(event_id, source["id"], CellStatus.done, prediction_count=total_preds)
            else:
                if had_errors:
                    update_cell(event_id, source["id"], CellStatus.failed)
                else:
                    update_cell(event_id, source["id"], CellStatus.no_predictions)

    async def search_articles(self, source: dict, event: dict, start: datetime, end: datetime) -> List[dict]:
        if self.mode == SearchMode.local_file:
            return self.local_file_search(source["id"], event["id"], start, end)
        
        if self.mode == SearchMode.api:
            domain = (
                source["url"]
                .replace("https://www.", "")
                .replace("http://www.", "")
                .replace("https://", "")
                .replace("http://", "")
                .split("/")[0]
            )
            if isinstance(self.ingestor, BraveIngestor):
                results = await self.ingestor.search(domain, event["search_keywords"], start, end)
                if not results:
                    console.print(f"    [dim]Brave empty → DDG fallback[/dim]")
                    results = await self.ddg.search(domain, event["search_keywords"], start, end)
            else:
                results = await self.ddg.search(domain, event["search_keywords"], start, end)

            results = results[:5]
            for res in results:
                if res.get("url"):
                    full_text = await self.get_full_text(res["url"])
                    if len(full_text) > 200:
                        res["text"] = full_text
            return [r for r in results if len(r.get("text", "")) > 200]
            
        return []

    def local_file_search(
        self, source_id: str, event_id: str,
        start: Optional[datetime] = None, end: Optional[datetime] = None,
    ) -> List[dict]:
        articles = []
        search_path = self.raw_ingest_dir / source_id / event_id
        if search_path.exists():
            for art_file in search_path.glob("*.json"):
                with open(art_file, "r") as f:
                    art = json.load(f)
                url = art.get("url", "")
                text = art.get("text", "")
                # Anti-lookahead: locally-staged articles bypass the web ingest's
                # date filter, so enforce the predictive window here too. Drop any
                # article published after the outcome (leak) or before the window.
                pub = art.get("published_at", "")
                if pub:
                    try:
                        pub_dt = datetime.fromisoformat(pub[:10])
                    except (ValueError, TypeError):
                        pub_dt = None
                    if pub_dt is not None:
                        if end is not None and pub_dt > end:
                            console.print(f"    [yellow]Dropping post-outcome article ({pub[:10]}): {url[:60]}[/yellow]")
                            continue
                        if start is not None and pub_dt < start:
                            console.print(f"    [dim]Skipping pre-window article ({pub[:10]}): {url[:60]}[/dim]")
                            continue
                # Skip liveblogs (day-long rolling updates starting with photo captions)
                if "liveblog" in url.lower():
                    console.print(f"    [dim]Skipping liveblog: {url[:60]}[/dim]")
                    continue
                # Skip very short articles (paywalled stubs)
                if len(text) < 500:
                    console.print(f"    [dim]Skipping stub ({len(text)} chars): {art.get('headline','')[:50]}[/dim]")
                    continue
                # Skip known paywall redirect pages (e.g. Bloomberg Mercury)
                if _is_stub_page(text):
                    console.print(f"    [dim]Skipping known stub page: {art.get('headline','')[:50]}[/dim]")
                    continue
                articles.append(art)
        return articles

    async def process_article(self, raw_art: dict, event: dict, source: dict) -> Optional[PipelineResult]:
        text = raw_art["text"]
        if len(text) < 500:
            console.print(f"    [dim]Skipping empty/stub article ({len(text)} chars)[/dim]")
            return None
        art_hash = self.get_article_hash(text)

        # Near-duplicate detection: if a semantically-similar article was already
        # processed, reuse its extraction rather than calling the LLM again.
        canonical_hash = self._simhash_idx.find_near_duplicate(text)
        if canonical_hash and canonical_hash != art_hash:
            console.print(f"    [dim]Near-duplicate of {canonical_hash[:8]} — reusing extraction[/dim]")
            canonical_extract = self.vault_dir / "extractions" / f"{canonical_hash}_{event['id']}_v1.json"
            if canonical_extract.exists():
                self.create_atlas_link(event["id"], source["id"], canonical_hash, canonical_extract, raw_art, event.get("outcome_date", ""))
                return None
            # Canonical extraction doesn't exist yet for this event — fall through to normal processing

        vault_path = self.vault_dir / "articles" / f"{art_hash}.json"
        if not vault_path.exists():
            with open(vault_path, "w") as f:
                json.dump(raw_art, f, indent=2)
            self._simhash_idx.add(art_hash, text)
            self._simhash_idx.save()
        
        model_v = "v1" 
        extract_path = self.vault_dir / "extractions" / f"{art_hash}_{event['id']}_{model_v}.json"
        
        if extract_path.exists() and not self.force_reextract:
            self.create_atlas_link(event["id"], source["id"], art_hash, extract_path, raw_art, event.get("outcome_date", ""))
            return None
        elif extract_path.exists() and self.force_reextract:
            console.print(f"    [yellow]--force-reextract: re-running LLM for {art_hash[:8]}[/yellow]")

        article_input = ArticleInput(
            text=text,
            source_id=source["id"],
            source_name=source["name"],
            article_date=raw_art["published_at"],
            event_id=event["id"],
            event_name=event["name"],
            event_description=event.get("llm_referee_criteria", ""),
            journalist=raw_art.get("author"),
            article_url=raw_art.get("url")
        )
        
        result = await run_article(article_input)

        if result.error:
            console.print(f"    [bold red]Runner Error:[/bold red] {result.error}")
        
        if result.extraction:
            preds = result.extraction.predictions
            console.print(f"    [green]Success![/green] Extracted {len(preds)} predictions.")

            # Article-level aggregation: collapse conflicting predictions into one
            if needs_aggregation(preds):
                stances = [p.stance for p in preds]
                spread = max(stances) - min(stances)
                console.print(f"    [yellow]Stance spread {spread:.2f} → aggregating {len(preds)} predictions[/yellow]")
                try:
                    agg = await aggregate_article_predictions(
                        preds,
                        event_name=event["name"],
                        source_name=source["name"],
                        article_date=article_input.article_date,
                    )
                    result.extraction.predictions = [agg]
                    console.print(f"    [cyan]Aggregated to 1 prediction (stance={agg.stance:.2f})[/cyan]")
                except Exception as e:
                    console.print(f"    [dim red]Article aggregation failed, keeping originals: {e}[/dim red]")

            try:
                with open(extract_path, "w") as f:
                    data = {
                        "extraction": result.extraction.model_dump(),
                        "prompt_version": model_v,
                        "extractor_model": settings.extractor_model,
                        "gatekeeper_model": settings.gatekeeper_model,
                        "gatekeeper_reason": result.gatekeeper_reason,
                        "run_date": datetime.now().isoformat()
                    }
                    json.dump(data, f, indent=2)
                console.print(f"    [dim]Saved to vault: {extract_path}[/dim]")
                self.create_atlas_link(event["id"], source["id"], art_hash, extract_path, raw_art, event.get("outcome_date", ""))
            except Exception as e:
                console.print(f"    [bold red]Failed to save to vault:[/bold red] {str(e)}")
        else:
            console.print(f"    [yellow]Extraction resulted in 0 predictions.[/yellow]")

        return result

    def _write_cell_signal(self, eid: str, sid: str):
        """Aggregate all predictions for this cell and write cell_signal.json."""
        cell_dir = self.atlas_dir / eid / sid
        if not cell_dir.exists():
            return
        all_predictions: list[PredictionExtraction] = []
        for entry_file in cell_dir.glob("entry_*.json"):
            try:
                data = json.loads(entry_file.read_text())
                for p in data.get("predictions", []):
                    all_predictions.append(PredictionExtraction(**p))
            except Exception:
                continue
        if not all_predictions:
            return
        try:
            signal = aggregate_predictions(all_predictions)
            (cell_dir / "cell_signal.json").write_text(
                json.dumps(signal.model_dump(), indent=2, ensure_ascii=False)
            )
        except Exception as e:
            console.print(f"    [dim red]cell_signal failed {eid}/{sid}: {e}[/dim red]")

    def create_atlas_link(self, eid: str, sid: str, art_hash: str, extract_path: Path, raw_art: dict, event_date: str = ""):
        link_dir = self.atlas_dir / eid / sid
        link_dir.mkdir(parents=True, exist_ok=True)
        link_path = link_dir / f"entry_{art_hash[:8]}.json"
        with open(extract_path, "r") as f:
            ext_data = json.load(f)

        link_data = {
            "article_hash": art_hash,
            "extraction_id": f"{art_hash}_{eid}_v1",
            "headline": raw_art.get("headline", "N/A"),
            "article_url": raw_art.get("url", ""),
            "author": raw_art.get("author", ""),
            "article_date": raw_art["published_at"],
            "event_date": event_date,
            "extractor_model": ext_data.get("extractor_model", ""),
            "gatekeeper_model": ext_data.get("gatekeeper_model", ""),
            "gatekeeper_reason": ext_data.get("gatekeeper_reason", ""),
            "predictions": ext_data["extraction"].get("predictions", []),
        }
        with open(link_path, "w") as f:
            json.dump(link_data, f, indent=2)

async def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="TruthMachine Orchestrator")
    parser.add_argument("mode", nargs="?", default="mock",
                        choices=["mock", "local_file", "api"],
                        help="Search mode (default: mock)")
    parser.add_argument("--force-reextract", action="store_true",
                        help="Ignore vault cache and re-run LLM extraction for all articles. "
                             "Use after updating the extractor prompt.")
    parser.add_argument("--events", nargs="+", default=None,
                        help="Specific event IDs to process (default: all in data/events/)")
    args = parser.parse_args()

    mode = SearchMode(args.mode)
    data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))

    console.print(f"[bold green]TruthMachine Orchestrator — {mode.value} mode[/bold green]")
    if args.force_reextract:
        console.print("[yellow]--force-reextract: vault cache will be ignored[/yellow]")

    orch = Orchestrator(data_dir, mode=mode, force_reextract=args.force_reextract)

    # Auto-discover events from data/events/ or use CLI filter
    if args.events:
        events = args.events
    else:
        events = sorted(p.stem for p in (data_dir / "events").glob("*.json"))

    console.print(f"Processing {len(events)} events...")
    for eid in events:
        await orch.run_event(eid)


if __name__ == "__main__":
    asyncio.run(main())
