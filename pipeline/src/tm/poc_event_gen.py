"""
PoC Event Generator — converts harvested Polymarket events into pipeline event JSONs.

Reads data/poc/pm_harvest/events.jsonl, auto-generates search keywords using
Nova Micro (batched), and writes data/poc/events/pm_{pm_id}.json for each event.

Usage:
    python -m tm.poc_event_gen
    python -m tm.poc_event_gen --data-dir data/poc --batch-size 50
"""

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, MofNCompleteColumn

from .config import Settings
from .llm import complete_text

logger = logging.getLogger(__name__)
console = Console()

KEYWORD_PROMPT = """\
You are helping build a news search system. Given a Polymarket prediction market question, \
generate exactly 5 short search keywords or phrases that journalists would use when writing \
about this topic BEFORE the event resolves.

Rules:
- Each keyword/phrase should be 2-5 words
- Focus on the core subject, not the outcome
- No yes/no framing (not "will X happen")
- Return a JSON array of strings only, no explanation

Question: {question}

Return format: ["phrase1", "phrase2", "phrase3", "phrase4", "phrase5"]
"""


async def _generate_keywords(question: str, settings: Settings) -> list[str]:
    """Generate search keywords for a question via Bedrock Nova (tm.llm)."""
    try:
        text = (await complete_text(
            settings.gatekeeper_model,
            KEYWORD_PROMPT.format(question=question),
            max_tokens=150,
            temperature=0.3,
        )).strip()
        # Extract JSON array from response
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            keywords = json.loads(match.group())
            if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
                return [k.strip() for k in keywords[:5] if k.strip()]
    except Exception as exc:
        logger.warning("Keyword generation failed for %r: %s", question[:60], exc)

    # Fallback: extract noun phrases from question
    words = re.findall(r'\b[A-Z][a-zA-Z]+\b', question)
    return words[:5] if words else [question[:60]]


def _question_to_event_id(pm_id: str) -> str:
    """Generate a stable short event ID from Polymarket market ID."""
    # Use first 8 chars of pm_id, prefixed with 'pm'
    safe = re.sub(r'[^a-zA-Z0-9]', '', str(pm_id))[:8]
    return f"pm{safe}"


def _infer_tags(question: str, category: str) -> list[str]:
    """Extract rough topic tags from question text."""
    tags = [category]
    # Country/region detection
    countries = [
        "US", "UK", "EU", "Russia", "China", "Ukraine", "Israel", "Iran",
        "France", "Germany", "Brazil", "India", "North Korea", "Taiwan",
    ]
    for c in countries:
        if c.lower() in question.lower():
            tags.append(c)
    return list(dict.fromkeys(tags))  # deduplicate, preserve order


async def generate_events(
    data_dir: Path,
    batch_size: int = 50,
    overwrite: bool = False,
) -> int:
    """
    Read events.jsonl, generate event JSONs for any not yet created.
    Returns count of events generated.
    """
    settings = Settings()
    harvest_path = data_dir / "pm_harvest" / "events.jsonl"
    events_dir = data_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    if not harvest_path.exists():
        console.print(f"[red]Harvest file not found: {harvest_path}[/red]")
        console.print("Run: python -m tm.polymarket_harvest first")
        return 0

    # Load all harvested events
    harvested: list[dict] = []
    with open(harvest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                harvested.append(json.loads(line))

    console.print(f"[bold cyan]PoC Event Generator[/bold cyan] — {len(harvested)} harvested events")

    # Filter to those not yet generated
    pending = []
    for ev in harvested:
        event_id = _question_to_event_id(ev["pm_id"])
        out_path = events_dir / f"{event_id}.json"
        if not out_path.exists() or overwrite:
            pending.append((ev, event_id, out_path))

    console.print(f"  {len(pending)} events to generate ({len(harvested) - len(pending)} already done)")
    if not pending:
        return 0

    generated = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating keywords...", total=len(pending))

        # Process in batches
        for i in range(0, len(pending), batch_size):
            batch = pending[i: i + batch_size]

            # Generate keywords for all events in batch concurrently
            keyword_tasks = [
                _generate_keywords(ev["question"], settings)
                for ev, _, _ in batch
            ]
            keywords_list = await asyncio.gather(*keyword_tasks, return_exceptions=True)

            for (ev, event_id, out_path), keywords in zip(batch, keywords_list):
                if isinstance(keywords, Exception):
                    logger.warning("Keyword error for %s: %s", event_id, keywords)
                    keywords = [ev["question"][:60]]

                event_json = {
                    "id": event_id,
                    "name": ev["question"],
                    "outcome": ev["outcome"],
                    "outcome_date": ev["outcome_date"],
                    "search_keywords": keywords,
                    "llm_referee_criteria": (
                        f"Polymarket market resolved {'Yes' if ev['outcome'] else 'No'}. "
                        f"Source: {ev.get('pm_url', '')}"
                    ),
                    "predictive_window_days": 30,
                    "category": [ev.get("category", "Politics")],
                    "tags": _infer_tags(ev["question"], ev.get("category", "Politics")),
                    "polymarket": ev.get("pm_url", ""),
                    "_pm_id": ev["pm_id"],
                    "_pm_prices": ev.get("prices", []),
                }

                with open(out_path, "w") as f:
                    json.dump(event_json, f, indent=2, ensure_ascii=False)

                generated += 1
                progress.advance(task)

    console.print(f"\n[bold green]Done.[/bold green] Generated {generated} event JSONs → {events_dir}")
    return generated


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Generate PoC event JSONs from Polymarket harvest")
    parser.add_argument("--data-dir", default="data/poc", help="Base PoC data directory")
    parser.add_argument("--batch-size", type=int, default=50, help="Keyword generation batch size")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing event JSONs")
    args = parser.parse_args()

    import os
    base = Path(os.environ.get("DATA_DIR", args.data_dir))
    asyncio.run(generate_events(base, batch_size=args.batch_size, overwrite=args.overwrite))
