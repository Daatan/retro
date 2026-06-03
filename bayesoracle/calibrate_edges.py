#!/usr/bin/env python3
"""
Calibrate Bayesian edge weights using live news search + LLM.

For each edge A→B in the PM analysis graph:
  1. Search Brave News for 3 queries covering the causal relationship
  2. Fetch full article text (≥4 articles)
  3. Ask the LLM (Bedrock Nova Lite, via tm.llm) to estimate P(B|A=1) and P(B|A=0), anchored to live PM prices
  4. Save to edge_weights.json (resumable — already-done edges are skipped)

Usage:
    cd /home/mark/projects/retro
    source pipeline/.venv/bin/activate
    python bayesoracle/calibrate_edges.py
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import trafilatura

# Make pipeline importable
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline" / "src"))

# ─── Env ──────────────────────────────────────────────────────────────────────
_ENV_FILE = Path(__file__).parent.parent / "pipeline" / ".env"

def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env(_ENV_FILE)

OUTPUT = Path(__file__).parent / "edge_weights.json"

# ─── Node metadata (labels match Polymarket question text; pm = current price) ─
NODES: dict[str, dict] = {
    "IRAN_DEAL":     {"label": "US-Iran nuclear deal before 2027",                      "pm": 0.560},
    "CEASEFIRE_X":   {"label": "Israel-Hamas ceasefire cancelled by June 30 2026",      "pm": 0.160},
    "TRUMP_IL":      {"label": "Donald Trump visits Israel in 2026",                    "pm": 0.470},
    "US_WAR_IRAN":   {"label": "US invades Iran before 2027",                           "pm": 0.270},
    "IRAN_REGIME":   {"label": "Iranian regime falls by end of 2026",                   "pm": 0.180},
    "IRAN_NUKE":     {"label": "Iran conducts a nuclear test before 2027",              "pm": 0.090},
    "ELECTIONS":     {"label": "Israeli parliament dissolved by June 30 2026",          "pm": 0.590},
    "BIBI_OUT":      {"label": "Netanyahu is not Prime Minister before 2027",           "pm": 0.420},
    "BIBI_PM":       {"label": "Benjamin Netanyahu is the next Prime Minister of Israel","pm": 0.410},
    "BENNETT_PM":    {"label": "Naftali Bennett is the next Prime Minister of Israel",  "pm": 0.340},
    "EIZENKOT_PM":   {"label": "Gadi Eizenkot is the next Prime Minister of Israel",   "pm": 0.133},
    "LIEBERMAN_PM":  {"label": "Avigdor Lieberman is the next Prime Minister of Israel","pm": 0.065},
    "LAPID_PM":      {"label": "Yair Lapid is the next Prime Minister of Israel",      "pm": 0.009},
    "ANNEX_WB":      {"label": "Israel annexes West Bank territory before 2027",        "pm": 0.110},
    "ANNEX_GAZA":    {"label": "Israel annexes Gaza territory by 2026",                "pm": 0.044},
    "US_GAZA":       {"label": "US forces deployed in Gaza before 2027",               "pm": 0.150},
    "TURKEY_CLASH":  {"label": "Israel-Turkey military clash before 2027",             "pm": 0.180},
    "SAUDI_NORM":    {"label": "Israel and Saudi Arabia normalize relations before 2027","pm": 0.170},
    "SAUDI_ACCORDS": {"label": "Saudi Arabia joins the Abraham Accords before 2027",   "pm": 0.120},
    "LEBANON_NORM":  {"label": "Israel and Lebanon normalize relations before 2027",    "pm": 0.220},
    "SYRIA_NORM":    {"label": "Israel and Syria normalize relations by Dec 31 2026",   "pm": 0.120},
    "STRIKE_5":      {"label": "Israel conducts a second major military operation against Iran","pm": 0.375},
    "PAHLAVI":       {"label": "US recognizes Reza Pahlavi as leader of Iran in 2026", "pm": 0.070},
    "US_DECLARE":    {"label": "US officially declares war on Iran by Dec 31 2026",    "pm": 0.080},
}

# (source, target, edge_type)
EDGES: list[tuple[str, str, str]] = [
    # ── Primary (solid arrows — used in BayesOracle LToTP) ──────────────────
    ("CEASEFIRE_X",  "ELECTIONS",    "primary"),
    ("ELECTIONS",    "BIBI_OUT",     "primary"),
    ("BIBI_OUT",     "BIBI_PM",      "primary"),
    ("BIBI_OUT",     "BENNETT_PM",   "primary"),
    ("BIBI_OUT",     "EIZENKOT_PM",  "primary"),
    ("BIBI_OUT",     "LIEBERMAN_PM", "primary"),
    ("BIBI_OUT",     "LAPID_PM",     "primary"),
    ("BIBI_PM",      "ANNEX_WB",     "primary"),
    ("BIBI_PM",      "ANNEX_GAZA",   "primary"),
    ("BIBI_PM",      "US_GAZA",      "primary"),
    ("BIBI_PM",      "TURKEY_CLASH", "primary"),
    ("BIBI_PM",      "SAUDI_NORM",   "primary"),
    ("BIBI_PM",      "SAUDI_ACCORDS","primary"),
    ("BIBI_PM",      "LEBANON_NORM", "primary"),
    ("BIBI_PM",      "SYRIA_NORM",   "primary"),
    ("IRAN_NUKE",    "STRIKE_5",     "primary"),
    ("IRAN_REGIME",  "PAHLAVI",      "primary"),
    ("US_WAR_IRAN",  "US_DECLARE",   "primary"),
    # ── Secondary (dashed arrows — shown for context, not in Bayes calc) ────
    ("IRAN_DEAL",    "ELECTIONS",    "secondary"),
    ("TRUMP_IL",     "ELECTIONS",    "secondary"),
    ("CEASEFIRE_X",  "BIBI_OUT",     "secondary"),
    ("TRUMP_IL",     "BIBI_OUT",     "secondary"),
    ("TRUMP_IL",     "BIBI_PM",      "secondary"),
    ("TRUMP_IL",     "STRIKE_5",     "secondary"),
    ("IRAN_DEAL",    "STRIKE_5",     "secondary"),
    ("BIBI_PM",      "STRIKE_5",     "secondary"),
    ("US_WAR_IRAN",  "PAHLAVI",      "secondary"),
    ("TRUMP_IL",     "US_DECLARE",   "secondary"),
]

CONTEXT = """
ISRAEL POLITICAL CONTEXT — May 14, 2026:
• Iran-Israel war ended Feb 28 2026 ("Op. Roaring Lion"): Khamenei killed, IRGC largely dismantled
• Dissolution bill filed May 13 2026 by Netanyahu coalition → early elections expected ~Oct 2026
• Beyachad bloc formed Apr 26 (Bennett + Lapid united); leads in seat projections
• Polls: Likud 28 seats vs Beyachad 25; right bloc 50 seats vs center-left 60 seats
• Netanyahu criminal trial ongoing; verdict expected 2027; presidential pardon bid rejected
• Gaza ceasefire Oct 2025 — all hostages recovered; reconstruction stalled; Hamas disarmed
• All Polymarket markets resolve by end of 2026 or early 2027
""".strip()

PROMPT_TEMPLATE = """\
You are a calibrated geopolitical probability analyst. Your task: estimate two conditional probabilities for one causal edge in a Bayesian network of Israeli political outcomes.

{context}

━━━ EDGE TO ESTIMATE ━━━

Source event (A): {source_label}
  Current Polymarket price P(A) = {source_pm:.1%}

Target event (B): {target_label}
  Current Polymarket price P(B) = {target_pm:.1%}

━━━ WHAT TO ESTIMATE ━━━

pY = P(B | A is certain to happen)     — B's probability in a world where A definitely resolves YES
pN = P(B | A is certain NOT to happen) — B's probability in a world where A definitely resolves NO

━━━ CALIBRATION RULES ━━━

1. Consistency check: pY × {source_pm:.3f} + pN × {one_minus:.3f} should be close to {target_pm:.3f} (the PM anchor).
   Deviation < 5pp is fine; > 10pp needs a strong reason.
2. Both values must be in [0.01, 0.99].
3. Sign rule:
   - Positive causal link (A makes B more likely):  pY > P(B) > pN
   - Negative causal link (A makes B less likely):  pY < P(B) < pN
   - Near-independent:                              pY ≈ pN ≈ P(B)
4. Strength guide: |pY − pN| ≥ 0.25 = strong, 0.10–0.24 = moderate, < 0.10 = weak.
5. Never let pY or pN collapse to P(B) unless you genuinely believe independence.

━━━ EVIDENCE (recent news articles) ━━━

{article_block}

━━━ RESPONSE ━━━
Respond with valid JSON only — no markdown, no preamble:
{{
  "pY": <float>,
  "pN": <float>,
  "reasoning": "<2 sentences explaining the causal mechanism and strength>"
}}"""


# ─── Search ───────────────────────────────────────────────────────────────────

import logging
logging.getLogger("tm.web_search").setLevel(logging.ERROR)  # suppress provider-quota noise

from tm.web_search import search_articles as _tm_search, SearchResult
from tm.config import settings as _tm_settings
from tm.llm import complete_text


def _search_articles(src_label: str, tgt_label: str) -> list[dict]:
    """Run 3 queries via the tm provider chain and return deduplicated candidates."""
    queries = [
        f"{tgt_label} Israel 2026",
        f"{src_label} {tgt_label} Israel probability",
        f"{src_label} political impact Israel 2026",
    ]
    seen, candidates = set(), []
    for q in queries:
        try:
            for r in _tm_search(q, limit=6):
                if r.url not in seen:
                    seen.add(r.url)
                    candidates.append({
                        "url": r.url,
                        "title": r.title or "",
                        "snippet": r.snippet or "",
                    })
        except Exception as exc:
            print(f"      search error ({q[:40]}): {exc}")
        time.sleep(0.3)
    return candidates


# ─── Article fetch ────────────────────────────────────────────────────────────

def _fetch_text(url: str, fallback: str) -> str:
    try:
        r = httpx.get(
            url, timeout=7, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; BayesOracle/1.0)"},
        )
        r.raise_for_status()
        text = trafilatura.extract(r.text, include_comments=False, include_tables=False)
        if text and len(text) > 400:
            return text[:3000]
    except Exception:
        pass
    return fallback[:600]


def _collect_articles(candidates: list[dict], target: int = 4) -> list[dict]:
    articles = []
    for item in candidates:
        if len(articles) >= target:
            break
        fallback = f"{item['title']} — {item['snippet']}"
        text = _fetch_text(item["url"], fallback)
        if len(text) > 80:
            articles.append({"url": item["url"], "title": item["title"], "text": text})
    return articles


# ─── LLM call ────────────────────────────────────────────────────────────────

def _llm(prompt: str) -> Optional[dict]:
    # Routed through tm.llm → Bedrock Nova Lite (complete_text handles routing and
    # rate-limit retry). asyncio.run is fine here: this is a synchronous batch
    # script, one short-lived event loop per edge.
    try:
        raw = asyncio.run(complete_text(
            _tm_settings.extractor_model,
            prompt,
            max_tokens=350,
            temperature=0.15,
            timeout=40,
        ))
        # Strip markdown code fences if the model wraps JSON in them
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        # Use raw_decode to tolerate trailing text after the JSON object
        obj, _ = json.JSONDecoder().raw_decode(raw)
        return obj
    except Exception as exc:
        print(f"      LLM call failed: {exc}")
        return None


# ─── Per-edge calibration ─────────────────────────────────────────────────────

def calibrate(src: str, tgt: str, edge_type: str) -> Optional[dict]:
    sn, tn = NODES[src], NODES[tgt]
    tag = "[P]" if edge_type == "primary" else "[S]"
    print(f"  {tag} {src} → {tgt}  (PM: {sn['pm']:.0%} → {tn['pm']:.0%})")

    candidates = _search_articles(sn["label"], tn["label"])
    articles   = _collect_articles(candidates, target=4)
    print(f"      articles collected: {len(articles)}/{len(candidates)} candidates")

    if not articles:
        print("      ✗ no articles — skipping")
        return None

    article_block = "\n\n".join(
        f"[{i+1}] {a['title']}\n{a['text'][:1800]}"
        for i, a in enumerate(articles)
    )

    prompt = PROMPT_TEMPLATE.format(
        context=CONTEXT,
        source_label=sn["label"],
        source_pm=sn["pm"],
        target_label=tn["label"],
        target_pm=tn["pm"],
        one_minus=round(1 - sn["pm"], 4),
        article_block=article_block,
    )

    result = _llm(prompt)
    if not result or "pY" not in result or "pN" not in result:
        print(f"      ✗ bad LLM response: {result}")
        return None

    pY = round(max(0.01, min(0.99, float(result["pY"]))), 3)
    pN = round(max(0.01, min(0.99, float(result["pN"]))), 3)
    implied = round(pY * sn["pm"] + pN * (1 - sn["pm"]), 3)
    drift   = abs(implied - tn["pm"])
    flag    = " ⚠ drift>10pp" if drift > 0.10 else ""

    print(f"      pY={pY:.3f}  pN={pN:.3f}  implied={implied:.3f}  PM={tn['pm']:.3f}  Δ={drift:.3f}{flag}")
    if result.get("reasoning"):
        print(f"      → {result['reasoning'][:120]}")

    return {
        "source": src, "target": tgt, "type": edge_type,
        "pY": pY, "pN": pN,
        "implied_p": implied, "pm_p": tn["pm"],
        "drift": round(drift, 3),
        "articles_used": len(articles),
        "reasoning": result.get("reasoning", ""),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load existing results (allows resuming a partial run)
    existing: dict[str, dict] = {}
    if OUTPUT.exists():
        for r in json.loads(OUTPUT.read_text()):
            existing[f"{r['source']}→{r['target']}"] = r
        print(f"Resuming — {len(existing)} edges already done\n")

    results = list(existing.values())

    for i, (src, tgt, etype) in enumerate(EDGES, 1):
        key = f"{src}→{tgt}"
        if key in existing:
            print(f"  [skip] {key}")
            continue

        print(f"\nEdge {i}/{len(EDGES)}: {key}")
        rec = calibrate(src, tgt, etype)
        if rec:
            results.append(rec)
            existing[key] = rec
            OUTPUT.write_text(json.dumps(results, indent=2))

        time.sleep(1.2)  # rate-limit headroom

    # ── Summary ───────────────────────────────────────────────────────────────
    done  = [r for r in results if r]
    warns = [r for r in done if r["drift"] > 0.10]
    print(f"\n{'─'*60}")
    print(f"Done: {len(done)}/{len(EDGES)} edges calibrated")
    if warns:
        print(f"⚠  {len(warns)} edges with >10pp consistency drift:")
        for r in warns:
            print(f"   {r['source']}→{r['target']}  implied={r['implied_p']:.3f}  PM={r['pm_p']:.3f}")

    # ── JS patch ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("JS PATCH — paste into pm_analysis/index.html\n")

    primary   = {f"{r['source']}→{r['target']}": r for r in done if r["type"] == "primary"}
    secondary = {f"{r['source']}→{r['target']}": r for r in done if r["type"] == "secondary"}

    print("// ── NODES: update pY / pN on nodes that have a primary parent ──")
    for key, r in primary.items():
        print(f"// {key}: pY:{r['pY']}, pN:{r['pN']},  // was: see HTML")

    print("\n// ── EXTRA_EDGES: replace array ──")
    print("const EXTRA_EDGES = [")
    for key, r in secondary.items():
        src, tgt = key.split("→")
        print(f"  {{ source:'{src}', target:'{tgt}', pY:{r['pY']}, pN:{r['pN']} }},")
    print("];")

    print(f"\nFull results: {OUTPUT}")


if __name__ == "__main__":
    main()
