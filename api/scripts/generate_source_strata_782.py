"""One-time (replayable) generator for `data/source_strata.json`, the outlet
lineage table retro#782 (Rule 3 of the source-dependence plan, umbrella #779)
needs to group evidence rows by (country, language) before it can discount a
stratum's over-share of pool mass.

**Derives, does not invent.** `data/sources/*.json` is already a curated,
committed, per-outlet catalog (country/language/type/etc, reviewed for other
features) — this script only re-keys it into the `source_id` space
`aggregation.py` actually groups on (the output of `forecaster._source_id_from_url`),
by resolving each catalog entry's `url` through the SAME `_DOMAIN_MAP` +
raw-domain-fallback logic the live pipeline uses. No new outlet facts are
authored here.

An entry is skipped (left for the singleton-stratum fallback the issue's own
spec describes) when:
  - `url` is null — `gdelt`/`web_search` are aggregators whose evidence rows
    carry the ARTICLE's domain as source_id, never "gdelt"/"web_search"
    itself, so a stratum entry keyed on their own id would never be looked up.
  - `country` is null — no stratum to assign.

Two outlets resolving to the SAME canonical `source_id` (e.g. `kan11.json`'s
url `kan.org.il` and `ch13.json`'s url `13tv.co.il` both match existing
`_DOMAIN_MAP` entries) collapse naturally — this is not a merge decision made
here, just the identity `_source_id_from_url` already uses in production.

Not merged: `n12.json` (url `n12.co.il`) and `mako.co.il` are the same
broadcaster editorially (the catalog's own `name` field says "N12 (Mako)"),
but `n12.co.il` is not a `_DOMAIN_MAP` key, so today's pipeline treats them as
two distinct `source_id`s ("n12.co.il" raw-domain fallback vs "mako"). This
script does not paper over that — reconciling `_DOMAIN_MAP` itself is a
separate, out-of-scope cleanup.

Usage:
    uv run python api/scripts/generate_source_strata_782.py
    # writes data/source_strata.json (repo root), overwriting it in place.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecast_api.forecaster import _source_id_from_url  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCES_DIR = REPO_ROOT / "data" / "sources"
OUTPUT_PATH = REPO_ROOT / "data" / "source_strata.json"


def build_strata() -> dict[str, str]:
    strata: dict[str, str] = {}
    for path in sorted(SOURCES_DIR.glob("*.json")):
        entry = json.loads(path.read_text())
        url = entry.get("url")
        country = entry.get("country")
        language = entry.get("language")
        if not url or not country or not language:
            continue
        source_id = _source_id_from_url(url)
        strata[source_id] = f"{country}:{language}"
    return strata


def main() -> None:
    strata = build_strata()
    payload = {
        "_generated_by": "api/scripts/generate_source_strata_782.py from data/sources/*.json — do not hand-edit, regenerate instead",
        "_stratum_key": "country:language",
        "strata": dict(sorted(strata.items())),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(strata)} source_id -> stratum mappings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
