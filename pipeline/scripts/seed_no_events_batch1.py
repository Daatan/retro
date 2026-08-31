"""
Seed batch 1 of NO-outcome events (Israel/region-centric) to break the all-YES
class imbalance. Each is a well-documented non-event: confidently predicted at
the time, did not happen by outcome_date. Pending Oracul in-window validation on
the paced backfill batch — events that return no in-window coverage get pruned.

Run from repo root:  python pipeline/scripts/seed_no_events_batch1.py
"""

import json
from pathlib import Path

EVENTS_DIR = Path("data/events")

# id, name, outcome_date, keywords, criteria, category, tags
EVENTS = [
    ("A20", "Early Knesset elections called in 2024", "2024-12-31",
     ["בחירות מוקדמות 2024", "Israel early elections 2024", "government collapse"],
     "Mark False if the 37th government did not fall and no early Knesset election was called during 2024.",
     ["Israeli Politics"],
     ["coalition", "early elections", "government collapse"]),
    ("B14", "Comprehensive Gaza ceasefire & hostage deal reached", "2024-08-31",
     ["עסקת חטופים והפסקת אש כוללת", "Gaza comprehensive ceasefire hostage deal"],
     "Mark False if no comprehensive ceasefire-and-hostage agreement was concluded by this date; partial or phased talks do not count.",
     ["Gaza War", "Global"],
     ["ceasefire", "hostage deal", "negotiations"]),
    ("C10", "Saudi-Israel normalization deal signed", "2023-09-30",
     ["נורמליזציה סעודיה ישראל", "Saudi Israel normalization deal"],
     "Mark False if no formal normalization agreement was signed by this date; pledges or 'getting closer' talks do not count.",
     ["Regional Geopolitics", "Israeli Politics"],
     ["Saudi Arabia", "normalization", "diplomacy"]),
    ("C11", "Israel strikes Iran's nuclear facilities (April 2024 retaliation)", "2024-04-19",
     ["תקיפה ישראלית מתקני גרעין איראן", "Israel strike Iran nuclear facilities"],
     "Mark False if Israel's retaliation did not strike Iranian nuclear sites; the Apr 19 Isfahan strike that avoided nuclear facilities counts as False.",
     ["Regional Geopolitics", "Gaza War"],
     ["Iran", "nuclear", "retaliation"]),
    ("C13", "Israel-Hezbollah escalate to all-out war (early 2024)", "2024-01-31",
     ["מלחמה כוללת חיזבאללה", "Israel Hezbollah all out war Lebanon"],
     "Mark False if no full-scale Israel-Hezbollah war erupted by this date; limited cross-border exchanges do not count.",
     ["Regional Geopolitics", "Gaza War"],
     ["Hezbollah", "Lebanon", "escalation"]),
    ("D07", "Israel enters recession in 2024", "2024-12-31",
     ["מיתון בישראל 2024", "Israel recession 2024 GDP"],
     "Mark False if Israel did not enter a technical recession (two consecutive quarters of GDP contraction) during 2024.",
     ["Israeli Economy"],
     ["recession", "GDP", "economy"]),
]


def main() -> None:
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for eid, name, date, keywords, criteria, category, tags in EVENTS:
        path = EVENTS_DIR / f"{eid}.json"
        if path.exists():
            print(f"skip {eid} (exists)")
            continue
        record = {
            "id": eid,
            "name": name,
            "outcome": False,
            "outcome_date": date,
            "search_keywords": keywords,
            "llm_referee_criteria": criteria,
            "predictive_window_days": 30,
            "category": category,
            "tags": tags,
            "polymarket": None,
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        written += 1
        print(f"wrote {eid}  (NO)")
    total = len(list(EVENTS_DIR.glob("*.json")))
    no_count = sum(
        1 for f in EVENTS_DIR.glob("*.json")
        if json.loads(f.read_text()).get("outcome") is False
    )
    print(f"\n{written} new NO events; {total} total events; {no_count} NO outcomes")


if __name__ == "__main__":
    main()
