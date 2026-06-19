"""
One-off seed: materialize the H (Israeli Tech) and I (Energy) domain events from
TruthMachine/EVENTS.md that were never written to data/events/. Brings the JSON
event set from 70 up to the 85 documented in the verified seed list.

Run from the repo root:  python pipeline/scripts/seed_missing_events.py
"""

import json
from pathlib import Path

EVENTS_DIR = Path("data/events")

# domain code -> category name (as used in the existing event JSONs)
DOMAIN = {
    "A": "Israeli Politics",
    "B": "Gaza War",
    "C": "Regional Geopolitics",
    "D": "Israeli Economy",
    "E": "Global",
    "F": "Israeli Society",
    "G": "AI & Tech",
    "H": "Israeli Tech",
    "I": "Energy",
}

# (id, name, outcome, outcome_date, keywords, criteria, tag_codes, tags)
EVENTS = [
    ("H01", "Mobileye IPO on Nasdaq", True, "2022-10-26",
     ["הנפקת מובילאיי נאסדק", "Mobileye IPO Nasdaq"],
     "Must focus on the Oct 26 trading debut.", "H,D",
     ["Mobileye", "IPO", "Nasdaq"]),
    ("H02", "Intel cancels Tower Semiconductor acquisition", True, "2023-08-16",
     ["ביטול רכישת טאואר אינטל", "Intel Tower acquisition cancelled"],
     "Must focus on the Aug 16 termination announcement.", "H,D",
     ["Intel", "Tower Semiconductor", "acquisition"]),
    ("H03", "IronSource completes merger with Unity", True, "2022-11-07",
     ["מיזוג אירון סורס יוניטי", "IronSource Unity merger complete"],
     "Must focus on the Nov 7 closing of the deal.", "H",
     ["IronSource", "Unity", "merger"]),
    ("H04", "Israeli VC investment drops 50%+ YoY (2022→2023)", True, "2023-12-31",
     ["ירידה השקעות הייטק ישראל", "Israeli VC investment drop 2023"],
     "Must cite IVC/Start-Up Nation Central data showing 50%+ decline.", "H,D",
     ["VC", "investment", "decline"]),
    ("H05", "Wiz rejects Google $23B acquisition offer", True, "2024-07-22",
     ["ויז דחתה גוגל", "Wiz rejects Google acquisition"],
     "Must focus on the July 22 rejection announcement.", "H,G",
     ["Wiz", "Google", "acquisition"]),
    ("H06", "Google agrees to acquire Wiz for $32B", True, "2025-03-18",
     ["גוגל רוכשת ויז", "Google acquires Wiz $32B"],
     "Must focus on the definitive agreement announcement.", "H,G",
     ["Wiz", "Google", "acquisition"]),
    ("H07", "Check Point acquires Cyberint", True, "2024-08-01",
     ["צ'ק פוינט רוכשת סייברינט", "Check Point acquires Cyberint"],
     "Must focus on the acquisition announcement in Aug 2024.", "H",
     ["Check Point", "Cyberint", "acquisition"]),
    ("H08", "Israeli AI startup raises $100M+ round (multiple)", True, "2024-06-01",
     ["סטארטאפ ישראלי בינה מלאכותית גיוס", "Israeli AI startup $100M raise"],
     "Must cite at least one Israeli AI company closing a $100M+ round in 2024.", "H,G,D",
     ["AI startup", "funding round", "Israel"]),
    ("H09", "Israeli tech layoffs exceed 15,000 cumulative (2023)", True, "2023-12-31",
     ["פיטורים הייטק ישראל 2023", "Israeli tech layoffs 2023"],
     "Must cite cumulative layoff figures above 15,000 for 2023.", "H,D",
     ["layoffs", "tech", "Israel"]),
    ("H10", "Start-Up Nation Central: Israel drops out of top 5 startup ecosystems", True, "2024-06-01",
     ["מדד אקוסיסטם ישראל", "Israel startup ecosystem ranking drop"],
     "Must cite a specific global ranking report showing Israel below top 5.", "H,D",
     ["startup ecosystem", "ranking", "Israel"]),
    ("I01", "Brent crude oil exceeds $100/barrel (Russia-Ukraine)", True, "2022-02-24",
     ["נפט מעל 100 דולר", "Brent crude 100 dollars barrel"],
     "Must focus on the Feb 24 price spike above $100 triggered by invasion.", "I,E",
     ["Brent crude", "oil price", "Russia-Ukraine"]),
    ("I02", "Israel-EU natural gas export MOU signed", True, "2022-06-15",
     ["הסכם גז ישראל אירופה", "Israel EU gas export MOU"],
     "Must focus on the June 15 tripartite agreement (Israel, Egypt, EU).", "I,E,D",
     ["natural gas", "EU", "MOU"]),
    ("I03", "Karish gas field begins production", True, "2022-10-27",
     ["שדה קריש ייצור גז", "Karish gas field production"],
     "Must focus on the first gas production from Karish.", "I,D",
     ["Karish", "natural gas", "production"]),
    ("I04", "Europe reduces Russian gas dependency below 15% of supply", True, "2023-12-31",
     ["אירופה גז רוסי תלות", "Europe reduces Russian gas dependency"],
     "Must cite IEA or Eurostat data confirming below 15% share by end 2023.", "I,E",
     ["Russian gas", "Europe", "energy"]),
    ("I05", "Global oil price drops below $70/barrel (demand fears)", True, "2023-06-12",
     ["נפט מתחת 70 דולר", "oil price below $70 demand"],
     "Must focus on Brent closing below $70 citing global demand slowdown.", "I,E,D",
     ["oil price", "Brent", "demand"]),
]


def main() -> None:
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for eid, name, outcome, date, keywords, criteria, codes, tags in EVENTS:
        path = EVENTS_DIR / f"{eid}.json"
        if path.exists():
            print(f"skip {eid} (exists)")
            continue
        record = {
            "id": eid,
            "name": name,
            "outcome": outcome,
            "outcome_date": date,
            "search_keywords": keywords,
            "llm_referee_criteria": criteria,
            "predictive_window_days": 30,
            "category": [DOMAIN[c.strip()] for c in codes.split(",")],
            "tags": tags,
            "polymarket": None,
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        written += 1
        print(f"wrote {eid}")
    print(f"\n{written} new events written; {len(list(EVENTS_DIR.glob('*.json')))} total")


if __name__ == "__main__":
    main()
