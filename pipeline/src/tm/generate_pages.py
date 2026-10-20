import json
from pathlib import Path
from typing import Dict

def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def generate_event_page(event_id: str, data_dir: Path):
    event_data = load_json(data_dir / "events" / f"{event_id}.json")
    atlas_event_dir = data_dir / "atlas" / event_id
    
    accurate_sources = []
    inaccurate_sources = []
    
    if not atlas_event_dir.exists():
        return

    for source_dir in atlas_event_dir.iterdir():
        if not source_dir.is_dir():
            continue
        
        source_id = source_dir.name
        source_meta = load_json(data_dir / "sources" / f"{source_id}.json")
        
        # Collect all entries for this source/event
        for entry_file in source_dir.glob("entry_*.json"):
            entry = load_json(entry_file)
            
            # For MVP, we assume entry has a link to extraction
            # In real run, we'd follow the pointer to the vault
            # Here we just mock the structure for the generator
            
            prediction = {
                "source_name": source_meta["name"],
                "source_url": source_meta["url"],
                "headline": entry.get("headline", "N/A"),
                "date": entry.get("article_date", "N/A"),
                "quote": entry.get("quote", "No quote available."),
                "stance": entry.get("stance", 0),
                "certainty": entry.get("certainty", 0)
            }
            
            # Simple logic: if stance matches outcome direction
            is_correct = (prediction["stance"] > 0) == event_data["outcome"]
            
            if is_correct:
                accurate_sources.append(prediction)
            else:
                inaccurate_sources.append(prediction)

    page_data = {
        "event_id": event_id,
        "event_name": event_data["name"],
        "outcome": event_data["outcome"],
        "outcome_date": event_data["outcome_date"],
        "accurate_sources": accurate_sources,
        "inaccurate_sources": inaccurate_sources,
        "analysis_summary": f"Analysis of predictions for {event_data['name']} completed."
    }
    
    output_path = data_dir / "pages" / f"{event_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(page_data, f, indent=2)
    print(f"Generated page data for {event_id}")

def main():
    root = Path(__file__).parent.parent.parent.parent
    data_dir = root / "data"
    
    events_dir = data_dir / "events"
    for event_file in events_dir.glob("*.json"):
        generate_event_page(event_file.stem, data_dir)

if __name__ == "__main__":
    main()
