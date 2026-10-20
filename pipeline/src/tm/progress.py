"""
Matrix progress tracker and Rich terminal visualizer.

Displays a compact event × source grid:

         YNT HAA N12 IHY GLB KAN ...
  A01    ▓   ▓   ░   ░   ▒   ·
  A02    ░   ░   ░   ░   ░   ░
  B01    ✗   ▓   ▓   ░   ░   ░
  ...

Legend: ░ pending  ▒ in_progress  ▓ done  ✗ failed  · no predictions
"""

from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

from .models import MatrixState, CellStatus, CELL_CHAR, CELL_COLOR
from .config import settings

console = Console()

# Short 3-char abbreviations for sources
SOURCE_ABBREV: dict[str, str] = {
    "ynet":         "YNT",
    "haaretz":      "HAA",
    "n12":          "N12",
    "israel_hayom": "IHY",
    "globes":       "GLB",
    "themarker":    "TMK",
    "walla":        "WAL",
    "maariv":       "MAR",
    "jpost":        "JPO",
    "toi":          "TOI",
    "calcalist":    "CAL",
    "ch13":         "C13",
    "ch14":         "C14",
    "kurlianchik":  "KUR",
    "bbc":          "BBC",
    "aljazeera":    "AJZ",
    "cnn":          "CNN",
    "reuters":      "REU",
    "bloomberg":    "BLM",
    "wsj":          "WSJ",
    "nyt":          "NYT",
    "ft":           "FT ",
    "guardian":     "GRD",
    "wapost":       "WPO",
    "axios":        "AXS",
}


def load_state() -> MatrixState:
    p = settings.progress_file
    if p.exists():
        return MatrixState.model_validate_json(p.read_text())
    return MatrixState()


def save_state(state: MatrixState) -> None:
    state.last_updated = datetime.now().isoformat()
    settings.progress_file.parent.mkdir(parents=True, exist_ok=True)
    settings.progress_file.write_text(state.model_dump_json(indent=2))


def render_matrix(
    state: MatrixState,
    event_ids: list[str],
    source_ids: list[str],
    title: str = "TruthMachine — Matrix Progress",
) -> None:
    table = Table(
        title=title,
        box=box.SIMPLE,
        show_header=True,
        show_edge=False,
        padding=(0, 0),
    )

    # Header: event col + one col per source
    table.add_column("Event", style="bold", width=5, no_wrap=True)
    for src_id in source_ids:
        abbrev = SOURCE_ABBREV.get(src_id, src_id[:3].upper())
        table.add_column(abbrev, width=3, justify="center", no_wrap=True)

    # Rows
    for event_id in event_ids:
        row: list[Text | str] = [Text(event_id, style="bold cyan")]
        for src_id in source_ids:
            cell = state.get(event_id, src_id)
            char = CELL_CHAR[cell.status]
            color = CELL_COLOR[cell.status]
            row.append(Text(char, style=color))
        table.add_row(*row)

    # Stats bar
    stats = state.stats()
    total = len(event_ids) * len(source_ids)
    done = stats.get("done", 0) + stats.get("no_predictions", 0)
    pct = (done / total * 100) if total > 0 else 0.0

    legend = Text()
    for status in CellStatus:
        legend.append(f" {CELL_CHAR[status]} {status.value} ", style=CELL_COLOR[status])

    stats_line = (
        f"  Progress: [bold green]{done}[/bold green]/[bold]{total}[/bold] "
        f"([bold]{pct:.1f}%[/bold]) | "
        f"[green]done: {stats.get('done', 0)}[/green] | "
        f"[bright_black]no_pred: {stats.get('no_predictions', 0)}[/bright_black] | "
        f"[yellow]in_progress: {stats.get('in_progress', 0)}[/yellow] | "
        f"[red]failed: {stats.get('failed', 0)}[/red] | "
        f"pending: {stats.get('pending', 0)}"
    )

    console.print(table)
    console.print(legend)
    console.print(stats_line)


def update_cell(
    event_id: str,
    source_id: str,
    status: CellStatus,
    prediction_count: int = 0,
    error: Optional[str] = None,
    failed_attempts: Optional[int] = None,
) -> MatrixState:
    state = load_state()
    state.set_status(
        event_id,
        source_id,
        status,
        prediction_count=prediction_count,
        error=error,
    )
    if failed_attempts is not None:
        state.get(event_id, source_id).failed_attempts = failed_attempts
    save_state(state)
    return state
