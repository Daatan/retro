"""
Smoke test: run the full pipeline against 3 hardcoded articles
(1 English, 1 Hebrew, 1 non-prediction) and display the matrix.

Usage:
    cd pipeline
    cp .env.example .env  # fill in OPENROUTER_API_KEY
    uv run python smoke_test.py
"""

import asyncio
from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.tm.runner import run_article, ArticleInput
from src.tm.progress import render_matrix, load_state
from src.tm.models import CellStatus

console = Console()

# --- Hardcoded test articles ---

ARTICLES = [
    ArticleInput(
        text=(
            "Benjamin Netanyahu's political future hangs in the balance as the coalition "
            "faces mounting pressure from ultra-Orthodox parties demanding draft exemptions. "
            "Senior Likud officials privately acknowledge that if the budget fails to pass by "
            "the end of March, the government will collapse within weeks. 'We are heading "
            "toward a fifth election,' one minister told this reporter on condition of "
            "anonymity. 'Netanyahu will not survive this crisis — the coalition arithmetic "
            "simply does not work.' Financial markets are already pricing in political "
            "instability, with the shekel weakening against the dollar for the third "
            "consecutive week."
        ),
        source_id="jpost",
        source_name="Jerusalem Post",
        article_date="2023-11-15",
        event_id="A17",
        event_name="Budget 2024 passes Knesset",
        event_description="Did the Israeli government pass the state budget for 2024?",
        journalist="Lahav Harkov",
        article_url="https://www.jpost.com/example",
    ),
    ArticleInput(
        text=(
            "הממשלה עומדת בפני משבר חסר תקדים. מקורות בכירים בליכוד מעריכים כי "
            "הסיכוי לאישור התקציב לפני סוף השנה עומד על פחות מ-30%. "
            "'אם נגיע לבחירות נוספות, הימין יאבד את הרוב,' אמר חבר כנסת בכיר "
            "שביקש להישאר בעילום שם. 'הבוחרים עייפים, והמשק לא יכול לסבול "
            "אי-ודאות נוספת.' הכלכלנים מזהירים כי אי-אישור תקציב עלול להוביל "
            "לירידה בדירוג האשראי של ישראל עוד לפני סוף 2024."
        ),
        source_id="haaretz",
        source_name="Haaretz",
        article_date="2023-11-20",
        event_id="A17",
        event_name="Budget 2024 passes Knesset",
        event_description="Did the Israeli government pass the state budget for 2024?",
        journalist="עמיר אורן",
    ),
    ArticleInput(
        text=(
            "The Knesset plenum held a heated debate yesterday on the state budget. "
            "Finance Minister Smotrich presented the budget figures to parliament. "
            "Opposition members walked out of the hall in protest. "
            "The session lasted four hours and concluded without a vote."
        ),
        source_id="toi",
        source_name="Times of Israel",
        article_date="2023-11-18",
        event_id="A17",
        event_name="Budget 2024 passes Knesset",
        event_description="Did the Israeli government pass the state budget for 2024?",
    ),
]

# All event/source IDs for the matrix display (subset for smoke test)
SMOKE_EVENTS = ["A16", "A17", "A18", "A19", "A20", "B01", "B02", "C08", "D02", "E06"]
SMOKE_SOURCES = ["ynet", "haaretz", "n12", "israel_hayom", "globes",
                 "jpost", "toi", "bbc", "aljazeera", "reuters"]


def print_result(result) -> None:
    article = result.article

    console.print(Rule(f"[bold]{article.source_name}[/bold] — {article.article_date}"))

    if result.error:
        console.print(f"[red]ERROR:[/red] {result.error}")
        return

    if not result.is_prediction:
        console.print(f"[bright_black]NOT A PREDICTION:[/bright_black] {result.gatekeeper_reason}")
        return

    console.print(f"[green]PREDICTION DETECTED:[/green] {result.gatekeeper_reason}")
    console.print()

    if result.extraction:
        for i, pred in enumerate(result.extraction.predictions, 1):
            t = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
            t.add_column("key", style="bold cyan", width=18)
            t.add_column("value")

            t.add_row("Claim", pred.claim)
            t.add_row("Quote", f'"{pred.quote[:120]}..."' if len(pred.quote) > 120 else f'"{pred.quote}"')
            t.add_row("Stance", f"{pred.stance:+.2f}  {'▲ bullish' if pred.stance > 0 else '▼ bearish' if pred.stance < 0 else '→ neutral'}")
            t.add_row("Certainty", f"{pred.claim_strength:.2f}  {'█' * int(pred.claim_strength * 10)}{'░' * (10 - int(pred.claim_strength * 10))}")
            t.add_row("Specificity", f"{pred.specificity:.2f}")
            t.add_row("Hedge ratio", f"{pred.hedge_ratio:.2f}")
            t.add_row("Magnitude", f"{pred.magnitude:.2f}")
            t.add_row("Time horizon", f"{pred.time_horizon}" + (f" (~{pred.time_horizon_days}d)" if pred.time_horizon_days else ""))
            t.add_row("Type", pred.prediction_type.value)
            t.add_row("Authority", f"{pred.source_authority:.2f}")

            console.print(Panel(t, title=f"Prediction #{i}", border_style="green"))


async def main() -> None:
    console.print(Panel.fit(
        "[bold white]TruthMachine Smoke Test[/bold white]\n"
        "Running 3 articles through the pipeline...",
        border_style="cyan",
    ))
    console.print()

    # Show initial matrix state
    console.print("[bold]Initial matrix state:[/bold]")
    render_matrix(load_state(), SMOKE_EVENTS, SMOKE_SOURCES)
    console.print()

    # Run pipeline for each article
    for article in ARTICLES:
        console.print(f"\n[bold yellow]Processing:[/bold yellow] {article.source_name} / {article.event_id}")
        result = await run_article(article)
        print_result(result)

    # Show final matrix state
    console.print()
    console.print(Rule("[bold]Final Matrix State[/bold]"))
    render_matrix(load_state(), SMOKE_EVENTS, SMOKE_SOURCES)


if __name__ == "__main__":
    asyncio.run(main())
