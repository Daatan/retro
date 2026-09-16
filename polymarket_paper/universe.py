"""Which Polymarket events the paper bot follows.

First cut (decided with Mark 2026-09-16, retro#620): the Israeli Knesset
election cluster. Everything here resolves on or shortly after the 2026-10-27
vote, which is what makes it usable as a scoreboard inside the credit runway —
most Polymarket markets resolve months out.

Override with ``PAPER_EVENT_SLUGS`` (comma-separated Gamma event slugs).
"""

DEFAULT_EVENT_SLUGS: tuple[str, ...] = (
    "israeli-legislative-election-winner",
    "israel-election-likud-of-seats",
    "israel-election-will-likud-lose-seats",
    "israeli-election-results-in-a-hung-parliament",
    "will-israel-election-happen-as-scheduled-20260720204824362",
    "which-parties-will-win-a-seat-in-the-2026-knesset-elections",
    "israel-election-utj-of-seats",
    "israel-election-shas-vote-share",
    "israel-election-raam-of-seats",
    # Ends 2026-12-31 and resolves on swearing-in, not on election night — it
    # is in the universe for the mark-to-market view, not the 10-28 Brier.
    "who-will-be-the-next-prime-minister-of-israel-after-the-next-election",
)


def event_slugs_from_env(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return DEFAULT_EVENT_SLUGS
    return tuple(s.strip() for s in raw.split(",") if s.strip())
