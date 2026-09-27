"""retro#884 — has_conditional_language must match every CONDITIONAL_LEXICON entry,
including the multi-word ones a word-set lookup can never hit."""
import pytest

from tm.extractor import CONDITIONAL_LEXICON, has_conditional_language


@pytest.mark.parametrize("entry", sorted(CONDITIONAL_LEXICON))
def test_every_lexicon_entry_matches_on_its_own(entry):
    assert has_conditional_language(f"Analysts said {entry} the talks resume, prices will ease.")


@pytest.mark.parametrize("text", [
    "In the event of a ceasefire, the Knesset will vote.",
    "So long as the coalition holds, the budget passes.",
    "The deal stands so\nlong  as both sides comply.",   # whitespace-tolerant
])
def test_phrases_match_without_any_single_word_entry(text):
    assert has_conditional_language(text)


@pytest.mark.parametrize("text", [
    "The Knesset passed the budget yesterday.",
    "Life is independent of the event calendar.",      # "if" in life, "depends" in independent
    "So longas the typo stays, nothing.",
    "",
])
def test_non_conditional_text_does_not_match(text):
    assert not has_conditional_language(text)
