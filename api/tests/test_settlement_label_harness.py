"""retro#691 step 2 — guards on the labelling harness itself.

The labelled set only means anything if the labeller cannot see what it is
grading. `label_settlement_candidates.py` gets rows that DO carry the gate
inputs (the scorer needs them from the same export), so the blindness lives in
`build_prompt` — which makes it exactly the kind of property that decays
silently when someone adds a field to the prompt for context. Hence a test.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from label_settlement_candidates import (  # noqa: E402
    INSTRUCTIONS,
    build_prompt,
    candidate_key,
)

# A row shaped like the export, with gate inputs carrying values distinctive
# enough that leaking any of them into the prompt is unambiguous.
ROW = {
    "pid": "cmtest0001",
    "question": "Benjamin Netanyahu will be the Prime Minister of Israel on December 31, 2026.",
    "deadline": "2026-12-31",
    "url": "https://example.com/a",
    "outlet": "example.com",
    "published": "2026-08-20",
    "claim": "An Iranian plot against Yair Netanyahu was foiled in Florida.",
    "quote": "Officials said the plot was disrupted.",
    # gate inputs — none of these may reach the model
    "stance": "0.9731",
    "certainty": "0.8817",
    "actors": "ZZACTORSZZ",
    "target": "ZZTARGETZZ",
    "event_date": "2026-08-19",
    "occ": "true",
    "facet": "ZZFACETZZ",
    "cls": "ZZCLASSZZ",
}

LEAKS = ["ZZACTORSZZ", "ZZTARGETZZ", "ZZFACETZZ", "ZZCLASSZZ", "0.9731", "0.8817"]


class TestPromptBlindness:
    def test_prompt_carries_the_question_claim_and_quote(self):
        prompt = build_prompt(ROW)
        assert ROW["question"] in prompt
        assert ROW["claim"] in prompt
        assert ROW["quote"] in prompt

    @pytest.mark.parametrize("leak", LEAKS)
    def test_prompt_leaks_no_gate_input(self, leak):
        assert leak not in build_prompt(ROW)

    def test_instructions_never_name_the_gate_fields(self):
        """The fixed prefix must not teach the labeller the gates' vocabulary
        either — describing the failure in the gates' own field names would
        steer it toward the same cut the gates make."""
        lowered = INSTRUCTIONS.lower()
        for field in ("event_actors", "event_target", "is_occurrence",
                      "evidence_class", "claim_strength", "stance"):
            assert field not in lowered

    def test_is_occurrence_shaped_field_absent_even_when_true(self):
        assert "is_occurrence" not in build_prompt(ROW)


class TestCandidateKey:
    def test_key_is_stable_across_added_columns(self):
        """The key must survive the export gaining fields — labels produced
        before the gate inputs were added to the SQL have to keep matching."""
        blind = {k: ROW[k] for k in ("pid", "question", "deadline", "url",
                                     "outlet", "published", "claim", "quote")}
        assert candidate_key(blind) == candidate_key(ROW)

    def test_key_changes_when_the_claim_changes(self):
        other = dict(ROW, claim="A different claim entirely.")
        assert candidate_key(other) != candidate_key(ROW)

    def test_key_distinguishes_same_claim_on_different_questions(self):
        other = dict(ROW, pid="cmtest0002")
        assert candidate_key(other) != candidate_key(ROW)
