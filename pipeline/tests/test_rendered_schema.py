"""`rendered_response_schema` must reproduce the text instructor really sends.

`llm.rendered_response_schema` (retro#700) hashes the response model's JSON schema
because MD_JSON mode puts it in the prompt. That claim is a coupling to instructor's
internals, and a hash of a string nobody sends is worse than no hash at all — it reads
as a guarantee while guaranteeing nothing. So this file checks the reproduction against
the message instructor actually builds, rather than asserting our own formatting back
to ourselves.

If an instructor upgrade changes the serialisation, these fail. That is the intended
outcome: the model's input changed, so the prompt hash should move and the version
should be bumped.
"""
import json

import pytest
from pydantic import BaseModel, Field

from tm.llm import rendered_response_schema
from tm.models import ExtractionOutput, GatekeeperOutput


def _handle_json_modes():
    """instructor's mode handler, across the module move it made mid-2.x.

    Deliberately raises rather than skipping when neither path exists: a skip here
    would silently retire the only check that the hashed text is the sent text.
    """
    try:
        from instructor.processing.response import handle_json_modes
    except ImportError:  # pragma: no cover - older instructor
        try:
            from instructor.process_response import handle_json_modes
        except ImportError as exc:  # pragma: no cover
            raise AssertionError(
                "instructor no longer exposes handle_json_modes at either known path, so "
                "tm.llm.rendered_response_schema can't be verified against what is actually "
                "sent. Re-derive the MD_JSON serialisation by hand before trusting the "
                "prompt hashes again (retro#700)."
            ) from exc
    return handle_json_modes


class _Probe(BaseModel):
    """A docstring, which Pydantic copies into the schema and MD_JSON then sends."""

    verdict: str = Field(description="a distinctive field description ZQ7")


def _injected_system_message(model) -> str:
    import instructor

    _, kwargs = _handle_json_modes()(
        model,
        {"messages": [{"role": "user", "content": "hello"}]},
        instructor.Mode.MD_JSON,
    )
    system = [m for m in kwargs["messages"] if m["role"] == "system"]
    assert system, "MD_JSON no longer prepends a system message"
    return system[0]["content"]


class TestReproducesWhatInstructorSends:
    @pytest.mark.parametrize("model", [_Probe, GatekeeperOutput, ExtractionOutput],
                             ids=["probe", "gatekeeper", "extractor"])
    def test_the_rendered_schema_appears_verbatim_in_the_prompt(self, model):
        assert rendered_response_schema(model) in _injected_system_message(model)

    def test_descriptions_and_docstrings_reach_the_model(self):
        """The two carriers that made retro#700 expensive, checked end to end."""
        sent = _injected_system_message(_Probe)
        assert "a distinctive field description ZQ7" in sent
        assert "Pydantic copies into the schema" in sent


class TestTheApproximationsWeRejected:
    """Both plausible shortcuts are measurably wrong; pin that so neither comes back."""

    def test_sorting_keys_would_hide_field_reordering(self):
        schema = ExtractionOutput.model_json_schema()
        props = list(schema["properties"])
        assert props != sorted(props), (
            "ExtractionOutput's fields happen to be alphabetical, so this test can no "
            "longer demonstrate the point — reorder the probe, don't delete the check."
        )
        reordered = dict(schema)
        reordered["properties"] = {k: schema["properties"][k] for k in reversed(props)}
        assert (json.dumps(schema, sort_keys=True)
                == json.dumps(reordered, sort_keys=True))          # sorted: blind to it
        assert (json.dumps(schema, indent=2, ensure_ascii=False)
                != json.dumps(reordered, indent=2, ensure_ascii=False))  # rendered: sees it

    def test_the_compact_form_understates_the_prompt_cost(self):
        schema = ExtractionOutput.model_json_schema()
        compact = len(json.dumps(schema))
        rendered = len(rendered_response_schema(ExtractionOutput))
        assert rendered > compact * 1.3, (
            f"rendered {rendered} vs compact {compact} — the gap is the reason the ratchet "
            "counts the rendered form"
        )
