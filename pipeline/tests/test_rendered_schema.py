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
import asyncio
import json

import pytest
from pydantic import BaseModel, Field

from tm import llm
from tm.llm import rendered_response_schema
from tm.models import ExtractionOutput, GatekeeperOutput


class _Probe(BaseModel):
    """A docstring, which Pydantic copies into the schema and MD_JSON then sends."""

    verdict: str = Field(description="a distinctive field description ZQ7")


class _Captured(Exception):
    """Raised by the fake transport once it has the request — nothing is sent anywhere."""


def _sent_request(model, monkeypatch, *, cache: bool = True) -> dict:
    """The kwargs instructor hands litellm for one real ``complete_structured`` call.

    Captured at the litellm boundary (a fake ``acompletion`` behind a client built the
    way ``tm.llm`` builds its own) rather than by calling an instructor handler: the
    handler moved modules twice and 1.16 rewrote it, while the litellm call is the one
    seam the model's input actually crosses (retro#1012).
    """
    import instructor

    seen: dict = {}

    async def fake_acompletion(*_args, **kwargs):
        seen.update(kwargs)
        raise _Captured

    monkeypatch.setattr(llm, "client", instructor.from_litellm(fake_acompletion, mode=llm.client.mode))
    monkeypatch.setattr(llm.settings, "enable_prompt_cache", cache)
    monkeypatch.setattr(llm.settings, "prompt_cache_ttl", None)
    with pytest.raises(Exception):
        asyncio.run(llm.complete_structured(
            "bedrock/us.anthropic.claude-haiku-5-5", model, "SUFFIX",
            max_tokens=10, timeout=5, cached_prefix="PREFIX", thinking=True,
        ))
    assert seen, "instructor never reached litellm"
    return seen


def _text(content) -> str:
    """A message body as plain text — 1.17 sends the system message as text blocks."""
    if isinstance(content, str):
        return content
    return "".join(block["text"] for block in content)


def _injected_system_message(model, monkeypatch) -> str:
    system = [m for m in _sent_request(model, monkeypatch)["messages"] if m["role"] == "system"]
    assert system, "MD_JSON no longer prepends a system message"
    return _text(system[0]["content"])


class TestReproducesWhatInstructorSends:
    @pytest.mark.parametrize("model", [_Probe, GatekeeperOutput, ExtractionOutput],
                             ids=["probe", "gatekeeper", "extractor"])
    def test_the_rendered_schema_appears_verbatim_in_the_prompt(self, model, monkeypatch):
        assert rendered_response_schema(model) in _injected_system_message(model, monkeypatch)

    def test_descriptions_and_docstrings_reach_the_model(self, monkeypatch):
        """The two carriers that made retro#700 expensive, checked end to end."""
        sent = _injected_system_message(_Probe, monkeypatch)
        assert "a distinctive field description ZQ7" in sent
        assert "Pydantic copies into the schema" in sent


class TestTheWrapperTextIsPinned:
    """instructor's own words around the schema are prompt text too (retro#1012).

    1.16 replaced them ("Parse the content and return a JSON object matching this
    schema" became "As a genius expert..."); the schema hash cannot see that, so it is
    pinned here in full. If this fails after an instructor bump, the model's input
    changed: A/B it (docs/AB_HARNESS.md) before updating the expected text.
    """

    def test_the_whole_system_message(self, monkeypatch):
        schema = rendered_response_schema(_Probe)
        assert _injected_system_message(_Probe, monkeypatch) == (
            "\n            As a genius expert, your task is to understand the content and provide\n"
            "            the parsed objects in json that match the following json_schema:\n\n\n"
            f"            {schema}\n\n"
            "            Make sure to return an instance of the JSON, not the schema itself\n"
        )

    def test_the_user_turn_keeps_the_cached_prefix_and_gets_the_json_reminder(self, monkeypatch):
        user = [m for m in _sent_request(_Probe, monkeypatch)["messages"] if m["role"] == "user"]
        assert len(user) == 1
        blocks = user[0]["content"]
        assert blocks[0] == {"type": "text", "text": "PREFIX", "cache_control": {"type": "ephemeral"}}
        assert [b["text"] for b in blocks[1:]] == [
            "SUFFIX",
            "Return the correct JSON response within a ```json codeblock. not the JSON_SCHEMA",
        ]


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
