"""retro#895 — extraction memo: an identical extractor call (same full prompt, schema, model)
within the TTL reuses the RAW output instead of calling Bedrock, and every post-processing step
still runs on a hit."""
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from forecast_api import extraction_memo_store as ems
from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from forecast_api.extraction_memo_store import ExtractionMemoStore
from tm import extractor
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction
from tm.web_search import SearchResult

HAIKU = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
ARGS = dict(
    article_text="The opposition bloc gained one seat, rising to 55, in the Maariv poll.",
    source_name="maariv.co.il",
    article_date="2026-09-30",
    event_name="Will the opposition bloc win 61 seats?",
    event_description="Resolves YES if the opposition bloc wins 61+ seats.",
    claim_deadline="2026-10-27",
)


def _output(stance: float = 0.3) -> ExtractionOutput:
    return ExtractionOutput(predictions=[PredictionExtraction(
        quote="The opposition bloc gained one seat, rising to 55", claim="c", stance=stance,
        certainty=0.8, specificity=1.0, settled=None, evidence_class="reported_fact",
    )])


@pytest.fixture
def memo(tmp_path):
    return ExtractionMemoStore(tmp_path / "memo", ttl_seconds=3600, schema_hash="schemaA")


@pytest.fixture
def llm(monkeypatch):
    cs = AsyncMock(side_effect=lambda *a, **k: (_output(), {"total_tokens": 1000}))
    monkeypatch.setattr(extractor, "complete_structured", cs)
    return cs


# ------------------------------------------------------------------ store


def test_key_covers_schema_model_and_prompt(tmp_path):
    a = ExtractionMemoStore(tmp_path, ttl_seconds=1, schema_hash="s1")
    b = ExtractionMemoStore(tmp_path, ttl_seconds=1, schema_hash="s2")
    base = a.key(model=HAIKU, prompt="P")
    assert base == a.key(model=HAIKU, prompt="P")
    assert base != a.key(model="bedrock/other", prompt="P")
    assert base != a.key(model=HAIKU, prompt="P ")
    assert base != b.key(model=HAIKU, prompt="P")


async def test_roundtrip_and_ttl_expiry(tmp_path):
    store = ExtractionMemoStore(tmp_path, ttl_seconds=0.2, schema_hash="s")
    await store.put("k", "v")
    assert await store.get("k") == "v"
    await asyncio.sleep(0.4)
    assert await store.get("k") is None


async def test_store_errors_fail_open(tmp_path, monkeypatch, caplog):
    store = ExtractionMemoStore(tmp_path, ttl_seconds=60, schema_hash="s")
    broken = SimpleNamespace(get=lambda *a, **k: 1 / 0, set=lambda *a, **k: 1 / 0)
    monkeypatch.setattr(ems, "_get_store", lambda path: broken)
    with caplog.at_level(logging.WARNING):
        assert await store.get("k") is None
        await store.put("k", "v")          # no raise
    assert "event=extract_memo_error op=get" in caplog.text
    assert "event=extract_memo_error op=put" in caplog.text


# ------------------------------------------------------------------ extract_predictions


async def test_miss_then_hit_skips_the_model(memo, llm, caplog):
    with caplog.at_level(logging.INFO, logger="tm.extractor"):
        out1, usage1 = await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo)
        out2, usage2 = await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo)
    assert llm.await_count == 1
    assert usage1 == {"total_tokens": 1000} and usage2 == {}
    assert out2.model_dump() == out1.model_dump()
    assert "event=extract_memo result=miss" in caplog.text
    assert "event=extract_memo result=hit" in caplog.text


async def test_stored_value_is_the_raw_output_not_the_post_processed_one(memo, llm):
    """Callers mutate extraction.predictions in place (the enforce_* chain). The memo must hold
    what the model returned, and every hit must be a fresh copy."""
    out1, _ = await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo)
    out1.predictions[0].stance = -1.0
    out1.predictions.append(out1.predictions[0])
    out2, _ = await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo)
    assert len(out2.predictions) == 1 and out2.predictions[0].stance == 0.3
    out2.predictions[0].stance = -0.5
    out3, _ = await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo)
    assert out3.predictions[0].stance == 0.3


@pytest.mark.parametrize("change", [
    {"article_text": ARGS["article_text"] + " Updated."},
    {"event_name": "Will the coalition win 61 seats?"},
    {"event_description": "Resolves YES on a different criterion."},
    {"claim_deadline": "2026-11-01"},
    {"short_form": True},
    {"language": "Hebrew"},
    {"model": "bedrock/us.anthropic.claude-sonnet-4-5"},
])
async def test_any_input_change_misses(memo, llm, change):
    await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo)
    kwargs = {**ARGS, "model": HAIKU, **change}
    await extractor.extract_predictions(**kwargs, memo=memo)
    assert llm.await_count == 2


async def test_changed_prompt_prefix_misses(memo, llm, monkeypatch):
    await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo)
    monkeypatch.setattr(extractor, "PROMPT_PREFIX", extractor.PROMPT_PREFIX + "\nNew rule.")
    await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo)
    assert llm.await_count == 2


async def test_changed_schema_hash_misses(tmp_path, llm):
    path = tmp_path / "memo"
    await extractor.extract_predictions(
        **ARGS, model=HAIKU, memo=ExtractionMemoStore(path, ttl_seconds=60, schema_hash="A"))
    await extractor.extract_predictions(
        **ARGS, model=HAIKU, memo=ExtractionMemoStore(path, ttl_seconds=60, schema_hash="B"))
    assert llm.await_count == 2


async def test_cache_split_does_not_change_the_key(memo, llm, monkeypatch):
    """The key is the full text the model sees, so a single-article call (with or without the
    retro#894 cache block) and a multi-article call on the same input share an entry."""
    from tm.config import settings as tm_settings
    monkeypatch.setattr(tm_settings, "extractor_cache_single_article", False)
    await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo, is_single_article=True)
    await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo, is_single_article=False)
    assert llm.await_count == 1


async def test_malformed_entry_is_a_miss(memo, llm):
    key = memo.key(model=HAIKU, prompt="irrelevant")  # warm the store dir
    await memo.put(key, "x")
    await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo)
    # overwrite the real entry with junk → next call re-asks the model
    real = [k for k in ems._get_store(memo.path).iterkeys() if k != key][0]
    await memo.put(real, "{not json")
    await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo)
    assert llm.await_count == 2


async def test_errors_are_not_cached(memo, monkeypatch):
    calls = {"n": 0}

    async def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("bedrock down")
        return _output(), {}
    monkeypatch.setattr(extractor, "complete_structured", flaky)
    with pytest.raises(RuntimeError):
        await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo)
    await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo)
    assert calls["n"] == 2


async def test_hit_bypasses_cache_coordinator(memo, llm):
    await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo)
    coord = extractor.CacheWriteCoordinator()
    await extractor.extract_predictions(**ARGS, model=HAIKU, memo=memo, cache_coordinator=coord)
    assert coord._claimed is False   # the hit never claimed the write slot


async def test_no_memo_is_unchanged(llm):
    await extractor.extract_predictions(**ARGS, model=HAIKU)
    await extractor.extract_predictions(**ARGS, model=HAIKU)
    assert llm.await_count == 2


# ------------------------------------------------------------------ _process_article wiring


def _sr():
    return SearchResult(
        title="A clear title about the event", url="http://x.com/1",
        snippet="A snippet long enough to clear the twenty-char fallback guard.",
        source="x.com", published_date="2026-09-30", _prefetched_text=ARGS["article_text"],
    )


async def test_post_processing_runs_on_a_hit(monkeypatch, tmp_path, llm):
    """The point of storing RAW output: on a hit, the enforce_* chain still runs and sees the
    model's original values, exactly as on the miss."""
    monkeypatch.setattr(api_settings, "extraction_memo_enabled", True)
    monkeypatch.setattr(api_settings, "extraction_memo_path", tmp_path / "memo")
    monkeypatch.setattr(api_settings, "jev_gate_enabled", False)
    monkeypatch.setattr(api_settings, "jev_shadow_enabled", False)
    monkeypatch.setattr(api_settings, "jev_class_enabled", False)
    monkeypatch.setattr(forecaster, "check_is_prediction", AsyncMock(return_value=(
        GatekeeperOutput(is_prediction=True, reason="judged", relevance_score=0.9), {})))
    seen: list[float] = []

    def deadline(preds, dl, direction):
        seen.append(preds[0].stance)
        for p in preds:              # mutate in place, as the real chain may
            p.stance = -p.stance
        return preds
    monkeypatch.setattr(forecaster, "enforce_deadline_arithmetic", deadline)

    usage: list[dict] = []
    for _ in range(2):
        await forecaster._process_article(
            _sr(), ARGS["event_name"], max_article_chars=4000, timings=[], article_debugs=[],
            claim_deadline=ARGS["claim_deadline"], usage_events=usage,
            extractor_model=HAIKU,
        )
    assert llm.await_count == 1                 # second article call was a memo hit
    assert seen == [0.3, 0.3]                   # post-processing ran both times, on raw output
    assert usage.count({"total_tokens": 1000}) == 1   # a hit spends no extractor tokens


async def test_disabled_setting_passes_no_memo(monkeypatch):
    monkeypatch.setattr(api_settings, "extraction_memo_enabled", False)
    assert forecaster._extraction_memo() is None
    monkeypatch.setattr(api_settings, "extraction_memo_enabled", True)
    monkeypatch.setattr(api_settings, "extraction_memo_ttl_hours", 168.0)
    m = forecaster._extraction_memo()
    assert m.ttl_seconds == 168 * 3600 and m.schema_hash == forecaster.EXTRACTOR_SCHEMA_HASH
