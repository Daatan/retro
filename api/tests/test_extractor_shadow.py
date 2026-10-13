"""retro#921: extractor model shadow — log-only, fail-safe, no shared state."""
import asyncio
import logging
from types import SimpleNamespace

import pytest

from forecast_api import extractor_shadow as es


def _p(stance, settled=False, fs=None):
    return SimpleNamespace(stance=stance, settled=settled, fact_signal=fs)


def _ex(*preds):
    return SimpleNamespace(predictions=list(preds))


class TestSummarize:
    def test_empty_extraction(self):
        assert es.summarize(_ex()) == {"n": 0, "max_abs_stance": 0.0, "mean_abs_stance": 0.0,
                                       "settled": 0, "fs_n": 0, "net_stance": 0.0}

    def test_counts_and_magnitudes(self):
        s = es.summarize(_ex(_p(0.6, fs=0.5), _p(-0.2), _p(1.0, settled=True, fs=1.0)))
        assert s["n"] == 3 and s["settled"] == 1 and s["fs_n"] == 2
        assert s["max_abs_stance"] == 1.0
        assert s["mean_abs_stance"] == pytest.approx(0.6, abs=1e-3)
        assert s["net_stance"] == pytest.approx(1.4, abs=1e-3)


class TestRunShadow:
    async def _run(self, extract, caplog, timeout_s=5):
        caplog.set_level(logging.INFO, logger=es.logger.name)
        await es.run_shadow(
            extract=extract, shadow_model="bedrock/shadow", live_model="bedrock/live",
            live_extraction=_ex(_p(0.6, fs=0.6)), call_kwargs={"article_text": "t"},
            url="http://x/a", prediction_id="p1", timeout_s=timeout_s,
        )
        return [r.getMessage() for r in caplog.records if "event=extractor_shadow" in r.getMessage()]

    async def test_logs_side_by_side_and_isolates_state(self, caplog):
        seen = {}

        async def extract(**kw):
            seen.update(kw)
            return _ex(_p(0.3)), {"prompt_tokens": 100, "completion_tokens": 20}

        lines = await self._run(extract, caplog)
        assert len(lines) == 1
        line = lines[0]
        assert "live_n=1" in line and "shadow_n=1" in line
        assert "live_max_abs_stance=0.6" in line and "shadow_max_abs_stance=0.3" in line
        assert "shadow_prompt_tokens=100" in line and "err=" in line and "err=timeout" not in line
        # The shadow call must not touch the live memo or prompt-cache coordination.
        assert seen["model"] == "bedrock/shadow"
        assert seen["memo"] is None and seen["cache_coordinator"] is None

    async def test_error_is_logged_not_raised(self, caplog):
        async def extract(**kw):
            raise RuntimeError("boom")

        lines = await self._run(extract, caplog)
        assert len(lines) == 1 and "err=RuntimeError" in lines[0] and "shadow_n=" not in lines[0]

    async def test_timeout_is_logged_not_raised(self, caplog):
        async def extract(**kw):
            await asyncio.sleep(5)

        lines = await self._run(extract, caplog, timeout_s=0.05)
        assert len(lines) == 1 and "err=timeout" in lines[0]


async def test_fire_is_non_blocking_and_tracked():
    gate = asyncio.Event()

    async def extract(**kw):
        await gate.wait()
        return _ex(), {}

    es.fire_extractor_shadow(
        extract=extract, shadow_model="m", live_model="l", live_extraction=_ex(),
        call_kwargs={}, url="u", prediction_id=None, timeout_s=5,
    )
    await asyncio.sleep(0)
    assert len(es._INFLIGHT) == 1  # returned immediately, task kept alive
    gate.set()
    await asyncio.gather(*list(es._INFLIGHT))
    await asyncio.sleep(0)
    assert len(es._INFLIGHT) == 0


def test_off_by_default():
    from forecast_api.config import ApiSettings
    assert ApiSettings.model_fields["extractor_shadow_model"].default == ""
