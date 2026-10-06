"""retro#911: switching EXTRACTOR_MODEL must move the extractor and nothing else.

Every other LLM stage (settlement verifier, subject gate, premise verifier, event
decomposition, v2 matching, the /llm proxy) falls back to `settings.judge_model`. Before
retro#911 they fell back to the extractor's model, so a model switch silently moved six
stages nobody had A/B'd. This pins the one sanctioned read of the extractor setting.
"""
import re
from pathlib import Path

from forecast_api.cache import ForecastCache
from forecast_api.config import settings

SRC = Path(__file__).resolve().parents[1] / "src" / "forecast_api"


def test_extractor_model_is_read_in_exactly_one_place():
    reads = [
        f"{path.name}: {line.strip()}"
        for path in sorted(SRC.glob("*.py"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if "_pipeline_settings.extractor_model" in line
    ]
    assert reads == [
        "forecaster.py: effective_extractor_model = req.model or _pipeline_settings.extractor_model"
    ]


def test_effective_extractor_model_only_feeds_extractor_arguments():
    text = (SRC / "forecaster.py").read_text(encoding="utf-8")
    uses = re.findall(r"[^\n]*\beffective_extractor_model\b[^\n]*", text)
    stray = [
        u.strip() for u in uses
        if "extractor_model=effective_extractor_model" not in u
        and not u.strip().startswith("effective_extractor_model =")
    ]
    assert stray == []


def test_judge_model_has_a_default():
    assert settings.judge_model


def test_forecast_cache_key_separates_a_per_request_model():
    default = ForecastCache.make_key("q", 5, "h", "m")
    assert ForecastCache.make_key("q", 5, "h", "m", None) == default
    assert ForecastCache.make_key("q", 5, "h", "m", "bedrock/other") != default


def test_gatekeeper_model_is_not_read_by_api_stages_without_their_own_knob():
    # /relevance and query distillation each have a knob; their fallback to the gatekeeper's
    # model is the only sanctioned api-side read of it.
    reads = [
        line.strip()
        for path in sorted(SRC.glob("*.py"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if "_pipeline_settings.gatekeeper_model" in line
    ]
    assert all(
        r.startswith(("settings.relevance_model or", "model=settings.relevance_model or",
                      "settings.query_distill_model or"))
        or "gatekeeper_model=_pipeline_settings.gatekeeper_model" in r
        for r in reads
    ), reads
