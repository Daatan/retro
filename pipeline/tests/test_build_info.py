"""Oracle build stamp on vault extraction records — retro#744.

Every vault JSON (positive extraction AND negative marker) carries
``oracle_version`` + ``git_sha`` so a record can be attributed to the code that
wrote it, not only to the prompt version. The resolver must never raise, and
the stamp must not become a re-extraction key.
"""

import json
import subprocess

import pytest

import tm.build_info as bi
import tm.orchestrator as orch_mod
from tm.config import settings as tm_settings
from tm.models import ExtractionOutput, PredictionExtraction
from tm.orchestrator import EXTRACTION_PROMPT_VERSION, Orchestrator, SearchMode
from tm.runner import PipelineResult


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tm_settings, "data_dir", tmp_path)
    monkeypatch.setattr(tm_settings, "vault_dir", tm_settings.__class__.model_fields["vault_dir"].default)
    monkeypatch.delenv("VAULT_DIR", raising=False)
    bi.git_sha.cache_clear()
    bi.oracle_version.cache_clear()
    yield
    bi.git_sha.cache_clear()
    bi.oracle_version.cache_clear()


# ── resolver ─────────────────────────────────────────────


class TestResolver:
    def test_in_repo_composes_like_version_endpoint(self):
        v = bi.oracle_version()
        assert v != "unknown"
        base, _, build = v.partition("+build.")
        assert base[0].isdigit()
        assert build.isdigit()
        sha = bi.git_sha()
        assert sha and len(sha) == 40

    def test_outside_git_never_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bi, "_REPO_ROOT", tmp_path)
        monkeypatch.setattr(bi, "_API_PYPROJECT", tmp_path / "nope.toml")
        monkeypatch.setattr(bi, "_pkg_version", lambda name: (_ for _ in ()).throw(bi.PackageNotFoundError(name)))
        assert bi.git_sha() is None
        assert bi.oracle_version() == "unknown"

    def test_git_failure_falls_back_to_base(self, monkeypatch):
        def boom(*a, **k):
            raise subprocess.TimeoutExpired("git", 2)
        monkeypatch.setattr(bi.subprocess, "run", boom)
        assert bi.git_sha() is None
        assert "+build." not in bi.oracle_version()
        assert bi.oracle_version() != ""

    def test_pyproject_fallback_when_package_missing(self, tmp_path, monkeypatch):
        (tmp_path / "api").mkdir()
        (tmp_path / "api" / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "9.9.9"\n')
        monkeypatch.setattr(bi, "_API_PYPROJECT", tmp_path / "api" / "pyproject.toml")
        monkeypatch.setattr(bi, "_pkg_version", lambda name: (_ for _ in ()).throw(bi.PackageNotFoundError(name)))
        assert bi.oracle_version().startswith("9.9.9")


# ── vault records ────────────────────────────────────────


def _event():
    return {"id": "E1", "name": "Test Event", "llm_referee_criteria": "crit", "outcome_date": "2024-12-08"}


def _raw_art():
    return {"text": "x" * 600, "published_at": "2024-12-01", "author": "Jane", "url": "https://ynet.com/a", "headline": "H"}


async def _write(tmp_path, monkeypatch, result_factory):
    orch = Orchestrator(tmp_path, mode=SearchMode.local_file)

    async def fake_runner(article, **kwargs):
        return result_factory(article)

    monkeypatch.setattr(orch_mod, "run_article", fake_runner)
    monkeypatch.setattr(orch_mod, "oracle_version", lambda: "1.4.0+build.123")
    monkeypatch.setattr(orch_mod, "git_sha", lambda: "a" * 40)
    await orch.process_article(_raw_art(), _event(), {"id": "ynet", "name": "Ynet"})
    art_hash = orch.get_article_hash(_raw_art()["text"])
    return json.loads((orch.vault_dir / "extractions" / f"{art_hash}_E1_{EXTRACTION_PROMPT_VERSION}.json").read_text())


class TestVaultStamp:
    async def test_positive_extraction_is_stamped(self, tmp_path, monkeypatch):
        pred = PredictionExtraction(quote="q", claim="c", stance=0.5, certainty=0.8)
        data = await _write(tmp_path, monkeypatch, lambda a: PipelineResult(
            article=a, is_prediction=True, gatekeeper_reason="ok",
            extraction=ExtractionOutput(predictions=[pred])))
        assert data["status"] == "done"
        assert data["oracle_version"] == "1.4.0+build.123"
        assert data["git_sha"] == "a" * 40

    async def test_negative_marker_is_stamped(self, tmp_path, monkeypatch):
        data = await _write(tmp_path, monkeypatch, lambda a: PipelineResult(
            article=a, is_prediction=False, gatekeeper_reason="not predictive"))
        assert data["status"] == "gate_rejected"
        assert data["oracle_version"] == "1.4.0+build.123"
        assert data["git_sha"] == "a" * 40

    def test_stamp_is_not_a_skip_key(self):
        """A marker from an older build with the same prompt + models still
        suppresses re-extraction — a redeploy must not re-run the LLM."""
        marker = {"status": "gate_rejected", "prompt_version": EXTRACTION_PROMPT_VERSION,
                  "extractor_model": tm_settings.extractor_model,
                  "gatekeeper_model": tm_settings.gatekeeper_model,
                  "oracle_version": "0.0.1+build.1", "git_sha": "0" * 40}
        assert Orchestrator._negative_marker_is_current(marker, tm_settings.extractor_model)
