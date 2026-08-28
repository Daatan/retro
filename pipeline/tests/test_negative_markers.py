"""Negative-result extraction markers — Daatan/docs#57 item 2.

The batch atlas loop runs every 5 minutes with --retry-empty, which re-opens
`no_predictions` cells. Before markers existed, a gate-rejected article left no
file in vault2/extractions/, so every cycle re-ran the gatekeeper LLM on the
same rejected articles. These tests pin the new contract:

  - a definitive negative outcome (gate_rejected / no_predictions) writes a
    marker file to the article's extract_path;
  - an infra error writes NOTHING (must stay retryable);
  - a marker matching the current prompt_version + extractor/gatekeeper models
    suppresses re-extraction; any mismatch (prompt or model changed) or
    --force-reextract re-runs the LLM;
  - every reader of vault2/extractions/*.json ignores markers instead of
    treating them as extractions (atlas links, poc_report, duel_report).
"""

import json

import pytest

import tm.orchestrator as orch_mod
from tm.config import settings as tm_settings
from tm.models import ExtractionOutput, PredictionExtraction, is_negative_marker
from tm.orchestrator import EXTRACTION_PROMPT_VERSION, Orchestrator, SearchMode
from tm.runner import PipelineResult


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Same isolation as test_orchestrator.py: keep progress state and the
    vault inside tmp_path, and neutralize a repo-local VAULT_DIR/.env pin."""
    monkeypatch.setattr(tm_settings, "data_dir", tmp_path)
    monkeypatch.setattr(tm_settings, "vault_dir", tm_settings.__class__.model_fields["vault_dir"].default)
    monkeypatch.delenv("VAULT_DIR", raising=False)
    yield


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _make_orch(tmp_path, **kwargs):
    return Orchestrator(tmp_path, mode=SearchMode.local_file, **kwargs)


def _event():
    return {"id": "E1", "name": "Test Event", "llm_referee_criteria": "crit", "outcome_date": "2024-12-08"}


def _source():
    return {"id": "ynet", "name": "Ynet"}


def _raw_art(text="x" * 600):
    return {"text": text, "published_at": "2024-12-01", "author": "Jane", "url": "https://ynet.com/a", "headline": "Headline"}


def _marker(status="gate_rejected", extractor_model=None, gatekeeper_model=None,
            prompt_version=EXTRACTION_PROMPT_VERSION):
    """A negative marker as the orchestrator writes it; model/version default
    to the CURRENT settings so the marker reads as fresh unless overridden."""
    return {
        "status": status,
        "extraction": None,
        "prompt_version": prompt_version,
        "extractor_model": extractor_model if extractor_model is not None else tm_settings.extractor_model,
        "gatekeeper_model": gatekeeper_model if gatekeeper_model is not None else tm_settings.gatekeeper_model,
        "gatekeeper_reason": "off-topic",
        "run_date": "2026-08-06T00:00:00",
    }


def _fake_runner(result_factory, called):
    async def fake_run_article(article_input, extractor_model=None):
        called["n"] += 1
        return result_factory(article_input)
    return fake_run_article


# ─────────────────────────────────────────────
# is_negative_marker
# ─────────────────────────────────────────────


class TestIsNegativeMarker:
    def test_true_for_both_negative_statuses(self):
        assert is_negative_marker(_marker("gate_rejected"))
        assert is_negative_marker(_marker("no_predictions"))

    def test_false_for_positive_files_old_and_new(self):
        # Pre-marker positive file: no status field at all.
        assert not is_negative_marker({"extraction": {"predictions": []}})
        # Post-marker positive file: explicit status "done".
        assert not is_negative_marker({"status": "done", "extraction": {"predictions": []}})

    def test_false_for_non_dict(self):
        assert not is_negative_marker(None)
        assert not is_negative_marker([1, 2])
        assert not is_negative_marker("gate_rejected")


# ─────────────────────────────────────────────
# Marker writing (process_article)
# ─────────────────────────────────────────────


class TestMarkerWrite:
    async def _run(self, tmp_path, monkeypatch, result_factory, **orch_kwargs):
        orch = _make_orch(tmp_path, **orch_kwargs)
        called = {"n": 0}
        monkeypatch.setattr(orch_mod, "run_article", _fake_runner(result_factory, called))
        result = await orch.process_article(_raw_art(), _event(), _source())
        art_hash = orch.get_article_hash(_raw_art()["text"])
        extract_path = orch.vault_dir / "extractions" / f"{art_hash}_E1_{EXTRACTION_PROMPT_VERSION}.json"
        return orch, called, result, extract_path

    async def test_gate_rejected_writes_marker(self, tmp_path, monkeypatch):
        orch, _, _, extract_path = await self._run(
            tmp_path, monkeypatch,
            lambda a: PipelineResult(article=a, is_prediction=False, gatekeeper_reason="not predictive"),
        )
        data = json.loads(extract_path.read_text())
        assert data["status"] == "gate_rejected"
        assert data["extraction"] is None
        assert data["prompt_version"] == EXTRACTION_PROMPT_VERSION
        assert data["extractor_model"] == tm_settings.extractor_model
        assert data["gatekeeper_model"] == tm_settings.gatekeeper_model
        assert data["gatekeeper_reason"] == "not predictive"
        assert data["run_date"]
        # A marker is not an atlas entry.
        assert not any(orch.atlas_dir.rglob("entry_*.json"))

    async def test_gate_passed_empty_extraction_writes_no_predictions_marker(self, tmp_path, monkeypatch):
        _, _, _, extract_path = await self._run(
            tmp_path, monkeypatch,
            lambda a: PipelineResult(article=a, is_prediction=True, gatekeeper_reason="ok", extraction=None),
        )
        data = json.loads(extract_path.read_text())
        assert data["status"] == "no_predictions"
        assert data["extraction"] is None

    async def test_error_writes_no_marker(self, tmp_path, monkeypatch):
        _, _, result, extract_path = await self._run(
            tmp_path, monkeypatch,
            lambda a: PipelineResult(article=a, is_prediction=False, gatekeeper_reason="", error="bedrock 500"),
        )
        assert result.error == "bedrock 500"
        assert not extract_path.exists()

    async def test_positive_extraction_gets_status_done(self, tmp_path, monkeypatch):
        pred = PredictionExtraction(quote="q", claim="c", stance=0.5, certainty=0.8)
        _, _, _, extract_path = await self._run(
            tmp_path, monkeypatch,
            lambda a: PipelineResult(article=a, is_prediction=True, gatekeeper_reason="ok",
                                     extraction=ExtractionOutput(predictions=[pred])),
        )
        data = json.loads(extract_path.read_text())
        assert data["status"] == "done"
        assert data["extraction"]["predictions"]


# ─────────────────────────────────────────────
# Marker skipping / invalidation (process_article)
# ─────────────────────────────────────────────


class TestMarkerSkip:
    def _cached_marker(self, orch, marker):
        art_hash = orch.get_article_hash(_raw_art()["text"])
        extract_path = orch.vault_dir / "extractions" / f"{art_hash}_E1_{EXTRACTION_PROMPT_VERSION}.json"
        _write_json(extract_path, marker)
        return extract_path

    async def test_matching_marker_skips_llm(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        self._cached_marker(orch, _marker())
        called = {"n": 0}
        monkeypatch.setattr(orch_mod, "run_article", _fake_runner(
            lambda a: pytest.fail("run_article must not be called"), called))
        result = await orch.process_article(_raw_art(), _event(), _source())
        assert result is None
        assert called["n"] == 0
        assert not any(orch.atlas_dir.rglob("entry_*.json"))

    @pytest.mark.parametrize("stale", [
        {"extractor_model": "bedrock/old-extractor"},
        {"gatekeeper_model": "bedrock/old-gatekeeper"},
        {"prompt_version": "v0"},
    ])
    async def test_stale_marker_reextracts(self, tmp_path, monkeypatch, stale):
        orch = _make_orch(tmp_path)
        extract_path = self._cached_marker(orch, _marker(**stale))
        called = {"n": 0}
        monkeypatch.setattr(orch_mod, "run_article", _fake_runner(
            lambda a: PipelineResult(article=a, is_prediction=False, gatekeeper_reason="still no"), called))
        await orch.process_article(_raw_art(), _event(), _source())
        assert called["n"] == 1
        # And the marker got refreshed to the current models.
        data = json.loads(extract_path.read_text())
        assert data["extractor_model"] == tm_settings.extractor_model
        assert data["prompt_version"] == EXTRACTION_PROMPT_VERSION

    async def test_force_reextract_overrides_matching_marker(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path, force_reextract=True)
        self._cached_marker(orch, _marker())
        called = {"n": 0}
        pred = PredictionExtraction(quote="q", claim="c", stance=0.5, certainty=0.8)
        monkeypatch.setattr(orch_mod, "run_article", _fake_runner(
            lambda a: PipelineResult(article=a, is_prediction=True, gatekeeper_reason="ok",
                                     extraction=ExtractionOutput(predictions=[pred])), called))
        await orch.process_article(_raw_art(), _event(), _source())
        assert called["n"] == 1

    async def test_positive_cache_without_status_still_skips_and_links(self, tmp_path, monkeypatch):
        # Backward compatibility: pre-marker positive files have no `status` field.
        orch = _make_orch(tmp_path)
        art_hash = orch.get_article_hash(_raw_art()["text"])
        extract_path = orch.vault_dir / "extractions" / f"{art_hash}_E1_{EXTRACTION_PROMPT_VERSION}.json"
        _write_json(extract_path, {
            "extraction": {"predictions": []}, "extractor_model": "m",
            "gatekeeper_model": "g", "gatekeeper_reason": "",
        })
        called = {"n": 0}
        monkeypatch.setattr(orch_mod, "run_article", _fake_runner(
            lambda a: pytest.fail("run_article must not be called"), called))
        result = await orch.process_article(_raw_art(), _event(), _source())
        assert result is None
        assert called["n"] == 0
        assert (orch.atlas_dir / "E1" / "ynet" / f"entry_{art_hash[:8]}.json").exists()


# ─────────────────────────────────────────────
# Near-duplicate reuse
# ─────────────────────────────────────────────


class TestNearDuplicateMarkers:
    async def test_near_duplicate_of_current_marker_skips(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        canonical_hash = "c" * 64
        monkeypatch.setattr(orch._simhash_idx, "find_near_duplicate", lambda text: canonical_hash)
        _write_json(orch.vault_dir / "extractions" / f"{canonical_hash}_E1_{EXTRACTION_PROMPT_VERSION}.json", _marker())
        called = {"n": 0}
        monkeypatch.setattr(orch_mod, "run_article", _fake_runner(
            lambda a: pytest.fail("run_article must not be called"), called))
        result = await orch.process_article(_raw_art(), _event(), _source())
        assert result is None
        assert called["n"] == 0
        assert not any(orch.atlas_dir.rglob("entry_*.json"))

    async def test_near_duplicate_of_stale_marker_falls_through(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        canonical_hash = "c" * 64
        monkeypatch.setattr(orch._simhash_idx, "find_near_duplicate", lambda text: canonical_hash)
        _write_json(orch.vault_dir / "extractions" / f"{canonical_hash}_E1_{EXTRACTION_PROMPT_VERSION}.json",
                    _marker(extractor_model="bedrock/old-extractor"))
        called = {"n": 0}
        monkeypatch.setattr(orch_mod, "run_article", _fake_runner(
            lambda a: PipelineResult(article=a, is_prediction=False, gatekeeper_reason="no"), called))
        await orch.process_article(_raw_art(), _event(), _source())
        assert called["n"] == 1


# ─────────────────────────────────────────────
# Readers must ignore markers
# ─────────────────────────────────────────────


class TestReadersIgnoreMarkers:
    def test_create_atlas_link_refuses_marker(self, tmp_path):
        orch = _make_orch(tmp_path)
        art_hash = "a" * 64
        extract_path = orch.vault_dir / "extractions" / f"{art_hash}_E1_{EXTRACTION_PROMPT_VERSION}.json"
        _write_json(extract_path, _marker())
        raw_art = {"headline": "H", "url": "https://ynet.com/a", "author": "A", "published_at": "2024-12-01"}
        orch.create_atlas_link("E1", "ynet", art_hash, extract_path, raw_art, event_date="2024-12-08")
        assert not (orch.atlas_dir / "E1" / "ynet" / f"entry_{art_hash[:8]}.json").exists()

    def test_poc_report_ignores_markers(self, tmp_path):
        from tm.poc_report import load_tm_predictions
        extractions = tmp_path / "vault2" / "extractions"
        _write_json(extractions / f"{'a' * 64}_E1_v1.json", {
            "extraction": {"predictions": [{"quote": "q", "claim": "c", "stance": 1.0, "certainty": 0.8}]},
        })
        _write_json(extractions / f"{'b' * 64}_E1_v1.json", _marker())
        _write_json(extractions / f"{'d' * 64}_E2_v1.json", _marker("no_predictions"))
        preds = load_tm_predictions(tmp_path)
        # Marker-only E2 contributes nothing; E1's probability comes from the
        # single real extraction (stance 1.0 -> 1.0), unpolluted by the marker.
        assert "E2" not in preds
        assert preds["E1"] == 1.0

    def test_duel_report_ignores_markers(self, tmp_path):
        from tm.duel_report import _load_vault2_articles
        extractions = tmp_path / "vault2" / "extractions"
        articles = tmp_path / "vault2" / "articles"
        good_hash, marker_hash = "a" * 64, "b" * 64
        _write_json(extractions / f"{good_hash}_E1_v1.json", {
            "extraction": {"predictions": []}, "extractor_model": "m",
        })
        _write_json(extractions / f"{marker_hash}_E1_v1.json", _marker())
        for h in (good_hash, marker_hash):
            _write_json(articles / f"{h}.json", {
                "url": f"https://ynet.com/{h[:4]}", "headline": "H",
                "published_at": "2024-12-01", "text": "body",
            })
        arts = _load_vault2_articles(tmp_path, "E1", "2024-12-07")
        assert len(arts) == 1
        assert arts[0]["url"].endswith(good_hash[:4])
