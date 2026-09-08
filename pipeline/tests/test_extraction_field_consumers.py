"""Projection-completeness check (retro#808).

retro#566: nine conditional fields (`is_conditional`, `antecedent_text`, ...) were
added to `PredictionExtraction` and the extraction prompt in PR#504, but
`build_claims_detail()` in `forecast_api/forecaster.py` was never updated to copy
them into the wire projection. The model answered, the fields were silently
`None` in production for roughly six weeks, and nothing failed — the tests added
alongside #504 only exercised the extraction model itself, never the projection
step. A manual conditionals-data audit caught it, not CI.

This isn't specific to that field set: any field added to `PredictionExtraction`
or `ExtractionOutput` with no downstream reader (a projection function, the
aggregator, a settlement/subject-card consumer) is silent by construction —
the model answers, nothing stores or reads it, and CI stays green throughout.

Not exhaustive static analysis (the issue explicitly doesn't ask for that) — a
manually-maintained map from field name to where it's consumed, checked for
completeness against the model's current field list. A field missing from the
map fails this test, which forces a decision in the same PR that adds the
field: wire it to a real consumer, or record it here as a named, deliberate
shadow field (as several already are) rather than an accident nobody notices.
"""
from __future__ import annotations

from tm.models import ExtractionOutput, PredictionExtraction

# Every PredictionExtraction field, mapped to a short pointer to where it's
# consumed. "SHADOW —" entries are fields with no consumer yet, by documented
# decision (verified in the field's own description or a citing issue), not
# by omission — the retro#566 failure was never having made that decision.
PREDICTION_FIELD_CONSUMERS = {
    "quote": "build_claims_detail() -> ClaimDetail.quote",
    "claim": "build_claims_detail() -> ClaimDetail.claim",
    "stance": "build_claims_detail() -> ClaimDetail.stance; aggregator.py fusion",
    "claim_strength": "build_claims_detail() -> ClaimDetail.certainty/claim_strength",
    "settled": "build_claims_detail() -> ClaimDetail.settled; enforce_settlement_event_date",
    "quantitative_estimate": "build_claims_detail() -> ClaimDetail.quantitative_estimate; resolve_stance_certainty",
    "evidence_class": "build_claims_detail() -> ClaimDetail.evidence_class; evidence_class_weight lookup",
    "fact_signal": "build_claims_detail() -> ClaimDetail.fact_signal",
    "fact_signal_absent_reason": "build_claims_detail() -> ClaimDetail.fact_signal_absent_reason",
    "facet": "build_claims_detail() -> ClaimDetail.facet",
    "event_actors": "build_claims_detail() -> ClaimDetail.event_actors",
    "event_target": "build_claims_detail() -> ClaimDetail.event_target",
    "is_occurrence": "build_claims_detail() -> ClaimDetail.is_occurrence",
    "verified": "build_claims_detail() -> ClaimDetail.verified",
    "event_date": "build_claims_detail() -> ClaimDetail.event_date",
    "event_date_reference": "extractor.py: _resolve_relative_reference() resolves it against event_date",
    "sentiment": "aggregator.py: optional_wmean('sentiment') article-level fusion",
    "specificity": "build_claims_detail() -> ClaimDetail.specificity",
    "hedge_ratio": "aggregator.py: optional_wmean('hedge_ratio'); render_atlas.py hedge_index",
    "conditionality": "aggregator.py: optional_wmean('conditionality')",
    "magnitude": "aggregator.py: optional_wmean('magnitude')",
    "time_horizon": "aggregator.py: weighted time-horizon reduction",
    "time_horizon_days": "aggregator.py: weighted time-horizon-days reduction",
    "prediction_type": "build_claims_detail() -> ClaimDetail.prediction_type",
    "source_authority": "aggregator.py: optional_wmean('source_authority')",
    "is_conditional": "build_claims_detail() -> ClaimDetail.is_conditional",
    "antecedent_text": "build_claims_detail() -> ClaimDetail.antecedent_text",
    "antecedent_text_en": "build_claims_detail() -> ClaimDetail.antecedent_text_en",
    "antecedent_polarity": "build_claims_detail() -> ClaimDetail.antecedent_polarity",
    "relation": "build_claims_detail() -> ClaimDetail.relation",
    "strength": "build_claims_detail() -> ClaimDetail.strength",
    "stated_probability": "build_claims_detail() -> ClaimDetail.stated_probability",
    "is_counterfactual": "build_claims_detail() -> ClaimDetail.is_counterfactual",
    "speaker": "build_claims_detail() -> ClaimDetail.speaker",
    "reader_confidence": "build_claims_detail() -> ClaimDetail.reader_confidence (retro#681)",
    "report_kind": "build_claims_detail() -> ClaimDetail.report_kind (retro#686)",
    "quantity": "build_claims_detail() -> ClaimDetail.quantity (retro#683)",
    "tone": "build_claims_detail() -> ClaimDetail.tone (retro#684)",
    "voice": "build_claims_detail() -> ClaimDetail.voice (retro#684)",
}

# Every ExtractionOutput (article-level) field except `predictions` itself —
# that's the list of PredictionExtraction objects, not an elicited field.
EXTRACTION_OUTPUT_FIELD_CONSUMERS = {
    "author_lean": "forecaster.py wire projection; resolution_scorer.py author-accuracy scoring",
    "author_lean_certainty": "forecaster.py wire projection (scoring lane only, per own field description)",
    "consensus_view": "SHADOW — forecaster.py projects it onto the wire (retro#686); own field "
                       "description: 'Nothing reads it yet', consumer is the future Phase 3 S2 "
                       "shared-information detector. Projected, not dropped — the retro#566 gap "
                       "was never reaching the wire at all.",
    "claim_actor": "settlement_semantic.py: claim_subject_from_fields() (retro#697)",
    "claim_predicate": "settlement_semantic.py: claim_subject_from_fields() (retro#697)",
    "claim_scope": "settlement_semantic.py: claim_subject_from_fields() (retro#697)",
    "article_card": "subject_card.py: verify_article_card() / derive_subject_card() (retro#805)",
}


def test_every_prediction_extraction_field_has_a_known_consumer():
    fields = set(PredictionExtraction.model_fields)
    known = set(PREDICTION_FIELD_CONSUMERS)
    missing = fields - known
    assert not missing, (
        f"PredictionExtraction gained field(s) {sorted(missing)} with no entry in "
        "PREDICTION_FIELD_CONSUMERS (test_extraction_field_consumers.py). This is the "
        "retro#566 failure mode: the model will answer and nothing will structurally "
        "notice if no projection/aggregator/consumer ever reads it. Add an entry citing "
        "the real consumer, or 'SHADOW — <reason>' if this is a deliberate no-consumer-yet "
        "field (and say so in the field's own Field(description=...) too)."
    )
    stale = known - fields
    assert not stale, (
        f"PREDICTION_FIELD_CONSUMERS has stale entries for removed/renamed field(s) "
        f"{sorted(stale)} — drop them so this map keeps tracking the real schema."
    )


def test_every_extraction_output_field_has_a_known_consumer():
    fields = set(ExtractionOutput.model_fields) - {"predictions"}
    known = set(EXTRACTION_OUTPUT_FIELD_CONSUMERS)
    missing = fields - known
    assert not missing, (
        f"ExtractionOutput gained field(s) {sorted(missing)} with no entry in "
        "EXTRACTION_OUTPUT_FIELD_CONSUMERS (test_extraction_field_consumers.py) — same "
        "retro#566 risk as PREDICTION_FIELD_CONSUMERS above, for article-level fields."
    )
    stale = known - fields
    assert not stale, (
        f"EXTRACTION_OUTPUT_FIELD_CONSUMERS has stale entries for removed/renamed field(s) "
        f"{sorted(stale)} — drop them so this map keeps tracking the real schema."
    )
