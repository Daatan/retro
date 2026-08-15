"""
Pipeline runner: orchestrates gatekeeper → extraction for one article.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from rich.console import Console

from .gatekeeper import check_is_prediction
from .extractor import (
    extract_predictions,
    enforce_anchor_provenance,
    enforce_decider_intent_stance_cap,
    enforce_interested_party_certainty,
    enforce_interested_party_stance_cap,
    enforce_precursor_cap,
    enforce_relative_date_resolution,
    enforce_settlement_event_date,
    enforce_winner_entity_consistency,
    flag_claim_stance_sign_conflicts,
)
from .models import ExtractionOutput, CellStatus
from .progress import update_cell

console = Console()


@dataclass
class ArticleInput:
    text: str
    source_id: str
    source_name: str
    article_date: str
    event_id: str
    event_name: str
    event_description: str
    journalist: Optional[str] = None
    article_url: Optional[str] = None


@dataclass
class PipelineResult:
    article: ArticleInput
    is_prediction: bool
    gatekeeper_reason: str
    extraction: Optional[ExtractionOutput] = None
    error: Optional[str] = None


async def run_article(article: ArticleInput) -> PipelineResult:
    update_cell(article.event_id, article.source_id, CellStatus.in_progress)

    try:
        # Stage 1: Gatekeeper
        gate, _ = await check_is_prediction(
            article_text=article.text,
            source_name=article.source_name,
            article_date=article.article_date,
            event_name=article.event_name,
        )

        if not gate.is_prediction:
            update_cell(article.event_id, article.source_id, CellStatus.no_predictions)
            return PipelineResult(
                article=article,
                is_prediction=False,
                gatekeeper_reason=gate.reason,
            )

        # Stage 2: Extraction
        extraction, _ = await extract_predictions(
            article_text=article.text,
            source_name=article.source_name,
            article_date=article.article_date,
            event_name=article.event_name,
            event_description=article.event_description,
            journalist=article.journalist or "unknown",
        )

        # Same enforce_*/flag_* chain forecaster.py runs on the live Oracle path
        # (retro#428) — batch feeds calibration, Brier/ELO scoring, and the
        # public atlas, and a bad extraction is cached indefinitely by
        # (article_hash, event_id, prompt_version), so skipping these here let
        # already-fixed bugs (24.4% of precursor rows, 30.3% of interested-party
        # rows over-cap) back into the vault permanently. enforce_deadline_arithmetic
        # is omitted: it needs a claim_deadline/claim_direction the batch pipeline's
        # per-event schema doesn't carry (retroactive events don't pose a single
        # binary question with a direction the way a live /forecast request does).
        flag_claim_stance_sign_conflicts(extraction.predictions)
        extraction.predictions = enforce_relative_date_resolution(
            extraction.predictions, article.article_date,
        )
        extraction.predictions = enforce_settlement_event_date(
            extraction.predictions, article.article_date,
        )
        extraction.predictions = enforce_precursor_cap(extraction.predictions)
        extraction.predictions = enforce_anchor_provenance(extraction.predictions)
        extraction.predictions = enforce_interested_party_stance_cap(
            extraction.predictions,
        )
        extraction.predictions = enforce_interested_party_certainty(
            extraction.predictions,
        )
        extraction.predictions = enforce_decider_intent_stance_cap(
            extraction.predictions,
        )
        # Deterministic winner-entity check (retro#401): does the dominant
        # fact's actor→target dyad actually agree with the stance sign it
        # carries, for a two-named-actor versus/sports question? Uses the
        # #313 facets that were populated but never read before this.
        extraction.predictions = enforce_winner_entity_consistency(
            extraction.predictions, article.event_name,
        )

        update_cell(
            article.event_id,
            article.source_id,
            CellStatus.done,
            prediction_count=len(extraction.predictions),
        )

        return PipelineResult(
            article=article,
            is_prediction=True,
            gatekeeper_reason=gate.reason,
            extraction=extraction,
        )

    except Exception as e:
        error_msg = str(e)
        update_cell(article.event_id, article.source_id, CellStatus.failed, error=error_msg)
        return PipelineResult(
            article=article,
            is_prediction=False,
            gatekeeper_reason="",
            error=error_msg,
        )
