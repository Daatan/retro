"""
Pipeline runner: orchestrates gatekeeper → extraction for one article.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

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
    enforce_settlement_fact_signal_agreement,
    enforce_winner_entity_consistency,
    audit_fact_signal_sign_mismatch,
    audit_named_entity_dyad_mismatch,
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

    # A t.me post is short-form (retro#297): same one-liner as forecaster.py's live path and
    # news-indexer's rematch.py — duplicated on purpose rather than threading a flag through
    # three repos. Without it the batch pipeline judged terse Telegram posts on the long-form
    # prompts, whose ~200-word floor rejects (or confabulates on) exactly that class (retro#542).
    # Forward-only: cached extractions keep their pre-fix results — see the issue's cache caveat.
    short_form = (
        urlparse(article.article_url or "").netloc.lower().removeprefix("www.") == "t.me"
    )

    try:
        # Stage 1: Gatekeeper
        gate, _ = await check_is_prediction(
            article_text=article.text,
            source_name=article.source_name,
            article_date=article.article_date,
            event_name=article.event_name,
            short_form=short_form,
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
            short_form=short_form,
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
        # A settlement whose own fact lane points the other way is contradicting
        # itself (retro#545): 46 live rows carry `settled` with a `fact_signal`
        # of the opposite sign, 41 of them on one ACTIVE forecast. Neutralised,
        # not inverted — the sign is untrustworthy, but which of the two lanes
        # is wrong is not knowable here. Runs last, so the demotions above have
        # already taken their rows out of the net.
        extraction.predictions = enforce_settlement_fact_signal_agreement(
            extraction.predictions,
        )
        # Log-only (retro#602): the same stance/fact_signal sign disagreement,
        # but for every strong-stance claim, not just settled=True ones — 90%
        # per-row precision on a 2026-08-23 sweep, promoted from shadow to a
        # warning per #602's recommendation (not yet to enforcement). Runs
        # after the settled-only enforcement above so an already-neutralised
        # row (stance zeroed) can't also fire here.
        extraction.predictions = audit_fact_signal_sign_mismatch(
            extraction.predictions,
        )
        # Log-only (retro#545 slice ii): does a strong-stance claim about a
        # single named actor land on a fact dyad that never names them? Real
        # precision on this shape is unmeasured, so this audits rather than
        # mutates — see docs/ORACLE_VARIABLES.md.
        extraction.predictions = audit_named_entity_dyad_mismatch(
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
