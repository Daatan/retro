from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""
    brave_api_key: str = ""

    gatekeeper_model: str = "bedrock/amazon.nova-micro-v1:0"
    extractor_model: str = "bedrock/amazon.nova-lite-v1:0"
    ground_truth_model: str = "bedrock/amazon.nova-lite-v1:0"

    # Kill-switch for Bedrock/Anthropic prompt caching (llm.py::complete_structured's
    # cached_prefix). Verified ON: smoke_test_prompt_cache.py confirmed reliable
    # cache_read/cache_creation token accounting against live Bedrock for Nova Micro
    # (gatekeeper) and the live Haiku extractor override, using the real prompts and
    # schemas (not synthetic ones) — 4/4 clean runs each, ~90-93% of input tokens
    # landing in the cached prefix. See docs/PROMPT_CACHING.md for the full results
    # and the one unrelated finding (a pre-existing Nova Lite JSON-formatting quirk,
    # independent of caching) surfaced while verifying this.
    enable_prompt_cache: bool = True

    # Magnitude ceiling for a PRECURSOR fact (`is_occurrence=false`): the largest
    # |fact_signal| a fact that merely precedes the event may carry. The extractor
    # prompt has taught this number since the fact lane shipped, but a prompt is
    # guidance — enforce_precursor_cap (extractor.py) is what makes it true. Lives
    # here, not in the prompt, because magnitude is estimator policy: the same
    # numbers-out-of-prompts direction as the evidence_class weight table
    # (forecast_api/config.py) and retro#354's D1. Value is the prompt's own
    # literal, changed only by a deliberate policy decision. See retro#367.
    fact_signal_precursor_cap: float = 0.3

    # ── cited_probability provenance (retro#369, lane-soundness F4) ──────────
    # `cited_probability` carries the highest class weight (4.0) and forces
    # certainty, with no check on WHO produced the number: one sentence of "a
    # market prices this at 80%" in any article we crawl buys the strongest
    # evidence class in the system. The allowlist is what may claim that
    # provenance — venues whose figure a reader could go and verify. Interim by
    # design: R5 replaces it with a provenance axis and this should then be
    # DELETED, not migrated, so do not grow it into a general source registry.
    #
    # Matched case-insensitively on word boundaries against the claim's verbatim
    # `quote`. Seeded from what the live pool actually cites (prod audit
    # 2026-08-01: Opta 5 rows, Kalshi 4, Polymarket 1) plus the two integrated
    # markets and the standard national pollsters.
    cited_probability_source_allowlist: list[str] = [
        # prediction markets — already integrated, independently checkable
        "Polymarket", "Kalshi", "Metaculus", "PredictIt", "Betfair", "Smarkets",
        # named forecasting models / stats providers
        "Opta", "FiveThirtyEight", "538", "Silver Bulletin", "Elo",
        # named pollsters
        "Gallup", "Ipsos", "YouGov", "Siena", "Quinnipiac", "Marist",
        "Rasmussen", "Morning Consult", "Pew", "Datafolha", "Angus Reid",
    ]
    # Whether the demotion is APPLIED. Off = shadow: the check runs and logs
    # `event=anchor_provenance_unattributed` on every claim it would demote, but
    # the class is left alone, so prod behaviour and the R8 fixtures are
    # unchanged. Same compute-but-don't-use shape as the credibility shadow lane.
    # Flip on together with regenerating the R8 cases named on retro#369.
    anchor_provenance_enforced: bool = False
    # What an unattributed cited_probability becomes once enforcement is on.
    # PLACEHOLDER pending the policy decision on retro#369 — `reporting` (0.6)
    # says "we cannot check who produced this figure, so it is ordinary hedged
    # coverage"; `cited_share` (1.5) would keep a premium on an uncheckable
    # number. Because the demotion is a class relabel, it also stops
    # resolve_stance_certainty rewriting stance from the figure (that rewrite
    # keys on cited_probability — retro#362), which is the actual attack in R8
    # case B9.
    # Typed as the Literal rather than a bare str on purpose: a typo here would
    # otherwise put an unknown label on a claim and silently fall through to
    # evidence_class_weight_default, breaking the "evidence_class is a Literal of
    # five keys, pydantic rejects anything else at extraction time" invariant the
    # api-side weight table documents. This way a bad value fails at startup.
    unattributed_probability_class: Literal[
        "reported_fact", "cited_probability", "cited_share", "reporting", "opinion",
    ] = "reporting"

    # Optional: override API base/key (for Ollama or other OpenAI-compatible backends)
    model_api_base: str = ""
    model_api_key: str = ""

    aws_region: str = "us-east-1"

    data_dir: Path = Path("/app/data")
    vault_dir: Path = Path("")  # empty = data_dir/vault2 (avoids root-owned vault/)

    @property
    def atlas_dir(self) -> Path:
        return self.data_dir / "atlas"

    @property
    def events_dir(self) -> Path:
        return self.data_dir / "events"

    @property
    def sources_dir(self) -> Path:
        return self.data_dir / "sources"

    @property
    def progress_file(self) -> Path:
        return self.data_dir / "progress.json"


settings = Settings()
