from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""
    brave_api_key: str = ""

    # Region-explicit (retro#548): bare model IDs used to fall back to whatever
    # region boto3 resolved, silently — CloudWatch showed these bare defaults
    # landing on us-east-1 in practice, so `us.` here is a no-op for current
    # behavior, just makes it deliberate instead of accidental.
    gatekeeper_model: str = "bedrock/us.amazon.nova-micro-v1:0"
    extractor_model: str = "bedrock/us.amazon.nova-lite-v1:0"
    ground_truth_model: str = "bedrock/us.amazon.nova-lite-v1:0"

    # retro#688 — per-event extractor override for threshold-shaped batch events, where
    # a number decides the stance ("Brent crude oil exceeds $100/barrel"). PR#671
    # (retro#664 P1) measured Nova Lite returning +0.00 on every between-bounds numeric
    # case and inverting both tone traps; Phase 4's E3a (THRESHOLD_PRICING_ENABLED) needs
    # usable batch rows on exactly that class. `archetype.is_threshold_shaped` decides
    # which events qualify and says why it is a bool rather than daatan's four-way
    # claim_archetype.
    #
    # EMPTY = OFF, and that is the shipped default: merging retro#688 changes no
    # behaviour. The batch tree self-syncs to origin/main and re-execs every cycle
    # (infra/ec2_run.sh), so a non-empty default here would be a live cost change within
    # ~5 minutes of merge, unmeasured. Turning it on is one line in the box's
    # ~/truthmachine/.env (which ec2_run.sh sources into the environment) — the batch
    # equivalent of the live lane's oracle-api drop-in:
    #
    #   THRESHOLD_EXTRACTOR_MODEL=bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0
    #
    # The live lane is unaffected either way: it passes its own `model=` per request and
    # is already on Haiku via infra/oracle-api.service.d/extractor-model.conf.
    threshold_extractor_model: str = ""

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

    # ── decider-statement facet caps (retro#354 D2c, ship inert) ─────────────
    # A decider statement of intent scores as a precursor today — same
    # `enforce_precursor_cap` (0.3) as any physical precursor, symmetric for an
    # announcement ("we will act") and a denial ("we will not"). D2's design
    # calls for splitting that into two independently-calibrated magnitudes once
    # `facet` (retro#483/#485) has accrued enough real announcement/denial rows
    # to refit against — that refit is #486, explicitly not this issue. These
    # two constants exist so the code is facet-keyed and #486 has somewhere to
    # write a real number; both default to today's symmetric
    # `fact_signal_precursor_cap` so introducing them changes no output. Same
    # "ship inert, calibrate later" pattern as `max_source_share` /
    # `cluster_downweight_exponent`. NOT YET enforced anywhere — #486's job.
    decider_statement_announcement_cap: float = 0.3
    decider_statement_denial_cap: float = 0.3

    # ── interested-party stance (retro#368, lane-soundness F20) ──────────────
    # Magnitude ceiling for a claim the extractor marked `verified=false` — the
    # dominant fact is only ASSERTED by an interested party, not independently
    # reported. The prompt's interested-party rule caps certainty at 0.5 while
    # explicitly saying "full stance magnitude still applies"; but a vote's
    # location in the pool is stance ALONE (`stance_to_prob = (stance+1)/2`,
    # aggregation.py), so a certainty-only discount is weight-only and cancels
    # under normalization. This is the location-side half. The weight-side half
    # — enforcing the certainty cap the prompt already promises, violated on
    # 30.3% of live unverified rows — is retro#378.
    #
    # Keyed on `verified` DIRECTLY, not on an inferred signature. The issue
    # originally proposed keying on high-|stance| + low-certainty; the prod audit
    # (2026-08-01, retro#368) killed that: |stance| and certainty are strongly
    # COUPLED on unverified rows (corr +0.68, vs +0.79 on verified), unverified
    # rows sit LOWER on both axes, and the proposed key fired on 1 row in 1,418
    # — which was verified=true. No threshold in the sweep was both precise and
    # material.
    #
    # Same policy-lives-in-config placement as fact_signal_precursor_cap above:
    # the model judges WHETHER the fact is independently reported, code decides
    # how far an unverified assertion may push the estimate.
    interested_party_stance_cap: float = 0.3

    # ── interested-party certainty (retro#378, lane-soundness F20 weight half) ─
    # The other axis of the same rule. The prompt has taught `certainty ≤ 0.5` on
    # an unverified interested-party claim for months and nothing in code checked
    # it: measured on prod 2026-08-01/02, 56 of 185 `verified=false` rows — 30.3%
    # — carry certainty above it, max 0.733, ten sitting at exactly 0.70 with avg
    # |stance| 0.76. That is a FLOOR on the per-claim rate, not the rate: the
    # stored value is the article-level reduction while `verified` comes from the
    # dominant claim, so a lone over-cap claim diluted by in-contract siblings is
    # invisible to that count.
    #
    # Unlike interested_party_stance_cap above, this is NOT a new policy number:
    # it is the prompt's own literal (0.5), so the two cannot drift — pinned by
    # test_extractor_prompt.py. Same F9 shape: the prompt keeps teaching the rule
    # so the model still produces good numbers unaided, and code stops trusting
    # that it did.
    interested_party_certainty_cap: float = 0.5

    # ── decider-intent stance cap (retro#518, F20 sibling) ───────────────────
    # A decider's own stated future intent ("we will act" / "we will not") is a
    # capped precursor in the fact lane — enforce_precursor_cap clamps its
    # fact_signal to ±0.3 — but voted at FULL magnitude in the stance lane, the
    # lane that prices forecasts: F20 above keys on `verified=false`, and a
    # decider's on-record statement is usually verified true (it demonstrably
    # happened) or unjudged. Prod audit 2026-08-15: of 119 pool rows with
    # `is_occurrence=false` and facet announcement/denial, 71 (60%) voted above
    # 0.3, max |0.85| — every one against a fact lane already capped at ±0.3
    # (the Netanyahu/Le Monde asymmetry, elections#141). Value deliberately
    # equals fact_signal_precursor_cap (decision on retro#518, 2026-08-15): the
    # audit found violations of the stance prompt's own expectations/threats
    # contract, not evidence for a stricter number. A SEPARATE constant from
    # decider_statement_announcement/denial_cap above — those are FACT-lane
    # knobs reserved for #486's facet refit; this is stance-lane policy.
    decider_intent_stance_cap: float = 0.3

    # ── cited_probability provenance (retro#369, lane-soundness F4) ──────────
    # `cited_probability` carries the highest class weight (4.0) and forces
    # certainty, with no check on WHO produced the number: one sentence of "a
    # market prices this at 80%" in any article we crawl buys the strongest
    # evidence class in the system. The allowlist is what may claim that
    # provenance — venues whose figure a reader could go and verify. PERMANENT:
    # R5's provenance axis was rejected on 2026-08-10 (retro#479, closing #402) —
    # `evidence_class` stays a flat 5-class enum, so this list is the end state,
    # not a stopgap awaiting a field. Still do not grow it into a general source
    # registry: it exists to answer "could a reader verify this figure", which is
    # a much shorter list than "is this outlet any good".
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
    # the class is left alone. Same compute-but-don't-use shape as the
    # credibility shadow lane.
    # ENFORCED since 2026-08-01 (retro#369). Shipped in shadow first (PR #376) so
    # the firing rate and the cited sources could be measured on live traffic
    # before the demotion target was picked; the R8 cases named on the issue
    # (B5/B6/B9) were regenerated in the same commit that flipped this.
    anchor_provenance_enforced: bool = True
    # What an unattributed cited_probability becomes once enforcement is on.
    # DECIDED 2026-08-01 on retro#369: `reporting` (0.6) — "we cannot check who
    # produced this figure, so it is ordinary hedged coverage". The rejected
    # alternative was `cited_share` (1.5), which would have kept a premium above
    # `reported_fact` (1.0) on an uncheckable number — i.e. left B6's finding
    # standing, that quoting a fake figure is the cheapest way past the
    # source-trust defence. Because the demotion is a class relabel, it also stops
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
