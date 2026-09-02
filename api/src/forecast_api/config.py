from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    oracle_api_key: str  # required — startup fails with clear error if missing
    # Additional named API keys with per-key limits (docs#57 item 1) — JSON map:
    #   ORACLE_API_KEYS='{"staging": {"key": "…", "max_articles": 3}}'
    # Auth accepts these alongside oracle_api_key; a key's max_articles hard-caps
    # how many articles its forecasts may fetch AND how many caller-supplied
    # articles are accepted (the supplied-articles path bypassed every cap).
    # Empty/absent = single-key behavior, byte-for-byte unchanged. Malformed
    # JSON fails closed: named keys 401, the primary key keeps working.
    oracle_api_keys: str = ""
    openrouter_api_key: Optional[str] = None  # DEAD: /llm now routes to Bedrock via tm.llm. Kept to avoid breaking env wiring; safe to remove later.

    data_dir: Path = Path("/home/ubuntu/truthmachine/data")
    leaderboard_path: Path = Path("")  # empty = data_dir/leaderboard.json
    leaderboard_refresh_seconds: int = 86400  # pipeline writes this at most once/day
    # Credibility feedback loop, step 1 (docs/ORACLE_VARIABLES.md §9 —
    # resolution-outcome feedback loop). Append-only JSONL of resolved
    # forecasts' per-source stances, pushed by daatan via
    # POST /leaderboard/ingest.
    resolution_feedback_path: Path = Path("")  # empty = data_dir/resolution_feedback.jsonl
    # Step 3: shadow-scored output of replaying resolution_feedback_path
    # through OpenSkill on every ingest (resolution_scorer.py). Separate from
    # leaderboard_path — get_credibility_weight() never reads this file.
    resolution_leaderboard_path: Path = Path("")  # empty = data_dir/resolution_leaderboard.json
    # Author-scoring lane (author-scoring redesign, Phase 1 step 3): shadow
    # per-(author, outlet) board replayed from the same feedback file's
    # author_signals (resolution_scorer.rescore_authors_from_disk).
    resolution_author_leaderboard_path: Path = Path("")  # empty = data_dir/resolution_author_leaderboard.json
    # Settlement-pin ledger (retro#361 Phase 1): append-only JSONL of every
    # settlement pin's snapshot (what the pin said) alongside the eventual
    # resolved outcome, written on POST /leaderboard/ingest whenever the
    # payload carries a settlement_snapshot. Separate file from
    # resolution_feedback_path — this ledger is specifically about pin
    # correctness, not per-source credibility.
    settlement_pin_ledger_path: Path = Path("")  # empty = data_dir/settlement_pin_ledger.jsonl
    # Step 4 — the cutover (docs/ORACLE_VARIABLES.md §9). When True,
    # get_credibility_weight() sources credibility from the resolution-informed
    # shadow board above instead of the vault-curated leaderboard.json. Full
    # REPLACEMENT, not a blend: under the flag the vault is never consulted,
    # and a source without enough resolution history falls back to neutral 1.0
    # rather than to a 2022 backtest of 5 outlets that has been frozen since
    # 2026-03-28 (nothing in prod regenerates it — see the §8 "credibility
    # still ≈1.0" note; the vault is legacy, kept only as this flag's OFF path).
    #
    # Default False — new behaviour, unlike settlement_revalidate's default-on
    # bug fix. Flipping it is a manual step after enough real resolutions
    # accumulate AND a human reviews the backtest
    # (pipeline/scripts/backtest_shadow_credibility.py): set
    # RESOLUTION_SHADOW_CREDIBILITY_ENABLED=true in the env and restart
    # oracle-api.service. Revert is the same in reverse — no deploy either way,
    # same revert story as SETTLEMENT_REVALIDATE below.
    resolution_shadow_credibility_enabled: bool = False
    # Global gate: scoreable resolutions (resolution_scorer.count_resolutions)
    # required before ANY source's shadow score is trusted; below it every
    # source gets neutral 1.0. Originally 50 (retro#337, uncommitted sim: corr
    # ~0.97 there). retro#341 found 50 is unreachable in any useful timeframe
    # on the real claim mix (~3.7 scoreable resolutions/week) and that there
    # was no simulation data between n=6 (corr 0.81) and n=50 — so 50 wasn't
    # actually the lowest-n floor, just the only point anyone had checked.
    # pipeline/scripts/simulate_shadow_gate_correlation.py fills that gap
    # against the real weight formula: corr is already 0.91 by n=15, the
    # lowest n clearing a 0.90 bound — a deliberate trade of ~0.06 correlation
    # for reaching the gate in weeks instead of ~3 months. Revisit upward once
    # real resolution volume makes a higher n cheap again.
    resolution_shadow_min_global_predictions: int = 15
    # Credibility is derived from the source's Brier score, NOT from its
    # OpenSkill skill_conservative: sigma barely moves in these large
    # multi-team matches, so mu-3*sigma stays pinned near 0 and the vault's
    # 1.0 + conservative/25 transform maps every source to ~1.0 (measured
    # spread 1.03x on the real board, 1.09x simulated at 100 resolutions —
    # i.e. the cutover would have been a no-op). Brier separates properly
    # (spread ~2.5x at 100 resolutions) and needs no ranking counterparty.
    # The OpenSkill fields stay on the board for display/ranking only.
    #
    # Shrinkage toward the uninformed prior (Brier 0.25) instead of a minimum
    # per-source count: it degrades smoothly rather than at a cliff, so a
    # lucky new source with 2 near-perfect calls lands at ~1.08, not ~1.48.
    # Higher = more conservative. Expressed in pseudo-resolutions.
    resolution_shadow_brier_prior_n: float = 10.0
    # Slope mapping Brier distance-from-0.25 to weight. Brier 0.25 (uninformed
    # or consistently hedged) maps to exactly 1.0, so an unknown source is
    # neutral by construction; a perfect source approaches the upper clamp.
    resolution_shadow_brier_slope: float = 2.0
    # Hard bounds on the resulting weight, so no single source can dominate or
    # be zeroed out of a pool on scoring alone.
    resolution_shadow_weight_min: float = 0.25
    resolution_shadow_weight_max: float = 2.0

    # ── Shadow hazard prior (retro#356) ──────────────────────────────────
    # Absence of occurrence evidence IS evidence against a by-deadline arrival
    # claim, but no article-driven extractor will ever emit "nothing has
    # happened", so a rumor-heavy claim holds its elevated P until the deadline
    # passes. These configure a SHADOW re-drift of the pooled mean toward the
    # resolved base rate — computed alongside the live estimate and read by
    # nothing, so its Brier can be compared once resolutions accumulate. Same
    # compute-but-don't-use contract as the resolution-shadow board above.
    #
    # Off by default: this is new modelling behaviour, not a bug fix, and it
    # cannot be calibrated on the 13 resolutions on record today.
    hazard_shadow_enabled: bool = False
    # Where a diffuse claim's P drifts TO, before shrinkage. 0.15 encodes "most
    # rumored by-deadline events do not happen by the deadline". Deliberately
    # NOT 0.5: that is maximum uncertainty rather than a base rate, and would
    # RAISE P on any claim currently below it — the exact inverse of the point.
    hazard_shadow_prior_p: float = 0.15
    # Pseudo-resolutions of shrinkage toward hazard_shadow_prior_p, mirroring
    # resolution_shadow_brier_prior_n. Higher = more conservative.
    hazard_shadow_prior_n: float = 10.0
    # Fraction of the claim's own [created_at, deadline] window at which half
    # the excess over the base rate has decayed. 0.5 = half gone at the window's
    # midpoint, a quarter of the excess left at the deadline.
    hazard_shadow_half_life_fraction: float = 0.5

    max_articles: int = 10
    host: str = "127.0.0.1"
    port: int = 8001

    # Hard ceiling on a single /forecast pipeline run. The pipeline has no
    # natural deadline — slow article fetches plus Bedrock throttling/retries
    # can stack past the client's own timeout (oracle-test.html aborts at 120s),
    # leaving the caller hanging. When this fires we cancel the run and return a
    # placeholder so the caller gets a fast, clean answer. Set below the client
    # abort to guarantee we win the race.
    forecast_timeout_seconds: int = 90

    # Per-article wall-clock ceiling for the gatekeeper+extractor LLM work.
    # Articles are processed in parallel (asyncio.gather), so a single slow LLM
    # call (gatekeeper stragglers of 30s+ observed; article-phase p99 ~226s)
    # stalls the whole batch and blows past the caller's timeout. Bounding each
    # article means one straggler is dropped instead of holding up the rest.
    # Set above normal extract latency (p50 ~8s, p90 ~13s on Haiku with the v11
    # extractor, retro#697) so a 5-6 article batch does not lose every article at
    # once, but well under forecast_timeout_seconds: primary + relaxed retry
    # (2 x 35s) still fit inside the 90s outer budget.
    per_article_timeout_seconds: int = 35

    # Cap article body fed to LLMs (both the gatekeeper relevance screen and the
    # extractor). The thesis is usually in the lead, but the relevance signal can
    # sit deeper, so we keep a generous window before latency/$$ diminish returns.
    max_article_chars: int = 4000

    # Publishers whose LIVE re-fetch is known (measured, retro#520) to fail almost
    # always — paywalls/bot-challenges that serve the crawler fine at ingest but block
    # a second visit. Comma-separated bare hostnames (no scheme, no `www.`). For a
    # listed host, article fetch skips straight to news-indexer's archived-S3-text
    # lookup (news-indexer#277) instead of attempting — and paying the latency for —
    # a live fetch we already have strong evidence will fail; a miss there still falls
    # through to the normal live fetch, so a stale/wrong entry only costs a missed
    # optimization, never a correctness regression. Seeded from a full prod
    # oracle_log.txt sweep (96,587 article_fetch events, 2026-08-12): aa.com.tr 2,681
    # fallbacks, reuters.com 1,594, nytimes.com ~1,495, bloomberg.com 500, lemonde.fr
    # 501, plus msn.com/thehill.com/israelnationalnews.com/middleeasteye.net/phys.org/
    # aljazeera.net. Not auto-derived from live failure rates (yet) — static list,
    # revisit if it needs frequent tuning.
    degraded_fetch_domains: str = (
        "aa.com.tr,reuters.com,nytimes.com,bloomberg.com,lemonde.fr,msn.com,"
        "thehill.com,israelnationalnews.com,middleeasteye.net,phys.org,aljazeera.net"
    )

    # ── Aggregation (logit pooling + recency) ──────────────────────────────
    # Sources are pooled in log-odds space, weighted by credibility ×
    # evidence_class_weight × recency × relevance². (Certainty is not a factor
    # for a classified claim — see evidence_class_weight.) Recency uses
    # exponential decay with this half-life (days): an
    # article this many days old counts half as much as one published today.
    # Aggressive (7d) by design — the latest reporting should dominate as an
    # event resolves, so stale pre-resolution coverage stops diluting a decided
    # outcome.
    recency_half_life_days: float = 7.0
    # Floor for the recency weight so very old articles still count a little.
    recency_floor: float = 0.02
    # Probability clamp before taking log-odds (keeps logits finite). Also caps
    # how extreme a single pooled estimate can get: [clamp, 1-clamp].
    logit_clamp: float = 0.01
    # Each source's aggregation weight is multiplied by relevance_score² (the
    # gatekeeper's graded topic relevance). If the summed relevance mass
    # Σ relevance² across surviving articles is below this floor, the whole set is
    # treated as off-topic and the forecast returns insufficient_data
    # (reason="all_articles_off_topic") rather than pooling junk. Conservative:
    # 0.05 ≈ one article at relevance ~0.22. Tune down using daatan's logged
    # relevance_score / all_articles_off_topic data.
    relevance_weight_floor: float = 0.05
    # Per-article relevance bar for /forecast (retro#393). **0.0 = no bar, which is exactly
    # what /forecast has always done** — an article was dropped only on `not is_prediction`,
    # and its graded score then went straight into the weight.
    #
    # The bar was an entry-path property, not a verdict property: news-indexer's rescue path
    # requires `relevance_score >= 0.7` before delivering, so an article the SAME judge, same
    # model, same prompt scores 0.30 was retired permanently if it arrived via rescue and
    # VOTED if it arrived via a cosine push, a retry, or on-demand search. Measured on daatan
    # prod voting rows: 1,186 of 5,827 (20.4%) below 0.7, 220 at <=0.40; by origin
    # news-indexer 18.1%, retry 45.5%, analyze 42.1%. The gatekeeper prompt even delegates
    # explicitly — "When in doubt, PASS — the graded relevance_score below handles weak or
    # loose signal" — to a threshold that only existed in the other repo.
    #
    # Making it a setting that DEFAULTS TO THE CURRENT BEHAVIOUR does three things without
    # changing a single forecast: the number now lives in one repo instead of none, raising
    # it is a config change rather than a code change, and the effective bar is recorded per
    # response so a caller can persist which regime admitted each row.
    #
    # Deliberately NOT raised to 0.7 here. That cut is 20.4% of the voting corpus and the
    # backtest that would justify it is not powered — as of 2026-08-04 only 6 resolved BINARY
    # forecasts have a usable evidence pool. Raise it when that sample can carry a Brier
    # comparison, not before. Note also (retro#394) that the score has zero mass in
    # (0.60, 0.70], so every bar in that interval is the identical filter.
    forecast_relevance_bar: float = 0.0
    # ── Correlated-evidence clustering (retro#355) ──────────────────────────────
    # Pooling counts N articles echoing ONE development as N independent facts. These
    # two settings are the seam that fixes it; the WEIGHT side ships INERT.
    # (The settlement COUNT is the exception since retro#372: it consumes the cluster
    # assignment unconditionally — "two copies of one report are one settling source"
    # is definitional, not tuned, so it is not gated on the backtest below. The
    # jaccard threshold still governs when any of this bites.)
    #
    # `cluster_downweight_exponent` = 0.0 is the identity: a cluster of k rows carries
    # k ** (1 - exponent) = k times a single row, exactly as today. At 0.5 it carries
    # sqrt(k). Nothing moves until this is non-zero, and a test pins that property.
    #
    # Deliberately NOT enabled here, for the same reason retro#404 did not raise the
    # relevance bar: the verification path (#350 — Brier with vs without) needs resolved
    # forecasts that have a usable evidence pool, and as of 2026-08-04 there are 6. A
    # Brier comparison over 6 outcomes returns a number and that number is noise. What
    # this PR ships instead is the measurement: `event=evidence_clusters` logs the echo
    # structure of every live pool, so the decision to enable is made on observed
    # redundancy rather than on the sketch's intuition.
    #
    # `cluster_jaccard_threshold` is the shingle-overlap bar for "same story". Tuned
    # 2026-08-09 (retro#414): 0.5 was never once reached live (max observed 0.457 across
    # 24 pools), so the discount above could never fire. Eyeballing the actual pair text
    # (api/scripts/eyeball_cluster_pairs.py) at every score in [0.20, 0.50] found two
    # separate reasons NOT to go as low as the 0.457 ceiling suggests: (1) three pairs at
    # 0.22–0.27 shared boilerplate lead-in text while reporting CONTRADICTORY claims (one
    # said a Patriot-missile license was granted, another that it was refused) — clustering
    # those would suppress real disagreement, not just avoid double-counting; (2) the
    # `test_aggregation_matrix.py` fixtures' default per-source claim/quote template
    # ("Fixture claim {i} from {source}.") scores an exact, source-name-independent
    # 0.3333 against itself across different sources — a hard structural floor a lower
    # threshold would cross, wrongly clustering synthetic "independent sources" fixtures
    # (and by the same mechanism, real short boilerplate-heavy articles). 0.40 clears
    # both: it sits with real margin above the 0.3333 fixture ceiling and still catches
    # 4 unambiguous real echo pairs (0.404–0.492, all one Hormuz-deal wire story plus an
    # identical Trump quote) with zero contradiction-cluster or fixture-collision risk.
    cluster_downweight_exponent: float = 0.0
    cluster_jaccard_threshold: float = 0.40
    cluster_shingle_size: int = 3
    # ── Confusion flags (retro#687, Oracle 1.5 Phase 1 item 1.9) ─────────────
    # Reporting only — `confusion_flags.py` emits `event=confusion_flag` per
    # flagged claim plus an always-on `event=confusion_flags` summary, and
    # nothing reads either. No LLM call, no weight change; Phase 3 is where
    # flagged rows leave the credibility bill.
    #
    # The bar for rule 1's "the source is flat" half. 0.8 is the issue's own
    # number and is deliberately NOT tuned here: `claim_strength` is elicited,
    # so its live distribution per rater is one of the things the Phase 1 exit
    # report is meant to produce. Tune this from that report, not from a guess —
    # the log line carries the rule id, so a re-run at a different bar is
    # comparable against the same rows.
    confusion_flag_claim_strength_min: float = 0.8
    # ── Per-source mass cap (retro#458, Phase 1) ────────────────────────────
    # Distinct from clustering above: clustering discounts near-duplicate TEXT
    # ("two outlets wrote up one wire report"); this caps the total pooled
    # weight one SOURCE ID can hold regardless of how many distinct, non-
    # echoing articles it contributed — a single aggregator quoting five
    # different analysts is five independent-looking rows but still one
    # outlet's editorial judgment about which analysts to quote. Measured on
    # prod 2026-08-08: one live pool (the S&P-crash forecast) carried 87.4% of
    # its evidence mass from a single aggregator (finance.yahoo.com) — cluster
    # downweighting alone doesn't catch this because the rows weren't
    # near-duplicate text, just the same outlet's repeated coverage.
    #
    # `max_source_share` = 1.0 is the identity: no group's share is ever
    # capped, so this is a no-op and the shipped default. Same reasoning as
    # `cluster_downweight_exponent` above and retro#404 before it: the
    # threshold that would actually change a published forecast needs a
    # resolved-forecast backtest to justify, and as of 2026-08-08 there isn't
    # one powered enough to pick a number by. This PR ships the mechanism and
    # the observability (an `event=source_mass_capped` log line the moment it
    # would bind) so the threshold can be chosen from prod evidence, not a
    # guess — the same two-step retro#355 and retro#404 already took.
    max_source_share: float = 1.0
    # ── Harmonic per-source discount (retro#781, source-dependence Rule 2,
    # umbrella #779) ─────────────────────────────────────────────────────
    # Distinct from `max_source_share` above: that caps a group's TOTAL share
    # against a tuned threshold; this discounts a group's OWN rows against
    # each other by recency (1/i for the i-th row), a property (bounded,
    # logarithmic growth) rather than a value needing a backtest, so unlike
    # `max_source_share` it ships as a plain on/off switch, no magnitude to
    # choose. `harmonic_source_discount` = False is the identity (no-op) and
    # the shipped default — flipping it on changes already-published pool
    # weights, so a republish sweep (this repo's CLAUDE.md) is required
    # before it counts as live for existing forecasts.
    harmonic_source_discount: bool = False
    # When a caller supplies a gatekeeper verdict on an ArticleInput (relevance +
    # is_prediction — news-indexer's POST /relevance result, threaded through daatan),
    # reuse it instead of re-running check_is_prediction. The SAME claim-aware judge already
    # ran once upstream before the article was pushed; re-judging is a duplicate call whose
    # only effect is to let the push decision and the aggregation weight disagree by
    # nondeterministic noise. On by default; the flag remains as a kill switch —
    # setting REUSE_SUPPLIED_RELEVANCE=false restores re-judging without a code change.
    # Oracul-discovered (SERP/GDELT) articles carry no verdict and are always judged.
    # Design: news-indexer docs/MATCHING_ARCHITECTURE.md §3.
    reuse_supplied_relevance: bool = True
    # Caller allowlist for the reuse path above (retro#536). Comma-separated ApiKeyClient
    # names (auth.py) permitted to hand the Oracul a gatekeeper verdict. The flag above
    # says "reuse is on"; this says WHOSE verdict may be reused — without it any holder of
    # any valid API key could skip claim-aware judging for its own requests just by
    # setting relevance/is_prediction on the request body. Defaults to the primary key
    # ("default") — the daatan backend that threads news-indexer's POST /relevance result,
    # the only caller that legitimately runs an upstream gatekeeper pass — so production
    # behaviour is unchanged while named/third-party keys can no longer self-certify.
    # Fail-safe, not fail-closed: a non-allowlisted caller's verdict is dropped and the
    # article is judged normally (today's behaviour for callers that supply nothing), never
    # a 4xx. Empty string = nobody may reuse, equivalent to REUSE_SUPPLIED_RELEVANCE=false.
    relevance_reuse_allowed_clients: str = "default"
    # ── Evidence-class weighting (S2 cutover, retro docs/ORACLE_VARIABLES.md §5) ─
    # Per-claim weight component for the cross-article `weight` term, keyed by
    # PredictionExtraction.evidence_class. Replaces the old certainty-as-weight
    # term, the 0.9 certainty floor resolve_stance_certainty applied on a cited
    # quantitative_estimate, and the standalone ×4 quantitative_anchor_multiplier
    # with one lookup table (see evidence_class_weight() in aggregation.py).
    # cited_probability keeps the old ×4 premium verbatim — it protects the same
    # France World Cup regression (a 75% pooled estimate against a cited Opta
    # baseline of 18.83%; see TestFranceWorldCupRegression). The other four are
    # new calibration: reported_fact and cited_share sit below the anchor premium
    # since neither is a stated probability for the event itself; reporting and
    # opinion are down-weighted relative to the old ~0.5-0.7 typical certainty
    # range to make room for that separation. Expect retuning once more
    # real-traffic evidence_class_weighted log volume accumulates.
    evidence_class_weight: dict[str, float] = {
        "cited_probability": 4.0,
        "reported_fact": 1.0,
        "cited_share": 1.5,
        "reporting": 0.6,
        "opinion": 0.25,
    }
    # Fallback for an evidence_class string absent from the map above (defensive
    # only — PredictionExtraction.evidence_class is a Literal of the five keys
    # above, so pydantic already rejects anything else at extraction time).
    # Unclassified evidence (evidence_class is None — the extractor omitted it)
    # does NOT use this: it falls back to the claim's own certainty instead, so
    # partial classification coverage doesn't regress weighting quality for
    # claims the classifier skipped. See evidence_class_weight() in
    # aggregation.py.
    evidence_class_weight_default: float = 0.6
    # Ceiling on that certainty fallback (F10, design rule R3: missing data never
    # increases influence). Certainty [0,1] and the class table [0.25, 4.0] are
    # different scales sharing one slot, so an uncapped fallback let a confident
    # UNCLASSIFIED claim (0.95) out-weigh an identically confident claim the
    # classifier did label `reporting` (0.6) — see evidence_class_weight().
    # Set to the weakest class's weight: an unlabelled claim can tie the weakest
    # classified one, never beat it, and a hedged unlabelled claim still lands
    # below on its own certainty. Measured blast radius when introduced
    # (prod, 2026-08-01): 33 of 5729 COMPLETE pool rows are unclassified (0.6%),
    # all of them above the cap (certainty 0.38–0.74, mean 0.58).
    evidence_class_weight_unclassified_cap: float = 0.25
    # Syndication dedupe: two search results are treated as the same (re-hosted)
    # story when their title-token Jaccard is >= this. Kept high so genuinely
    # different stories sharing a topic word are not merged; only true re-prints
    # collapse to one source. 0.0 disables title clustering (URL-dedupe still runs).
    syndication_title_similarity: float = 0.8
    # Decisiveness floor: the certainty-weighted evidence mass
    # (Σ credibility·evidence_class_weight·recency·relevance² over surviving articles) at or
    # above which the pooled CI is trusted as-is. BELOW it the pool is thin/hedged,
    # so rather than abstain (which surfaced as "no AI estimate" even for on-topic
    # coverage) we *widen the CI* toward maximal uncertainty in proportion to the
    # shortfall — a thin on-topic pool then self-reports as a low-confidence estimate
    # with a wide band instead of a deceptively tight one. 0.5 ≈ one solid, on-topic,
    # confident article's worth of evidence. Abstention is now reserved for the
    # relevance floor (genuinely off-topic) and a truly empty pool. See
    # thin_evidence_ci_inflation and defer_on_thin_evidence.
    decisiveness_floor: float = 0.5
    # Maximum half-width (in probability space, [0,1]) added to the pooled 95% CI
    # when evidence_mass → 0, scaled linearly by the decisiveness shortfall
    # (deficit = (floor − mass)/floor). pool_sources' interval reflects only how much
    # the sources *disagree*, not how *much* evidence there is, so a thin pool that
    # happens to agree gets a deceptively tight CI; this term restores the missing
    # uncertainty. 0.45 ≈ a near-zero-mass on-topic pool spans almost the full [0,1].
    # 0.0 disables the widening (CI reflects dispersion only).
    thin_evidence_ci_inflation: float = 0.45
    # Escape hatch: when True, restore the old behaviour — a pool below
    # decisiveness_floor returns insufficient_data (reason="no_decisive_signal")
    # instead of emitting a wide-CI estimate. Default False (widen, don't defer).
    defer_on_thin_evidence: bool = False
    # F16 (retro#365) — minimum published interval width, expressed as a
    # between-source standard deviation on the PROBABILITY scale (the same scale
    # as pool_sources' std_p; the stance-scale equivalent is 2×, so 0.05 → 0.10).
    #
    # pool_sources derives its interval from observed DISAGREEMENT alone, so a
    # unanimous pool — or a single strong source that clears decisiveness_floor
    # on its own, so widen_ci_for_thin_evidence never fires — publishes std=0 and
    # a zero-width 95% band. widen_ci_for_unresolved_dispersion floors the band at
    # 1.96·σ/√min(n_eff, k), so the floor decays with corroboration and binds
    # exactly when std_p < σ, independent of pool size. The k term (retro#382,
    # k = Σ min(wᵢ, decisiveness_floor) / decisiveness_floor) stops equal-weight
    # row volume from buying the decay on multiplicity alone: Kish n_eff is
    # exactly the row count for equal weights, so without it fifty rows at
    # w=0.02 — one strong row's worth of mass, the funnel's fan-out shape —
    # shrank the floor 7×. Matrix case C15 pins this.
    #
    # THIS IS A POLICY NUMBER, NOT A MEASUREMENT — the same status as
    # interested_party_stance_cap = 0.3 (F20, retro#368). Nobody has measured the
    # quantity it stands for (the extractor's run-to-run stance variance). What IS
    # derived, all measured on prod 2026-08-02:
    #   • Hard ceiling 0.0765. The widest band the floor can produce is 2·1.96·σ;
    #     mcp_server._confidence buckets high/medium at width 0.30, so any
    #     σ > 0.0765 would silently demote every unanimous pool for every
    #     Polymarket trader, on a path no test covered. 0.05 → 0.196, a 35%
    #     margin, and provably cannot move a bucket on its own.
    #   • Visibility floor. daatan persists the CI as integer percent
    #     (oracle-snapshot.ts stanceToPercent) and its Telegram panel needs ≥2pp.
    #     At σ=0.05 the floored band stays ≥2pp out to n_eff≈96; at σ=0.02 it goes
    #     invisible by n_eff≈16, i.e. on the median prod pool.
    #   • Binding rate 5.2%. Over the 97 live pools with n≥3, measured
    #     between-source dispersion has min 0.0308 and p05 0.1015 (stance scale),
    #     and not one is at zero. σ=0.05 (= 0.10 stance) binds on 5 of them, plus
    #     the degenerate single-source pools it exists for. That calibration is
    #     circular — it floors a distribution at its own percentile — and its only
    #     honest content is the 5.2%.
    # To replace it with a measurement: run the extractor A/B kit (≥3 runs per
    # side) over a ~20-article sample, take the median per-article run-to-run
    # stance sd and halve it for probability space. Retune against empirical
    # interval coverage at the n≈40 calibration milestone (system-model §11).
    # 0.0 disables the floor entirely — a clean kill switch that leaves the
    # effective-N and clamp-unification halves of F16 in place.
    pool_dispersion_floor: float = 0.05
    # ── Settlement override ─────────────────────────────────────────────────
    # When at least this many independent sources carry a settlement claim (the
    # extractor's `settled` flag: the outcome is reported as an accomplished
    # fact, not a prediction) agreeing in direction, the pooled estimate is
    # pinned to ±settlement_stance and the response carries settled=true.
    # Pooling can never exceed its most confident member, so a decided event
    # otherwise tops out wherever the extractor's stances land (the Knicks
    # "82% the day after the title" case). 2 = one wire story can't settle a
    # forecast alone; 0 disables the override entirely.
    settlement_min_sources: int = 2
    # Stance the pinned estimate takes on settlement (0.94 stance = 0.97
    # probability). Deliberately short of 1.0 — sources can be wrong together.
    settlement_stance: float = 0.94
    # A settlement claim counts toward the pin only when the extractor followed
    # its own accomplished-fact rules: near-boundary stance and high certainty
    # (the prompt mandates ±1.0 / ≥0.9). Enforced in code because the 2026-07-08
    # F-35 false pin was driven by "settled" claims at stance −0.8 / certainty
    # 0.52 — hedged half-settlements the prompt should never produce. Claims
    # failing the gates are demoted to ordinary (non-settled) evidence.
    settlement_min_claim_stance: float = 0.9
    settlement_min_claim_certainty: float = 0.9
    # Re-validate every settlement vote at aggregation time against its stored
    # anchor date and the claim's window (settlement_vote_validity,
    # aggregation.py), instead of trusting the caller's settled bits — a stale
    # or poisoned pool row otherwise re-pins the estimate on every recompute
    # forever (the 2026-07-16 false-pin audit: 11 of 19 pins wrong). Also
    # replaces the majority vote with unanimity: valid settled votes in BOTH
    # directions suppress the pin (settlement_conflict) rather than letting
    # the larger side win. Kill switch: SETTLEMENT_REVALIDATE=false in the env
    # restores the legacy trust-the-flags behavior without a deploy (restart
    # oracle-api.service after changing it).
    settlement_revalidate: bool = True
    # A dated NON-occurrence settlement vote is honored at most this many days
    # past a closed claim window (settlement_vote_validity). Within the grace it
    # is the flipped late-arrival class (the Knesset dissolving July 17 against
    # a July 15 deadline settles NO — the occurrence itself proves the miss);
    # beyond it, it is the repeatable-event non-sequitur the 2026-07-19 pool
    # audit caught (July-2026 US-strikes articles settling "US bombs Iran in
    # 2025" NO at 0.93+ certainty — an out-of-window occurrence says nothing
    # about the window, and ground truth there was YES). Undated expiry votes
    # are unaffected.
    settlement_post_deadline_grace_days: int = 14
    # Evidence window (retro#545 slice iii, Gate-0 decision 2026-08-19,
    # Daatan/docs decisions.md): rows dated outside
    # ``[claim_created_at − this, claim_deadline]`` on non-scheduled archetypes
    # are excluded from the pooled estimate (still logged as
    # event=evidence_window_outside, still counted in ``n``). Shadow-only
    # 2026-08-19 through 2026-08-26 (measure first, the F4/F20 pattern);
    # enforced from the PR that added this comment, after the shadow week's
    # numbers were reviewed (retro#545 election-pool sweep, 2026-08-26: 127
    # of 2,057 recent evidence rows across 17 forecasts were outside the
    # window). 30 is deliberate: it keeps the precursor/trend coverage that
    # makes a young forecast estimable while catching the adjacent-event class
    # (an earlier, similar incident counted as the forecasted one). Set
    # negative to disable (EVIDENCE_WINDOW_LOOKBACK_DAYS=-1 in the env,
    # restart oracle-api.service).
    evidence_window_lookback_days: int = 30
    # Aggregate quality floor (retro#279): settlement_min_sources only counts
    # HOW MANY valid votes agree, not how much evidence they carry — a pool of
    # uniformly weak sources (low credibility, thin relevance, recency-decayed)
    # that each barely clear settlement grade could still out-count its way to
    # a pin. When set above 0, requires the winning direction's votes to also
    # carry at least this much combined weight (credibility · evidence_weight ·
    # recency · relevance²) — same units as decisiveness_floor's evidence_mass,
    # just scoped to the settling subset instead of the whole pool. Below it
    # the pin is suppressed (suppression_reason="settlement_quality_floor") and
    # the pooled mean stands, same as any other suppressed pin.
    #
    # ENABLED at 0.20 (retro#372), calibrated 2026-08-02 against every pin
    # production has ever published — the "tune from real pool data" this
    # comment used to defer. Method: the latest pinning snapshot per prediction
    # (29; 2 excluded as unmeasurable — their rows predate evidence_weight /
    # relevance_score being stored, so they read as 0 rather than being 0),
    # winning-direction weight reconstructed as credibility · evidence_weight ·
    # recency · relevance² with recency recomputed from published_at against
    # the snapshot's own timestamp (recency_weight is not persisted).
    #
    #   min 0.022 · p25 0.28 · median 0.60 · p75 1.65 · max 12.2
    #
    # At 0.20 exactly three of the 27 measurable pins are suppressed, and each
    # is indefensible on its face: one pinned on articles published in **2021**
    # (decayed to the 0.02 recency floor — fixture B12 in the wild), one on a
    # single settled vote in a one-row pool, one at 0.171. The next pin up sits
    # at 0.22, so the cut lands in a real gap rather than mid-cluster.
    #
    # Two things this deliberately does NOT buy, stated so the number is not
    # oversold:
    #   - It suppresses NONE of the three pins the ORACLE got wrong (0.432,
    #     2.292, 12.245 — France WC, England-Argentina, and a NO pin on "will
    #     the USA bomb Iran in 2025" whose ground truth was YES). Note that is
    #     not the same set as "predictions that resolved WRONG": a NO pin on a
    #     claim that resolved wrong was a pin the Oracul got RIGHT. Mass was
    #     not what was wrong with any of the three — sign, subject and
    #     timeframe were (retro#360, #388). A quality floor is a floor on
    #     evidence, not on relevance-of-evidence.
    #   - It suppresses the #388 Patriot pin at the snapshot where it fired
    #     (0.134) but not indefinitely: by the latest snapshot that pool's
    #     settled mass has grown to 0.297. A fixed floor delays a wrong pin
    #     that keeps accumulating corroboration; it does not prevent it.
    # Reconstruction ignores per-vote validity demotions, so the real winning
    # weight is <= the measured one and suppression is a lower bound.
    settlement_quality_floor: float = 0.20

    # The settlement match gate (retro#388/#360) — see settlement_verifier.py.
    # Before a pin publishes, one LLM call asks whether the settling facts ARE
    # the claim's own outcome: right party, right action, actually carried out
    # rather than announced. It fires only when a pin is about to fire, which
    # production has done 33 times in its whole history, so the cost is
    # structurally negligible.
    #
    # Two flags, deliberately: `enabled` runs it and LOGS the verdict,
    # `enforce` lets that verdict suppress a pin. It shipped with enforce off to
    # buy the measurement this decision needed — replay against the pins we
    # already have ground truth for — without an LLM silently changing published
    # numbers first. Every failure path is fail-OPEN (see Verdict): an
    # unreachable model must never suppress a pin by itself.
    #
    # `enforce` is ON as of 2026-08-03. The replay (scripts/replay_settlement_
    # verifier.py, all 33 pins production has ever published, 0 errors) reproduced
    # 5/5 on every pin with a known outcome — vetoing all three the Oracul got
    # WRONG and keeping both it got right — and every one of the 11 vetoes it
    # casts on the 27 still-active pins was reviewed individually and is
    # defensible: settled on 2021 articles, on a 2022 election, on an earthquake
    # in Egypt, on nine countries mistaken for the EU, or on an action announced
    # but not carried out. See docs/ORACLE_VARIABLES.md §2.1.
    #
    # What tips the risk calculation is the asymmetry: a veto is a DEMOTION, not
    # a deletion. Enforcement re-runs the same aggregate_pool with the vetoed
    # rows' settled flags cleared, so those rows keep voting as ordinary
    # evidence. A false pin publishes a confidently wrong 97% over a pool reading
    # 44%; a false veto merely publishes the pooled estimate — less confident,
    # not wrong.
    # ── Deterministic semantic gates (retro#691) — SHADOW ONLY ───────────────
    # The verifier above is one LLM call that fails open by design, and it is
    # the ONLY check asking whether a settling fact is the claim's own event:
    # every `enforce_*` guard and `settlement_vote_validity` reason is temporal.
    # In prod logs 106 of its blocks carried >=2 votes, i.e. had already cleared
    # every deterministic guard. These gates are the deterministic second
    # opinion.
    #
    # They LOG and do nothing else. There is deliberately no `_enforce` knob:
    # an enforcement path nobody has run is an inert seam that reads as shipped,
    # and enforcing this belongs in its own reviewed change once a week of
    # shadow data exists. `_enabled` is the kill switch, no deploy needed.
    #
    # Default set, measured against 387 independently labelled (question,
    # settled claim) pairs from the prod pool — 0.79 precision, 0.47 recall,
    # 26 -> 19 pins, 2 defensible pins lost:
    #   point_in_time           0.73 prec, 0 pins lost
    #   occurrence_consistency  1.00 prec, 0 pins lost
    #   facet_missing           0.80 prec, 2 pins lost
    # `predicate_echo` is left OUT of the default: it scores 0.68/0.46 on a
    # regex proxy for the claim side and overlaps facet_missing heavily (86 and
    # 82 catches alone, 111 together). It is worth turning on once retro#697
    # supplies a real dyad, not before. `announcement_facet` is REFUTED — it
    # destroys 13 of 26 defensible pins; never add it here.
    settlement_semantic_gates_enabled: bool = True
    settlement_semantic_gates: str = "point_in_time,occurrence_consistency,facet_missing"

    settlement_verifier_enabled: bool = True
    settlement_verifier_enforce: bool = True
    settlement_verifier_timeout_seconds: int = 12
    # The extractor's model, not the gatekeeper's: this is a semantic judgment
    # about aspect and role, which is exactly the class Nova Lite failed on in
    # the adjacent-event A/B (see docs/ORACLE_VARIABLES.md). Prod's oracle-api
    # drop-in overrides extractor_model to Claude Haiku 4.5, so this follows it.
    settlement_verifier_model: Optional[str] = None
    # retro#532 — decide each verdict ONCE, then remember it. The gate's
    # verdict is not idempotent even at temperature=0 (6 of the 13 questions
    # it ever saw twice returned both verdicts on an unchanged vote-set), and
    # daatan's `settled` latch is one-way, so re-rolling on every recompute is
    # a ratchet: a question the gate mostly vetoes still pins permanently on
    # its first lucky YES. Three knobs:
    #   - `votes`: samples per FIRST decision, majority wins (variance
    #     reduction exactly where it is cheap — once per vote-set, not per
    #     recompute). Keep it odd; a tie or an errored sample leaves the roll
    #     undecided (fail-open for this recompute, nothing cached). 1 =
    #     single-shot, the pre-#532 behaviour.
    #   - `cache_enabled`: the verdict store (settlement_verdict_store.py).
    #     Keyed on the built prompt + model + sample count + the settlement
    #     config fingerprint; BOTH verdict directions are sticky until the
    #     vote-set, config, model or prompt changes. Kill switch, no deploy —
    #     off restores the legacy roll-every-time behaviour.
    #   - `cache_path`: a diskcache DIRECTORY (not a file), default under
    #     data_dir so verdicts survive reloads and deploys like the pin
    #     ledger does. Deleting the directory on the box is the manual
    #     invalidation lever.
    settlement_verifier_votes: int = 3
    settlement_verdict_cache_enabled: bool = True
    settlement_verdict_cache_path: Path = Path("")  # empty = data_dir/settlement_verdict_cache

    # WHO/WHAT/SCOPE decomposition injected into the extractor's event_description
    # input (retro#758). Measured on the deadline/denial sentinel: Nova Lite
    # 11/15 -> 15/15, zero prompt/schema change (the decomposition is data
    # appended to event_description, not PROMPT_PREFIX/PROMPT_SUFFIX). Ships OFF
    # by default — retro#758's own proposal calls for measuring on the retro#691
    # 387-pair labelled adjacency set before wider rollout, which the sentinel
    # alone does not establish. Same knob shape as threshold_extractor_model
    # (pipeline/src/tm/config.py): exists, off, one line to enable.
    inject_event_decomposition: bool = False
    # None = settlement_verifier_model, which already follows the live
    # extractor (Haiku 4.5 via the oracle-api drop-in) — see that setting's
    # comment. A decomposition call is the same class of semantic judgment.
    event_decomposition_model: Optional[str] = None
    event_decomposition_timeout_seconds: int = 15
    event_decomposition_cache_enabled: bool = True
    event_decomposition_cache_path: Path = Path("")  # empty = data_dir/event_decomposition_cache

    # The premise verifier (retro#575 slice 1) — shadow/log-only, off by
    # default. Asks whether a question's premise is already dead (resolved
    # or structurally impossible) before pricing it. `enforce` is unread this
    # slice; it exists only as a placeholder for the follow-up that will act
    # on the verdict once trigger/precision data justifies it, same
    # shadow-then-promote shape settlement_verifier itself used.
    premise_verifier_enabled: bool = False
    premise_verifier_enforce: bool = False
    premise_verifier_model: Optional[str] = None
    premise_verifier_timeout_seconds: int = 12

    # Precursor candidate-match (retro#608) — shadow/log-only, off by default.
    # Before pricing a v2-playground precursor fresh, checks whether it already
    # matches an open forecast in Daatan's own bank or a live Polymarket market,
    # and logs a typed relation verdict (alias/nested/complement/implies/
    # independent) without changing pricing, recursion, or the propagated result.
    # `enforce` is unread this slice; same shadow-then-promote placeholder shape
    # premise_verifier_enforce uses, for the follow-up that decides whether to act
    # on the verdict once precision data justifies it.
    precursor_match_enabled: bool = False
    precursor_match_enforce: bool = False
    precursor_match_model: Optional[str] = None
    precursor_match_timeout_seconds: int = 12

    # Settled-grounding (retro#609) — shadow/log-only, off by default. No LLM or
    # network call: node["flat"]["settled"] is a free-to-compute signal
    # _price_flat already produces (a majority of the pool's claims were
    # already-decided fact, not forecast) and today discards. This logs what
    # would happen if it were used to lock a node, for correlation against
    # premise_verifier's own shadow log (retro#601) before either is promoted —
    # see the issue's own point 3 against duplicating that check.
    settled_grounding_enabled: bool = False
    settled_grounding_enforce: bool = False

    # Retry-relaxed-search fallback ladder rung 1 (retro#621) — shadow/log-only,
    # off by default. When the primary /forecast pass returns
    # insufficient_data (no_usable_predictions and friends), re-runs the same
    # question once with a wider article limit and logs whether the retry
    # would have recovered a usable forecast. `enforce` is unread while False;
    # same shadow-then-promote placeholder shape premise_verifier_enforce
    # uses — it exists so this can be turned on for real once the shadow log
    # shows the retry actually recovers forecasts worth the extra spend.
    retry_relaxed_search_enabled: bool = False
    retry_relaxed_search_enforce: bool = False
    retry_relaxed_search_limit_multiplier: float = 2.0

    # Forecast-response cache keyed by sha256(question, max_articles).
    # cache_ttl_seconds=0 disables caching entirely.
    cache_ttl_seconds: int = 3600
    # diskcache directory shared by both gunicorn workers (retro#405). The unit
    # sets no PrivateTmp, so this survives reloads and restarts, clearing only
    # on reboot — all fine at a 1 h TTL. The byte bound replaces the old
    # cache_max_entries count bound; eviction is LRU.
    cache_dir: str = "/tmp/oracle-forecast-cache"
    cache_size_limit_mb: int = 128

    # Search-result cache keyed by sha256(question, limit). Longer TTL than
    # forecast cache — article lists for a given query are stable for hours.
    # search_cache_ttl_seconds=0 disables search caching.
    search_cache_ttl_seconds: int = 14400  # 4 hours
    search_cache_max_entries: int = 256

    # ── MCP server / OAuth 2.1 (Cognito) ───────────────────────────────────
    # The /mcp endpoint (docs/ORACLE_MCP.md) is a Model Context Protocol server
    # exposing the Oracul's tools to AI agents. Auth is MCP-native OAuth 2.1:
    # the Oracul is a Resource Server that verifies Cognito-issued JWT access
    # tokens. The whole mount is CONDITIONAL on cognito_user_pool_id being set —
    # a deploy without these vars simply omits /mcp rather than failing startup,
    # so the REST API is never held hostage to Cognito config. (Unset here rather
    # than required precisely so the two-PR rollout — Cognito infra, then this
    # code — can't crash the box if the env lags the merge.)
    cognito_user_pool_id: Optional[str] = None  # e.g. eu-central-1_ABC123 — presence enables /mcp
    cognito_region: Optional[str] = None  # falls back to the region prefix of the pool id
    # Comma-separated Cognito app-client IDs allowed to call /mcp (the Claude
    # public client + daatan's M2M client). Empty = reject all (fail closed).
    cognito_allowed_client_ids: str = ""
    # Public URL of this Resource Server — the OAuth resource indicator and the
    # `resource` value in the protected-resource metadata. Must match what
    # clients request tokens for.
    mcp_resource_url: str = "https://oracle.daatan.com/mcp"
    # ── DCR façade (human Claude-connector login) ──────────────────────────
    # Cognito has no Dynamic Client Registration, which Claude's MCP connector
    # requires (it refuses a static client_id and hard-fails discovery without a
    # registration_endpoint in the AS metadata). When BOTH of these are set, the
    # Oracul origin advertises ITSELF as the authorization server (mcp_dcr.py) so
    # it can inject a registration_endpoint that hands back the one pre-provisioned
    # public client. Unset = the protected-resource metadata points straight at
    # Cognito (M2M client_credentials path only, no human login). See ORACLE_MCP.md.
    cognito_hosted_ui_domain: Optional[str] = None  # e.g. https://daatan-oracle.auth.eu-central-1.amazoncognito.com
    cognito_claude_client_id: Optional[str] = None  # public PKCE client id the /register façade returns

    @property
    def cognito_region_resolved(self) -> Optional[str]:
        """Region for the Cognito issuer/JWKS. Explicit override, else the
        prefix of the pool id (Cognito pool ids are '<region>_<hash>')."""
        if self.cognito_region:
            return self.cognito_region
        if self.cognito_user_pool_id and "_" in self.cognito_user_pool_id:
            return self.cognito_user_pool_id.split("_", 1)[0]
        return None

    @property
    def cognito_issuer(self) -> Optional[str]:
        region = self.cognito_region_resolved
        if not (self.cognito_user_pool_id and region):
            return None
        return f"https://cognito-idp.{region}.amazonaws.com/{self.cognito_user_pool_id}"

    @property
    def cognito_jwks_url(self) -> Optional[str]:
        iss = self.cognito_issuer
        return f"{iss}/.well-known/jwks.json" if iss else None

    @property
    def cognito_allowed_client_id_set(self) -> set[str]:
        return {c.strip() for c in self.cognito_allowed_client_ids.split(",") if c.strip()}

    @property
    def relevance_reuse_allowed_client_set(self) -> set[str]:
        return {c.strip() for c in self.relevance_reuse_allowed_clients.split(",") if c.strip()}

    @property
    def degraded_fetch_domain_set(self) -> set[str]:
        return {d.strip().lower() for d in self.degraded_fetch_domains.split(",") if d.strip()}

    @property
    def mcp_enabled(self) -> bool:
        """True when enough Cognito config is present to mount /mcp. When False,
        main.py skips the mount entirely (REST API unaffected)."""
        return bool(self.cognito_issuer)

    @property
    def mcp_allowed_hosts(self) -> list[str]:
        """Host header values the MCP streamable-HTTP transport accepts.

        The mcp SDK's DNS-rebinding guard auto-restricts to localhost when the
        app binds 127.0.0.1; behind nginx the app sees the public Host header
        (e.g. oracle.daatan.com) and every authenticated call would 421. Allow
        the resource URL's host (exact + any explicit port) plus localhost for
        in-box health probes."""
        from urllib.parse import urlparse

        host = urlparse(self.mcp_resource_url).netloc
        localhost = ["127.0.0.1:*", "localhost:*"]
        return [host, f"{host}:*", *localhost] if host else localhost

    @property
    def mcp_allowed_origins(self) -> list[str]:
        """Origin header values the MCP transport accepts. Non-browser clients
        omit Origin (allowed outright); a browser client on the resource host's
        own origin is allowed. Also allows the GitHub Pages origin that hosts
        oracle-mcp-test.html — same trust boundary as the CORSMiddleware
        allow_origins list in main.py, since that page calls /mcp directly
        from the browser with a real Bearer token, not just REST endpoints."""
        from urllib.parse import urlparse

        parsed = urlparse(self.mcp_resource_url)
        if not parsed.netloc:
            return ["https://daatan.github.io"]
        origin = f"{parsed.scheme}://{parsed.netloc}"
        return [origin, f"{origin}:*", "https://daatan.github.io"]

    @property
    def mcp_as_issuer(self) -> Optional[str]:
        """The authorization-server issuer the Oracul advertises for the DCR
        façade — its own origin (scheme://host of mcp_resource_url). This is the
        value clients fetch the AS metadata from, so it must equal metadata.issuer."""
        from urllib.parse import urlparse

        parsed = urlparse(self.mcp_resource_url)
        return f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else None

    @property
    def cognito_authorize_endpoint(self) -> Optional[str]:
        """Cognito hosted-UI OAuth2 authorize endpoint (browser login)."""
        d = (self.cognito_hosted_ui_domain or "").rstrip("/")
        return f"{d}/oauth2/authorize" if d else None

    @property
    def cognito_token_endpoint(self) -> Optional[str]:
        """Cognito hosted-UI OAuth2 token endpoint (code→token exchange)."""
        d = (self.cognito_hosted_ui_domain or "").rstrip("/")
        return f"{d}/oauth2/token" if d else None

    @property
    def dcr_enabled(self) -> bool:
        """True when the DCR façade should be served (the human Claude login
        flow). Requires the RS enabled plus the Cognito hosted-UI domain and the
        static public client id. When False the M2M-only path is served unchanged."""
        return bool(
            self.mcp_enabled
            and self.cognito_hosted_ui_domain
            and self.cognito_claude_client_id
            and self.mcp_as_issuer
        )

    @property
    def resolved_leaderboard_path(self) -> Path:
        if self.leaderboard_path != Path(""):
            return self.leaderboard_path
        return self.data_dir / "leaderboard.json"

    @property
    def resolved_resolution_feedback_path(self) -> Path:
        if self.resolution_feedback_path != Path(""):
            return self.resolution_feedback_path
        return self.data_dir / "resolution_feedback.jsonl"

    @property
    def resolved_resolution_leaderboard_path(self) -> Path:
        if self.resolution_leaderboard_path != Path(""):
            return self.resolution_leaderboard_path
        return self.data_dir / "resolution_leaderboard.json"

    @property
    def resolved_resolution_author_leaderboard_path(self) -> Path:
        if self.resolution_author_leaderboard_path != Path(""):
            return self.resolution_author_leaderboard_path
        return self.data_dir / "resolution_author_leaderboard.json"

    @property
    def resolved_settlement_pin_ledger_path(self) -> Path:
        if self.settlement_pin_ledger_path != Path(""):
            return self.settlement_pin_ledger_path
        return self.data_dir / "settlement_pin_ledger.jsonl"

    @property
    def resolved_settlement_verdict_cache_path(self) -> Path:
        if self.settlement_verdict_cache_path != Path(""):
            return self.settlement_verdict_cache_path
        return self.data_dir / "settlement_verdict_cache"

    @property
    def resolved_event_decomposition_cache_path(self) -> Path:
        if self.event_decomposition_cache_path != Path(""):
            return self.event_decomposition_cache_path
        return self.data_dir / "event_decomposition_cache"


settings = ApiSettings()
