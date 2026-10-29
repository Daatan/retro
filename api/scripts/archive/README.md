# Archived one-off scripts

Spent measurements and one-off backfills. Kept, not deleted: they are the provenance of
constants and decisions still cited in `config.py`, `aggregation.py` and
`docs/ORACLE_VARIABLES.md`. Not run by CI or any job. Moved here under retro#977.

| Script | Issue | What it produced |
|---|---|---|
| `replay_claims_detail.py` | retro#398 | Replay check that stored pool rows still reduce to the numbers they shipped with (`forecast_api.claims_replay`) |
| `measure_dependence_key_780.py` | retro#780 | Step 0 measurement behind `event_key_dependence_collapse` (event-key-only spec, 2.3% collapse) |
| `eyeball_cluster_pairs.py` | retro#414 | Pair-text review behind the 2026-08-09 `cluster_jaccard_threshold` retune |
