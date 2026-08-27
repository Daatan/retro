-- Vote side of the retro#691 backtest dataset: the settled pool rows behind the
-- settlement verifier's recorded verdicts. Read-only; run against daatan PROD.
--
--   psql -U daatan -d daatan -f settlement_backtest_export.sql
--
-- Emits one JSON object per row, ready for
-- scripts/backtest_settlement_semantic.py --rows.
--
-- Set :ids to the prediction ids whose question hashes appear in the verdict log
-- (sha256(claimText.strip().casefold())[:12] == forecaster._question_hash).
--
-- The three filters below are load-bearing for reconstruction fidelity — the
-- harness reports how often its vote count matches the log, and dropping any of
-- them lowers it:
--   * status='COMPLETE'      — FAILED/abstained rows never reach the pool
--   * settled                — already post-settlement_grade (applied at CLAIM
--                              level in forecaster.py:329), so do NOT re-apply
--                              the grade bar to the row's mean stance
--   * superseded_at IS NULL  — a re-extracted row's older version never votes
--   * NOT excluded           — the admin kill switch
\pset format unaligned
\pset tuples_only on
SELECT json_build_object(
  'pid', a."predictionId", 'url', a.url, 'source', a.source,
  'published', left(a.published_date, 10),
  'added_at', to_char(a.added_at, 'YYYY-MM-DD HH24:MI:SS'),
  'stance', a.stance, 'certainty', a.certainty,
  'sed', a.settlement_event_date, 'occ', a.is_occurrence, 'facet', a.facet,
  'actors', left(a.event_actors, 80), 'target', left(a.event_target, 80),
  'cls', a.evidence_class,
  'claims', (
    SELECT json_agg(json_build_object(
      'claim', left(c->>'claim', 180), 'q', left(c->>'quote', 160),
      'st', (c->>'stance'), 'ct', (c->>'certainty'),
      'occ', (c->>'is_occurrence'), 'ac', left(c->>'event_actors', 60),
      'tg', left(c->>'event_target', 60), 'ed', (c->>'event_date'),
      'fc', (c->>'facet'), 'cls', (c->>'evidence_class')))
    FROM jsonb_array_elements(a.claims_detail) c
    WHERE (c->>'settled')::boolean IS TRUE)
)::text
FROM evidence_pool_articles a
WHERE a.status = 'COMPLETE'
  AND a.settled
  AND a.claims_detail IS NOT NULL
  AND a.superseded_at IS NULL
  AND NOT a.excluded
  AND a."predictionId" IN (:ids)
ORDER BY a.added_at;
