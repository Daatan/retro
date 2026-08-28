-- Input side of scripts/score_settlement_ledger.py (retro#691). Read-only; run
-- against daatan PROD.
--
-- Set :ids to the `prediction_id`s in settlement_pin_ledger.jsonl (12 rows as of
-- 2026-08-28) — those are the only questions that were both pinned and resolved,
-- and therefore the only ones with a non-model label.
--
-- Emits one JSON object per line, `pred` rows and `row` rows mixed, straight
-- into --data:
--
--   docker exec -i daatan-postgres psql -U daatan -d daatan -tA \
--     -f settlement_ledger_export.sql
--
-- Over SSM this output blows the ~24KB Run Command cap on any question with a
-- large pool (one had 84 settled rows), so pipe it through `gzip -9 | base64 -w0`
-- and decompress locally. Note also that psql prints "Output format is
-- unaligned." to stdout — skip any line that is not a JSON object.
--
-- The row filters are the same load-bearing four as settlement_backtest_export.sql;
-- the as-of and settlement_vote_validity filtering happens in Python, because it
-- depends on the resolution timestamp carried by the ledger, not by this table.
\pset format unaligned
\pset tuples_only on
SELECT json_build_object(
  'kind', 'pred', 'id', p.id, 'claim', p."claimText",
  'deadline', to_char(p.claim_deadline, 'YYYY-MM-DD'),
  'created', to_char(p."createdAt", 'YYYY-MM-DD'),
  'archetype', p.claim_archetype, 'direction', p.claim_direction
)::text
FROM predictions p
WHERE p.id IN (:ids)
UNION ALL
SELECT json_build_object(
  'kind', 'row', 'pid', a."predictionId", 'source', a.source,
  'published', left(a.published_date, 10),
  'added_at', to_char(a.added_at, 'YYYY-MM-DD HH24:MI:SS'),
  'stance', a.stance, 'certainty', a.certainty,
  'sed', a.settlement_event_date, 'occ', a.is_occurrence, 'facet', a.facet,
  'actors', left(a.event_actors, 60), 'target', left(a.event_target, 60),
  'cls', a.evidence_class,
  'claims', (
    SELECT json_agg(json_build_object(
      'claim', left(c->>'claim', 110), 'st', (c->>'stance'), 'ct', (c->>'certainty'),
      'occ', (c->>'is_occurrence'), 'ac', left(c->>'event_actors', 50),
      'tg', left(c->>'event_target', 50), 'ed', (c->>'event_date'),
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
  AND a."predictionId" IN (:ids);
