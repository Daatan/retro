-- retro#691 step 2: the labelling candidate set — every settled pool row in prod,
-- not just the ones the settlement verifier happened to be called on.
--
--   docker exec -i daatan-postgres psql -U daatan -d daatan < export_settlement_labels.sql
--
-- Read-only. Emits one JSON object per (row, settled claim) pair.
--
-- Why this and not the verifier log: the log covers 23 distinct questions and
-- yields 9 blocks (see scripts/backtest_settlement_semantic.py). This query
-- covers 52 and ~282 rows. Pin-vs-outcome ground truth is NOT the widening
-- path — prod has 59 resolved predictions but only 7 that were ever pinned,
-- so outcome labels top out at 7. The label here is per (question, settled
-- claim): "does this settled fact establish the question's own event?", which
-- is answerable from the claim text alone and does not need the resolution.
--
-- The row filters mirror settlement_backtest_export.sql and are load-bearing
-- for the same reason (reconstruction fidelity): see that file's header.
--
-- The gate inputs (stance, claim_strength, event_actors, event_target,
-- is_occurrence, facet, evidence_class, event_date) ARE selected — the scorer
-- needs them to run the gates — but `label_settlement_candidates.py` never puts
-- them in the prompt: `build_prompt` sends only the question, the claim, the
-- quote and the dates. That asymmetry is the point. A labeller shown the gate's
-- inputs is grading the gate's own worksheet, so the blindness is enforced in
-- the prompt builder and asserted in tests, not by omitting columns a second
-- consumer legitimately needs.
\pset format unaligned
\pset tuples_only on
SELECT json_build_object(
  'pid',        a."predictionId",
  'question',   p."claimText",
  'deadline',   to_char(p.claim_deadline, 'YYYY-MM-DD'),
  'url',        a.url,
  'outlet',     a.source,
  'published',  left(a.published_date, 10),
  -- when the row entered the pool: `facet` went live in the extractor the week of
  -- 2026-08-10, so gate_facet_missing must be scored on rows added after that or
  -- it grades a schema rollout instead of an elicitation failure.
  'added',      to_char(a.added_at, 'YYYY-MM-DD'),
  'claim',      c->>'claim',
  'quote',      c->>'quote',
  -- gate inputs: for the scorer only, never for the labeller's prompt
  'outlet_row', a.source,
  'stance',     (c->>'stance'),
  'certainty',  coalesce(c->>'claim_strength', c->>'certainty'),
  'actors',     (c->>'event_actors'),
  'target',     (c->>'event_target'),
  'event_date', (c->>'event_date'),
  'occ',        (c->>'is_occurrence'),
  'facet',      (c->>'facet'),
  'cls',        (c->>'evidence_class')
)::text
FROM evidence_pool_articles a
JOIN predictions p ON p.id = a."predictionId"
CROSS JOIN LATERAL jsonb_array_elements(a.claims_detail) c
WHERE a.status = 'COMPLETE'
  AND a.settled
  AND a.claims_detail IS NOT NULL
  AND a.superseded_at IS NULL
  AND NOT a.excluded
  AND (c->>'settled')::boolean IS TRUE
ORDER BY a."predictionId", a.added_at;
