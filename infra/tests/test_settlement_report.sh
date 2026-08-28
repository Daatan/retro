#!/bin/bash
# Regression tests for infra/settlement_report.py (retro#691, PR #698).
#
# The report is the tool that will answer "is it safe to enforce the settlement
# gates yet", so the ways it can quietly lie are the things worth pinning:
#
#   1. Counting re-pricings as decisions. Both log events re-fire on every
#      recompute; the verifier's own log once looked like 622 decisions and was
#      23 questions, one re-priced 144 times. A report that inflates its sample
#      size 27x reads as conclusive when it is not.
#   2. Pairing a shadow line with a verdict from a *different* pricing of the
#      same question, which invents agreement out of nothing.
#   3. Folding errored verdicts in with allowed ones. An errored verifier did
#      not allow the pin, it failed to check it — and that population is the
#      entire argument for having deterministic gates at all.
#
# Fed synthetic log text, never the real one: a test that needs prod data is a
# test that stops running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT="$SCRIPT_DIR/../settlement_report.py"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAIL=0

shadow() { # ts would_block votes demoted outlets reasons qhash
  printf '%s,123 WARNING forecast_api.forecaster — event=settlement_semantic_gates would_block=%s votes=%s demoted=%s outlets_left=%s gates=point_in_time,occurrence_consistency,facet_missing reasons=%s question=%s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7"
}
verifier() { # ts settles errored enforced votes qhash
  printf '%s,456 WARNING forecast_api.forecaster — event=settlement_verifier settles=%s errored=%s enforced=%s votes=%s cached=True samples=0 agree=0 question=%s reason='"'"'because'"'"'\n' \
    "$1" "$2" "$3" "$4" "$5" "$6"
}

run() { python3 "$REPORT" --log "$TMP/log.txt" "${@:2}"; }

# Anchored on purpose: an unanchored "1" matches "10", which is how a dedupe
# regression sails past a test that looks like it checks the decision count.
check() { # label expected_pattern actual
  if grep -qE "$2" <<<"$3"; then
    echo "  ok   $1"
  else
    echo "  FAIL $1 — expected /$2/ in:"; echo "$3" | sed 's/^/       /'; FAIL=1
  fi
}

echo "== the same vote-set re-priced many times is ONE decision =="
: > "$TMP/log.txt"
for i in 01 02 03 04 05 06 07 08 09 10; do
  shadow "2026-08-28 10:$i:00" True 4 2 1 "{'settled_without_facet': 2}" aaaabbbbcccc >> "$TMP/log.txt"
  verifier "2026-08-28 10:$i:00" False False True 4 aaaabbbbcccc >> "$TMP/log.txt"
done
OUT="$(run)"
check "raw shadow lines are still reported"   'shadow=10 '      "$OUT"
check "but they collapse to one decision"     'DECISIONS +1 +<--'   "$OUT"
check "distinct questions counted separately" 'distinct questions 1$' "$OUT"

echo "== a verdict from a different pricing is NOT paired =="
# Same question, verdict 20 minutes later: a separate recompute, not this call.
: > "$TMP/log.txt"
shadow   "2026-08-28 11:00:00" True 4 2 1 "{'settled_without_facet': 2}" aaaabbbbcccc >> "$TMP/log.txt"
verifier "2026-08-28 11:20:00" False False True 4 aaaabbbbcccc >> "$TMP/log.txt"
OUT="$(run)"
check "shadow line left unpaired"  'gates ran, no verdict line  1 ' "$OUT"
check "verdict line left unpaired" 'verdict line, no gates      1 ' "$OUT"

echo "== agreement matrix counts all four cells =="
: > "$TMP/log.txt"
shadow   "2026-08-28 12:00:00" True  4 2 1 "{'settled_without_facet': 2}" 111111111111 >> "$TMP/log.txt"
verifier "2026-08-28 12:00:01" False False True 4 111111111111 >> "$TMP/log.txt"
shadow   "2026-08-28 12:01:00" True  4 2 1 "{'settled_but_not_occurrence': 2}" 222222222222 >> "$TMP/log.txt"
verifier "2026-08-28 12:01:01" True  False False 4 222222222222 >> "$TMP/log.txt"
shadow   "2026-08-28 12:02:00" False 4 0 3 "{}" 333333333333 >> "$TMP/log.txt"
verifier "2026-08-28 12:02:01" False False True 4 333333333333 >> "$TMP/log.txt"
shadow   "2026-08-28 12:03:00" False 4 0 3 "{}" 444444444444 >> "$TMP/log.txt"
verifier "2026-08-28 12:03:01" True  False False 4 444444444444 >> "$TMP/log.txt"
OUT="$(run)"
check "one true positive, one false positive" 'gates would block +1 +1$' "$OUT"
check "one false negative, one true negative" 'gates would allow +1 +1$' "$OUT"
check "agreement is reported"                 'agreement +50%'          "$OUT"
check "cost in lost pins is named"            'cost +1 pins'            "$OUT"
check "reasons are aggregated"                'settled_without_facet'   "$OUT"

echo "== an errored verifier is unchecked, not allowed =="
: > "$TMP/log.txt"
shadow   "2026-08-28 13:00:00" True 3 2 1 "{'settled_without_facet': 2}" 555555555555 >> "$TMP/log.txt"
verifier "2026-08-28 13:00:01" True True False 3 555555555555 >> "$TMP/log.txt"
OUT="$(run)"
check "errored decisions broken out"     'it ERRORED +1 ' "$OUT"
# Pinned to the actual number, not to the word "would": a substring loose enough
# to match the surrounding prose is a test that cannot fail.
check "fail-open exposure is quantified" 'of 1 unchecked pins' "$OUT"
check "and the gates' verdict on them"   'have blocked 1[.]' "$OUT"
if grep -q 'AGREEMENT on the' <<<"$OUT"; then
  echo "  FAIL an errored-only window must not print an agreement matrix"; FAIL=1
else
  echo "  ok   an errored-only window prints no agreement matrix"
fi

echo "== the shadow path raising is surfaced, not swallowed twice =="
: > "$TMP/log.txt"
printf '2026-08-28 14:00:00,001 ERROR forecast_api.forecaster — event=settlement_semantic_gates outcome=error\n' >> "$TMP/log.txt"
OUT="$(run)"
check "errors counted" 'shadow_errors=1$' "$OUT"
check "and called out" 'shadow path raised' "$OUT"

echo "== --since filters on the line timestamp =="
: > "$TMP/log.txt"
shadow   "2026-08-20 09:00:00" True 4 2 1 "{'settled_without_facet': 2}" 666666666666 >> "$TMP/log.txt"
verifier "2026-08-20 09:00:01" False False True 4 666666666666 >> "$TMP/log.txt"
shadow   "2026-08-27 09:00:00" True 4 2 1 "{'settled_without_facet': 2}" 777777777777 >> "$TMP/log.txt"
verifier "2026-08-27 09:00:01" False False True 4 777777777777 >> "$TMP/log.txt"
OUT="$(run . --since 2026-08-25)"
check "older window dropped" 'DECISIONS +1 +<--' "$OUT"

echo "== an empty window says so rather than reporting a clean bill =="
: > "$TMP/log.txt"
printf '2026-08-28 15:00:00,001 INFO forecast_api.forecaster — event=something_else\n' >> "$TMP/log.txt"
OUT="$(run)"
check "explicit nothing-to-report" 'Nothing to report yet' "$OUT"

echo "== a mid-window gate-set change invalidates comparison =="
: > "$TMP/log.txt"
shadow "2026-08-28 16:00:00" True 4 2 1 "{'settled_without_facet': 2}" 888888888888 >> "$TMP/log.txt"
printf '2026-08-28 16:05:00,123 WARNING forecast_api.forecaster — event=settlement_semantic_gates would_block=False votes=4 demoted=0 outlets_left=3 gates=point_in_time reasons={} question=999999999999\n' >> "$TMP/log.txt"
OUT="$(run)"
check "config change flagged" 'gate set changed mid-window' "$OUT"

if [[ $FAIL -eq 0 ]]; then echo "ALL PASS"; else echo "FAILURES"; exit 1; fi
