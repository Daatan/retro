"""Relation-linker (retro#574) calibration eval.

NOT a CI test — it calls Bedrock (Nova-Lite). Run manually after any change to the
relation_linker prompt or model:

    cd pipeline && AWS_REGION=us-east-1 .venv/bin/python eval_relation_linker.py

IMPORTANT — this is NOT the "110-pair hand-checked set" retro#574's Acceptance section
names. That full labelled set does not exist as a persisted artifact anywhere: checked
`Daatan/docs audits/oracle-v2-first-experiments-2026-08-21.md` (the source of the
"8 of 110 incoherent" measurement), the docs PR#129 that introduced it, and
`conditionals.md` §5.3 — the 110-pair pass was a one-off hand-check against prod, and
only its 8 incoherent pairs plus a handful of contrast examples were ever written down.
There is nothing to re-derive the other ~100 pairs' types from.

CASES below is the full set of pairs that ARE documented with enough detail (claim
wording + the audit's own stated constraint type) to hand-derive a relation_type/
direction/polarity label with confidence: all 8 real incoherent pairs from that audit,
plus 2 real coherent contrast pairs it also names, plus 1 pair illustrating the
"topical but unconstrained" majority case the audit describes in prose (no exact
forecast IDs given for that one, since none were published — only the "~99% of pairs
are topical relatives" framing). None of these 11 appear in relation_linker.PROMPT_PREFIX's
worked examples (those are synthetic, to avoid contaminating this eval).

So: this reports agreement on an 11-pair honestly-sourced gold set, not the 110-pair,
>=90% bar from the issue text verbatim. Treat the number below as the best currently
measurable proxy for that bar, not a substitute for it — the real 90%-of-110 check
needs someone to hand-label the other ~100 pairs first (a follow-up, not blocked by
this classifier existing).
"""
import asyncio

from tm.relation_linker import classify_relation

# (claim_a, claim_b, expected_relation_type, expected_direction, expected_polarity)
# Sourced from Daatan/docs audits/oracle-v2-first-experiments-2026-08-21.md §1's 8-row
# violation table (rows 1-8) and its "coherent examples for contrast" paragraph (rows
# 9-10). Row 11 is the audit's own prose example of the ~94/110 "topical, unconstrained"
# majority (no real forecast IDs published for it).
CASES = [
    # --- the 8 real incoherent pairs (audit table, in its row order) ---
    ("The IDF maintains a military presence in southern Lebanon through 31 December 2026.",
     "Israel withdraws its forces from southern Lebanon by 31 December 2026.",
     "alias", None, "opposite"),
    ("Benjamin Netanyahu is Prime Minister of Israel on 31 December 2026.",
     "Yair (Gadi) Eisenkot is Prime Minister of Israel after the 27 October 2026 election.",
     "complement", None, "same"),
    ("Brent crude oil reaches or exceeds $200 per barrel by December 2026.",
     "Brent crude oil reaches or exceeds $130 per barrel by December 2026.",
     "nested", "a_to_b", "same"),
    ("Israel maintains a long-term military occupation of southern Lebanon.",
     "Israel withdraws its forces from southern Lebanon by 31 December 2026.",
     "implies", "a_to_b", "opposite"),
    ("Israel conducts a ground invasion and occupation of Iran's Kharg Island.",
     "Israel conducts a ground invasion of Iran's Kharg Island.",
     "nested", "a_to_b", "same"),
    ("The United States deploys ground troops to Iran.",
     "President Trump orders the deployment of U.S. troops to Iran.",
     "implies", "a_to_b", "same"),
    ("Benjamin Netanyahu forms the next government of Israel.",
     "Yair (Gadi) Eisenkot becomes Prime Minister of Israel.",
     "complement", None, "same"),
    ("Benjamin Netanyahu wins the election and is appointed Prime Minister.",
     "Benjamin Netanyahu is appointed Prime Minister after the election.",
     "nested", "a_to_b", "same"),
    # --- 2 real coherent contrast pairs (audit prose, same section) ---
    ("A building at the Tel Aviv central bus station is demolished by 2029.",
     "A building at the Tel Aviv central bus station is demolished by 2030.",
     "nested", "a_to_b", "same"),
    ("Israel and Lebanon sign a normalization agreement by 6 November 2026.",
     "Israel and Lebanon hold direct talks.",
     "implies", "a_to_b", "same"),
    # --- 1 "topical, unconstrained" example (audit's own prose pairing) ---
    ("Vladimir Putin's health seriously declines due to illness.",
     "Vladimir Putin makes a public appearance.",
     "independent", None, None),
]


async def main() -> None:
    correct_type = correct_all = 0
    print(f"\n=== relation_linker eval, n={len(CASES)} (see module docstring: NOT the 110-pair set) ===")
    for claim_a, claim_b, exp_type, exp_dir, exp_pol in CASES:
        try:
            out, _ = await classify_relation(claim_a, claim_b)
            got_type, got_dir, got_pol = out.relation_type, out.direction, out.polarity
        except Exception as e:
            got_type, got_dir, got_pol = f"ERR:{type(e).__name__}", None, None
        type_ok = got_type == exp_type
        all_ok = type_ok and got_dir == exp_dir and got_pol == exp_pol
        correct_type += type_ok
        correct_all += all_ok
        mark = "✓" if all_ok else ("~" if type_ok else "✗")
        print(f"  {mark} expect=({exp_type},{exp_dir},{exp_pol}) got=({got_type},{got_dir},{got_pol})  "
              f"A={claim_a[:40]!r} B={claim_b[:40]!r}")
    n = len(CASES)
    print(f"\n  type-only agreement={correct_type}/{n} ({correct_type/n:.0%})  "
          f"type+direction+polarity agreement={correct_all}/{n} ({correct_all/n:.0%})")


if __name__ == "__main__":
    asyncio.run(main())
