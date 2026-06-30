"""Gatekeeper coarse-gate calibration eval.

NOT a CI test — it calls Bedrock (Nova-Micro). Run manually after any change to the
gatekeeper prompt or model to confirm the coarse gate keeps high recall on
indirect-but-relevant evidence WITHOUT losing precision on different-matter / off-domain
noise:

    cd pipeline && AWS_REGION=us-east-1 .venv/bin/python eval_gatekeeper.py

Labels are the expected `is_prediction` for the (claim, article) pair, spanning multiple
claim domains. The current prompt should score 100% / 100% on this set; a regression here
means the gate is back to rejecting indirect evidence (recall drop) or waving through
off-topic noise (precision drop).
"""
import asyncio

from tm.gatekeeper import PROMPT
from tm.models import GatekeeperOutput
from tm import llm

GATE_MODEL = "bedrock/amazon.nova-micro-v1:0"

# (claim, article, expected_is_prediction)
CASES = [
    # --- should PASS: relevant, much of it indirect (never names the outcome) ---
    ("Bibi Netanyahu Wins the Elections 2026",
     "Sharp internal conflict has erupted inside Likud over the party's primary procedures ahead of the 2026 election, with senior figures challenging Netanyahu's control of the candidate lists.", True),
    ("Bibi Netanyahu Wins the Elections 2026",
     "Israel's shifting relationship with the U.S. over the Iran nuclear deal has dented Netanyahu's standing among security-focused voters, commentators say, weeks before the 2026 vote.", True),
    ("Bibi Netanyahu Wins the Elections 2026",
     "Naftali Bennett's centrist party has cratered in the polls and Bennett announced he will not stand in 2026, leaving Netanyahu without his main challenger from the center.", True),
    ("A sitting G20 leader dies in 2026",
     "President Alvarez, 78, was hospitalized overnight in critical condition with a serious cardiac event, his office confirmed, and has cancelled all public appearances indefinitely.", True),
    ("The EU admits a new member state by 2027",
     "Accession negotiations with Montenegro advanced this week as Brussels provisionally closed two more policy chapters, the bloc's fastest progress in years.", True),
    ("Manchester City wins the Premier League this season",
     "City's title hopes were dealt a blow as their top scorer suffered a season-ending knee injury, ruling him out for the rest of the campaign.", True),
    ("Bitcoin exceeds $200,000 in 2026",
     "The Federal Reserve signalled two more rate cuts and risk appetite surged across markets, with investors rotating into higher-beta assets.", True),
    ("Country X and Country Y reach a ceasefire in 2026",
     "X and Y completed a prisoner exchange and have opened indirect back-channel talks mediated by a neutral third party, officials said.", True),
    # --- should REJECT: different matter / domain / no content ---
    ("Bibi Netanyahu Wins the Elections 2026",
     "An AI model predicts Spain as the most likely winner of the 2026 World Cup, with France and Argentina close behind in the statistical forecast.", False),
    ("Bibi Netanyahu Wins the Elections 2026",
     "U.S. equities: the growth-vs-value trade is back in focus as investors weigh 2026 rate-cut forecasts and tech earnings season.", False),
    ("Bibi Netanyahu Wins the Elections 2026",
     "A new survey explores Israeli women's leisure interests and consumer habits across age groups, with no political content.", False),
    ("Company Z releases its new phone by Q4 2026",
     "Z's chief executive spoke at length about the firm's unrelated overseas tax-restructuring strategy and dividend policy; no products were mentioned.", False),
    ("Bitcoin exceeds $200,000 in 2026",
     "A pop star tweeted a joke meme-coin name to fans; the post was a gag and referenced no real asset or market.", False),
    ("Bibi Netanyahu Wins the Elections 2026",
     "Subscribe to continue reading. This article is for premium members only. Sign in or create an account.", False),
]


async def _gate(claim: str, text: str) -> GatekeeperOutput:
    prompt = PROMPT.format(article_text=text, source_name="Test", article_date="2026-06-30", event_name=claim)
    out, _ = await llm.complete_structured(GATE_MODEL, GatekeeperOutput, prompt, max_tokens=240, timeout=60)
    return out


async def main() -> None:
    tp = fp = tn = fn = 0
    print(f"\n=== gatekeeper coarse-gate eval on {GATE_MODEL} ===")
    for claim, text, expect in CASES:
        try:
            o = await _gate(claim, text)
            got, rel = o.is_prediction, o.relevance_score
        except Exception as e:
            got, rel = f"ERR:{type(e).__name__}", "-"
        if expect and got is True:
            tp += 1
        elif expect:
            fn += 1
        elif got is False:
            tn += 1
        else:
            fp += 1
        mark = "✓" if got is expect else "✗"
        print(f"  {mark} expect={'PASS' if expect else 'REJ '} got={str(got):5} rel={rel}  {text[:56]}")
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    print(f"\n  PASS-recall={recall:.0%} ({tp}/{tp + fn})  REJECT-precision={precision:.0%} (fp={fp})  correct={tp + tn}/{len(CASES)}")


if __name__ == "__main__":
    asyncio.run(main())
