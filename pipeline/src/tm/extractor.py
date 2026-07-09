from .models import ExtractionOutput
from .config import settings
from .llm import complete_structured

PROMPT = """\
You are a forensic prediction analyst. Your job is to extract EVERY signal — \
explicit or implicit — that bears on whether the RELATED EVENT will occur.

## What counts as a signal

Extract ALL of the following:
- Explicit forecasts: "X will happen", "Y is expected to..."
- Implied directional views: "the economy is heading toward...", "pressure is mounting"
- Factual reports whose content logically implies an outcome — e.g. reporting that \
  troops are advancing implies a battle outcome; reporting that negotiations collapsed \
  implies a deal is less likely. INFER the implication.
- Quotes from officials, analysts, or experts that imply a position
- Vague sentiment that colors likelihood: "things are deteriorating", "a breakthrough \
  looks distant"
- Even near-zero certainty signals (certainty=0.1) are valuable — include them

## What does NOT count
- Pure background with zero bearing on the event (e.g. article only covers geography)
- Statements about a wholly different event with no link to the related event
- Statements about the SAME subject but a DIFFERENT timeframe, edition, or deadline \
than the related event — e.g. next season's title odds when the event is this season's \
final, a later election, a different year's target. Check any year/date in the claim \
against the related event's timeframe; if they refer to a different occurrence of the \
event, do NOT extract it.
- Once an "assumes a role/office" claim is already settled true, later coverage of \
that person's ACTIONS IN OFFICE — policy disputes, approval ratings, political \
controversies, conflicts with other officials — is NOT a signal about whether they \
assumed the role. That question is already decided; an article entirely about their \
post-assumption governance and conduct has no bearing on the arrival claim at all.

## STANCE — the most important field
Stance measures how strongly this signal implies the RELATED EVENT will occur.
  +1.0 = certain the event WILL happen
  -1.0 = certain the event WILL NOT happen
   0.0 = neutral / no directional signal

Ask yourself: "If this quote/fact is true, does it make the related event more likely \
(positive stance) or less likely (negative stance)?"

Examples — related event: "Assad regime falls in Syria":
  "Rebel forces are closing in on Hama"        → stance +0.7, certainty 0.6
  "Assad's army is holding the line"           → stance −0.6, certainty 0.5
  "The conflict has dragged on for two years"  → stance +0.2, certainty 0.2
  "International sanctions remain in place"   → stance +0.3, certainty 0.3
  "Rebels have taken Damascus; Assad has fled the country" \
                                               → stance +1.0, certainty 0.95, settled true
  "Assad crushed the uprising; the rebellion is over" \
                                               → stance −1.0, certainty 0.95, settled true

Note: even factual/contextual sentences have a stance if they imply a direction.
Do NOT use stance to indicate good/bad — only more/less likely to happen.

## Numeric thresholds — compare the numbers, not the sentiment
When the related event states a quantitative threshold ("more than 33 seats", \
"below $50,000", "at least 10 medals", "reaches 2800 rating"), judge each signal \
by COMPARING its number against the threshold — never by momentum or by how \
positive the news sounds for the subject. A reported or projected value on the \
wrong side of the threshold is NEGATIVE stance even when it is good news for \
the subject; general success without a number that clears the bar is at most \
weakly positive.

Examples — related event: "Likud wins more than 33 seats in the election":
  "Poll projects Likud at 31 seats"        → stance −0.6, certainty 0.7  (31 ≤ 33: contradicts)
  "Likud is leading in the polls"          → stance +0.2, certainty 0.3  (leading ≠ >33 seats)
  "Poll gives Likud 36 seats"              → stance +0.7, certainty 0.7  (36 > 33: supports)
  "Likud gained two seats since last poll" → stance +0.2, certainty 0.3  (trend, no level given)

## Multi-stage / bracket events — discount single-stage "favorite" framing
When the related event requires winning a SEQUENCE of separate future contests \
(a tournament bracket, a playoff series, a multi-round election, a series of \
confirmation votes) rather than one determination, an article's "favorite," \
"front-runner," or "strong candidate" framing about ONE upcoming stage is weak \
support for the event as a whole — it says nothing about the stages still to \
come. Advancing past one stage narrows the field but does not itself imply the \
final outcome; only raise stance as stages actually clear, and reserve high \
certainty for articles that address the full remaining path, not just the next \
match or round.

Examples — related event: "France wins the 2026 World Cup" (tournament bracket):
  "France is a strong favorite entering the Round of 16"     → stance +0.3, certainty 0.3  (one stage of several remaining)
  "France beats Paraguay to reach the quarter-finals"        → stance +0.4, certainty 0.5  (one stage cleared, more remain)
  "France reaches the final after a dominant semi-final win" → stance +0.6, certainty 0.6  (one stage left)

Examples — related event: "Judge Alvarez is confirmed to the Supreme Court" (committee vote, then floor vote):
  "Alvarez is seen as the clear favorite to be confirmed"       → stance +0.3, certainty 0.3  (favorite framing, no vote yet)
  "The Judiciary Committee advances Alvarez's nomination 12-10" → stance +0.4, certainty 0.5  (one stage cleared, floor vote remains)

Examples — related event: "Diaz wins the presidential runoff" (first round, then runoff):
  "Diaz leads first-round polling by 8 points"        → stance +0.2, certainty 0.3  (first round ≠ runoff win)
  "Diaz advances to the runoff after finishing first" → stance +0.4, certainty 0.5  (one stage cleared, runoff remains)

## Cited quantitative estimates — extract them as a distinct anchor
When the article itself cites an explicit modeled probability, poll number, seat \
projection, or market price FOR THE RELATED EVENT ITSELF (not a proxy stage) — e.g. \
"a model gives Team X an 18.83% chance to win the tournament", "the poll puts \
Candidate Y at 45%", "the prediction market prices the deal at 33%" — extract that \
figure into `quantitative_estimate` as a probability in [0, 1] (convert percentages: \
18.83% → 0.1883). Set `stance` to match it (`stance = 2 × quantitative_estimate − 1`) \
and `certainty` high (≥ 0.8) — a named model, poll, or market is a much stronger \
anchor than qualitative "favorite"/"strong candidate" framing, even when several \
qualitative articles exist alongside it. Leave `quantitative_estimate` null when the \
article has no such explicit cited figure — general "leading in the polls" or "seen \
as the favorite" language without a stated number is NOT a quantitative estimate; \
keep using the sections above for that. Also leave it null for a CASUAL or \
CONVERSATIONAL figure of speech — a pundit, fan, coach, or player tossing out "I'd \
give it a 50-50 chance" or "there's maybe a 90% chance" is voicing a personal opinion, \
not citing a model/poll/market; only a NAMED formal source counts.

Examples — related event: "France wins the 2026 World Cup":
  "Simulations by Opta give France the best chance of winning the tournament, at 18.83%" \
                                               → stance −0.62, certainty 0.85, quantitative_estimate 0.1883
  "Betting markets rank France as favorites to lift the trophy" \
                                               → stance +0.3, certainty 0.3, quantitative_estimate null (no number given)
  "The team's own coach joked there's maybe a 90% chance they choke again" \
                                               → stance −0.3, certainty 0.3, quantitative_estimate null (casual personal opinion, not a named model/poll/market)

Examples — related event: "Likud wins more than 33 seats in the election":
  "A poll-aggregator model gives Likud a 22% chance of winning more than 33 seats" \
                                               → stance −0.56, certainty 0.85, quantitative_estimate 0.22
  "Likud is seen as gaining momentum heading into the vote" \
                                               → stance +0.2, certainty 0.3, quantitative_estimate null (momentum, no cited figure)

## EVIDENCE CLASS — optional; classify the KIND of evidence this claim is
This is a new, EXPERIMENTAL field. It is not yet used to weight anything —
classify it independently and honestly; do not let it influence stance or
certainty, and vice versa. If a claim genuinely does not fit one category
cleanly, OMIT the field entirely rather than guessing — a missing
evidence_class is fine, a wrong one is worse than none.

Choose exactly one of:
  reported_fact      — a plain declarative statement of something that
                        happened or is currently true (not hedged, not a
                        forecast). Independent of `settled`: a reported_fact
                        can be about a sub-event that doesn't itself settle
                        the related event (e.g. "the committee voted 12-10"
                        is a reported_fact but the confirmation isn't settled).
  cited_probability   — the claim cites an explicit modeled/poll/market
                        PROBABILITY for the related event itself (same
                        figure that would populate `quantitative_estimate`
                        as a genuine probability of the event occurring).
  cited_share         — the claim cites a poll SHARE, vote share, or seat
                        count. This is explicitly NOT a probability the
                        event occurs — "the party polls at 28%" is a share
                        of the vote, not a 28% chance of winning. Use this
                        even when the same figure would also populate
                        `quantitative_estimate` under today's (separate,
                        unrelated) rules for that field.
  reporting           — ordinary hedged or prospective news coverage: "is
                        expected to", "sources say", "is likely to" — a
                        genuine report, but about a future or uncertain
                        state, not a settled fact.
  opinion             — a pundit's, fan's, or individual's subjective view,
                        speculation, or casual figure of speech — "seen as
                        the favorite", "I'd give it 50-50" from an unnamed
                        or non-expert voice.

Examples — related event: "Assad regime falls in Syria":
  "Rebels seized the capital on Sunday as Assad fled to Moscow" \
                                               → evidence_class reported_fact
  "Analysts expect the rebel offensive to reach Damascus within weeks" \
                                               → evidence_class reporting
  "One commentator said Assad is basically finished at this point" \
                                               → evidence_class opinion

Examples — related event: "Likud wins more than 33 seats in the election":
  "A poll-aggregator model gives Likud a 22% chance of winning more than 33 seats" \
                                               → evidence_class cited_probability
  "The latest poll puts Likud at 28% of the vote" \
                                               → evidence_class cited_share
  "Likud is seen as gaining momentum heading into the vote" \
                                               → evidence_class reporting

## SETTLED — the event already happened (or definitively cannot)
When the article REPORTS THE OUTCOME AS AN ACCOMPLISHED FACT — the event occurred, \
or became permanently impossible (deadline passed, subject died, contest decided) — \
set settled to true and use the full ±1.0 stance with certainty ≥ 0.9. Past-tense \
reporting of the outcome ("X won", "the deal was signed", "Y has died") is settled; \
predictions, odds, and expectations ("X is likely to win") are NOT settled, however \
confident. Do not soften a settled outcome into a likelihood — a report that the \
event happened is stance +1.0, not +0.7.

### Numeric-threshold events — a mid-event tally is NOT settled
When the related event is a threshold claim ("scores at least 8 goals", "wins more \
than 33 seats") and the article reports a running tally from an ONGOING contest, \
only mark settled true if the reported number ALREADY crosses the threshold — an \
"at least N" claim is monotonic, so once N is reached it cannot be undone by later \
events, even mid-contest. A tally that has NOT yet crossed the threshold is never \
settled while the contest is still open, no matter how final-sounding the framing \
("leading scorer", "tied for first") — the outcome is still undetermined. Judge the \
number against the threshold exactly as in the Numeric thresholds section above; do \
not treat a running total as an accomplished fact just because it's stated as fact.

Examples — related event: "Messi scores at least 8 goals in the tournament":
  "Messi has scored 9 goals so far, tournament still underway" \
                                               → stance +1.0, certainty 0.95, settled true (9 ≥ 8: already locked in)
  "Messi and Mbappe are tied for the tournament lead with 6 goals each, group stage ongoing" \
                                               → stance −0.3, certainty 0.4, settled false (6 < 8, contest still open — a tally, not a verdict)
  "The tournament has concluded; Messi finished with 7 goals" \
                                               → stance −1.0, certainty 0.95, settled true (contest over, 7 < 8 is now permanent)

### Buried facts — extract settlement even when it's incidental to the article's main topic
A clear past-tense statement of the RELATED EVENT can appear as a single supporting \
clause inside an article whose main subject is something else entirely (e.g. a piece \
about downstream diplomatic fallout that mentions, in passing, that the event already \
happened). Scan the WHOLE article, not just the headline or opening paragraph — extract \
the fact and mark settled true regardless of how minor its role in the article is. \
Do not require the fact to be the article's primary subject to count it as settled.

Examples — related event: "Peter Magyar will officially assume the role of Prime Minister \
of Hungary by December 31, 2026":
  Article mainly about Ukraine-Hungary pipeline relations, mentioning in passing: \
  "...its leader Peter Magyar became Prime Minister on May 9" \
                                               → stance +1.0, certainty 0.95, settled true \
                                                 (clear past-tense fact, however incidental to the article's main topic)

### Historical background is NOT a settlement of the current question
A past-tense fact about an EARLIER, already-known episode — background that predates \
the question's own timeframe — does not settle the current claim, no matter how \
definitively it is stated. History that set the stage for the question is context, \
not its outcome: treat it as ordinary (usually weak) signal, never as settled. Only \
a past-tense report of THIS question's outcome, within its own window, settles it.

Examples — related event: "The U.S. will formally approve an F-35 sale to Turkey by \
December 31, 2026":
  "Turkey was removed from the F-35 program in 2019 over its S-400 purchase" \
                                               → stance −0.1, certainty 0.4, settled false \
                                                 (background history predating the question's window — not this question's outcome)
  "The State Department formally approved the F-35 sale to Turkey on Tuesday" \
                                               → stance +1.0, certainty 0.95, settled true (this question's outcome, reported as fact)

## Article language
The article may be in Hebrew, Arabic, or English. Always write the claim in English.
Quote the original language verbatim in the quote field.

## Output
Extract up to 5 signals. Prefer higher-certainty ones but do not omit low-certainty \
signals if they are the only content available.

Article:
<article>
{article_text}
</article>

Source: {source_name}
Journalist: {journalist}
Date: {article_date}
Related event: {event_name} — {event_description}

IMPORTANT: Your response must be a JSON object with a "predictions" key containing a list.
Example: {{"predictions": [ {{...}}, {{...}} ]}}

Each prediction has five core fields, plus two used only when applicable:
  quote (string — original language), claim (string — English), \
stance (float −1 to 1), certainty (float 0 to 1), settled (boolean — true only when \
the source reports the outcome as an accomplished fact), quantitative_estimate \
(float 0 to 1, OMIT this field entirely unless the source cites an explicit modeled \
probability/poll/market figure for the event itself — see the section above), \
evidence_class (one of reported_fact / cited_probability / cited_share / reporting / \
opinion, OMIT this field entirely if none fits cleanly — see the section above)

Example — related event: "Assad regime falls in Syria":
{{
  "predictions": [
    {{
      "quote": "Syrian rebel forces pushed close on Tuesday to the major city of Hama",
      "claim": "Rebel advances toward Hama make Assad's fall increasingly likely",
      "stance": 0.7,
      "certainty": 0.6,
      "settled": false,
      "evidence_class": "reporting"
    }},
    {{
      "quote": "Rebels seized the capital on Sunday as Assad fled to Moscow",
      "claim": "The Assad regime has fallen; rebels control Damascus",
      "stance": 1.0,
      "certainty": 0.95,
      "settled": true,
      "evidence_class": "reported_fact"
    }}
  ]
}}

Example — related event: "France wins the 2026 World Cup" (a source citing a named model):
{{
  "predictions": [
    {{
      "quote": "Simulations by Opta indicate France has the highest chance of winning the 2026 World Cup at 18.83%",
      "claim": "Opta's model gives France an 18.83% chance to win the tournament",
      "stance": -0.62,
      "certainty": 0.85,
      "settled": false,
      "quantitative_estimate": 0.1883,
      "evidence_class": "cited_probability"
    }}
  ]
}}
"""


async def extract_predictions(
    article_text: str,
    source_name: str,
    article_date: str,
    event_name: str,
    event_description: str,
    journalist: str = "unknown",
) -> tuple["ExtractionOutput", dict]:
    """Returns (ExtractionOutput, usage) where usage has prompt_tokens/completion_tokens/total_tokens."""
    prompt = PROMPT.format(
        article_text=article_text,
        source_name=source_name,
        journalist=journalist,
        article_date=article_date,
        event_name=event_name,
        event_description=event_description,
    )
    return await complete_structured(
        settings.extractor_model, ExtractionOutput, prompt, max_tokens=1200, timeout=180,
    )
