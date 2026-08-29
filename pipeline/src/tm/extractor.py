import asyncio
import logging
import re
from datetime import date, timedelta
from typing import Optional

from .models import ExtractionOutput, PredictionExtraction
from .config import settings
from .llm import complete_structured

logger = logging.getLogger(__name__)

# Split into a fixed, cacheable PROMPT_PREFIX (identical on every single call, system-wide —
# no article, no prediction, no .format() placeholders in it) and a PROMPT_SUFFIX that
# carries the article body and the per-(article, prediction) variable fields, plus the
# output-format spec. Together they are byte-identical to the single PROMPT string this
# used to be — the split only changes how the two pieces are wired into the LLM call
# (llm.py::complete_structured's cached_prefix), not the prompt content itself.
PROMPT_PREFIX = """\
You are a forensic prediction analyst. Your job is to extract EVERY signal — \
explicit or implicit — that bears on whether the RELATED EVENT will occur.

## What counts as a signal

Extract ALL of the following:
- Explicit forecasts: "X will happen", "Y is expected to..."
- Implied directional views: "the economy is heading toward...", "pressure is mounting"
- Factual reports whose content logically implies an outcome — e.g. reporting that \
  troops are advancing implies a battle outcome; reporting that negotiations collapsed \
  implies a deal is less likely. INFER the implication — but infer only the implication \
  the report's own subject and target carry; a capability or an intent is not an \
  occurrence (see below).
- Quotes from officials, analysts, or experts that imply a position
- Vague sentiment that colors likelihood: "things are deteriorating", "a breakthrough \
  looks distant"
- Even near-zero certainty signals (claim_strength=0.1) are valuable — include them

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

## When there is no signal
If the article, read honestly, contains no sentence that implies THIS outcome is more or \
less likely, return an empty predictions list. Reporting that the underlying process or \
contest will take place (a date confirmed, a deadline announced, rules of procedure) is \
NOT a directional signal about which outcome it will produce. Do not manufacture a lean \
from neutral facts — stance 0.0 with low claim_strength, or extracting nothing at all, is a \
correct and valuable answer.

## Process evidence vs. outcome evidence
When the related event names one specific outcome of a multi-outcome process (one winner \
among contenders, one option among alternatives), distinguish two kinds of reporting:
- Evidence that the process will occur, stay on schedule, or follow its rules — this \
applies equally to every possible outcome and therefore says nothing about which one \
will happen. At most it bears weakly on a deadline component of the claim.
- Evidence favoring one outcome over the others — the only thing that moves stance \
materially.

Examples — related event: "Candidate A wins contest C by date D":
  "Contest C is confirmed to take place on schedule" → no extraction (occurrence ≠ outcome)
  "A major rival of Candidate A withdrew from contest C" → stance +0.4, claim_strength 0.5
  "Candidate A cleared the previous stage of contest C"  → stance +0.2, claim_strength 0.3

## Capability and intent are not occurrence — match the TARGET, not the skill
Evidence that a subject CAN do something, has done it to a DIFFERENT target, is \
building toward it, or says it INTENDS to do it, is not evidence that it has done or \
will do it to THIS target within THIS deadline. A demonstrated capability, a new \
weapon or product, a success against another target, a stated ambition, or a threat is \
a PRECONDITION of the related event, never the event itself: it raises likelihood \
weakly at most (|stance| <= 0.3, claim_strength <= 0.4) and is NEVER settled. The trap is an \
article about target B that showcases exactly the skill the claim needs against target \
A — the claim you write then has no target in it at all, which is the tell. Name the \
specific target, action and deadline in the related event and check the article reports \
THAT one; never let a capability, an intent, or a success against another target stand \
in for the occurrence the claim asks about.

Examples — related event: "Force F will successfully strike Bridge K by date D":
  "Force F has demonstrated the capability to destroy major bridges using upgraded munitions" \
                                               → stance +0.2, claim_strength 0.3, settled false (a capability, not a strike on Bridge K)
  "Force F struck a fuel depot and a military airfield overnight" \
                                               → no extraction (a different target — the skill is shared, the event is not)
  "Officials of Force F vowed Bridge K would be hit again" \
                                               → stance +0.3, claim_strength 0.3, settled false (stated intent, not an occurrence)
  "Explosions damaged Bridge K's roadway on Tuesday, halting traffic" \
                                               → stance +1.0, claim_strength 0.95, settled true, event_date resolved from "on Tuesday" (this target, this action)

Examples — related event: "Company X will launch a commercial quantum computer by 2027":
  "Company X demonstrated error correction on a 100-qubit test chip" \
                                               → stance +0.2, claim_strength 0.3, settled false (a capability milestone, not a commercial launch)
  "Company X opened orders for its first commercial quantum system" \
                                               → stance +0.9, claim_strength 0.8, settled false (the launch itself, imminent — not yet an accomplished fact)

## The capability/intent cap applies PER CLAIM, not to the article's overall urgency
Do not let an article that reads as urgent, on-topic, or saturated with intent signals lift the \
cap above. Five distinct threats, expectations, and preparations reported in the same article \
are still five capability/intent signals — each is capped individually at |stance| <= 0.3, and \
their number or density does not aggregate into occurrence. An article can be entirely ABOUT the \
possibility of a war without any single sentence in it reporting the war itself; extract each \
claim on its own modality (threat, expectation, preparation) and cap each one, regardless of how \
charged or imminent the surrounding coverage reads.

Example — related event: "Force F will engage in a significant military conflict with Force G by \
date D", one article reporting several accumulating signals:
  "Force F's defense minister threatens a strong response to any attack by Force G" \
                                               → stance +0.3, claim_strength 0.3 (a threat, not an attack)
  "Force F's security assessments expect senior Force G officials will order strikes" \
                                               → stance +0.3, claim_strength 0.3 (an expectation, not an occurrence)
  "A third party is preparing to escalate military attacks on Force G" \
                                               → stance +0.3, claim_strength 0.3 (another actor's preparation, not this conflict occurring)
Each stays capped even though all three appear in one urgent, on-topic article about the same \
brewing conflict — the aggregate reading of "this is clearly heading to war" is not itself a \
signal that qualifies for a higher cap.

## STANCE — the most important field
Stance measures how strongly this signal implies the RELATED EVENT will occur.
  +1.0 = certain the event WILL happen
  -1.0 = certain the event WILL NOT happen
   0.0 = neutral / no directional signal

Ask yourself: "If this quote/fact is true, does it make the related event more likely \
(positive stance) or less likely (negative stance)?"

Examples — related event: "Assad regime falls in Syria":
  "Rebel forces are closing in on Hama"        → stance +0.7, claim_strength 0.6
  "Assad's army is holding the line"           → stance −0.6, claim_strength 0.5
  "The conflict has dragged on for two years"  → stance +0.2, claim_strength 0.2
  "International sanctions remain in place"   → stance +0.3, claim_strength 0.3
  "Rebels have taken Damascus; Assad has fled the country" \
                                               → stance +1.0, claim_strength 0.95, settled true (+ event_date — see SETTLED)
  "Assad crushed the uprising; the rebellion is over" \
                                               → stance −1.0, claim_strength 0.95, settled true (+ event_date of the crushing if the article dates it — see SETTLED)

Note: even factual/contextual sentences have a stance if they imply a direction.
Do NOT use stance to indicate good/bad — only more/less likely to happen.

## Unverified claims by an interested party — cap claim_strength
A claim of fact made by a party TO the underlying dispute or conflict, about its OWN \
actions, casualties inflicted, or operational results — a belligerent's own damage or \
casualty count, a company's own success claim in a commercial dispute, a claimed strike \
outcome — carries claim_strength no higher than 0.5, however declaratively it reads, UNLESS \
the article ALSO reports independent confirmation (a different party, a neutral \
observer, satellite imagery, an official body). Wartime and dispute claims from an \
interested source are routinely inflated or unverifiable; the direction (stance sign) \
can still be correct and full stance magnitude still applies, but do not let declarative \
phrasing ("claims to have destroyed X targets") buy full confidence. This is the same \
VERIFIED vs CLAIMED judgement as the FACT_SIGNAL section below, applied here to \
claim_strength, which does feed the live estimate.

Examples — related event: "Maritime traffic through the Strait of Hormuz returns to \
pre-conflict normal levels by September 30":
  "Iran's Islamic Revolutionary Guard Corps claims to have destroyed 85 U.S. military \
targets in Bahrain and Qatar overnight" \
                                               → stance −0.556, claim_strength 0.4 (an interested party's own unconfirmed damage claim — sign follows the escalation, claim_strength capped)
  "Satellite imagery confirms extensive damage to the reported U.S. facilities in Bahrain" \
                                               → stance −0.6, claim_strength 0.8 (independently corroborated — no longer capped)

## Negated events — score the claim AS WRITTEN
When the related event is itself phrased as something NOT happening ("X will NOT \
happen", "no ceasefire will be reached", "X fails to pass", "X will remain below..."), \
stance measures the NEGATED statement as written, not the inner event. Evidence that \
the inner event is approaching or underway CONTRADICTS the related event → negative \
stance; evidence the inner event is receding or blocked SUPPORTS it → positive stance. \
Re-read the related event's exact wording before scoring and apply the flip yourself — \
never score the inner event and leave the negation to the reader.

Examples — related event: "A Russia-Ukraine ceasefire will NOT be implemented before November":
  "Deep strikes on refineries intensify; talks have collapsed" \
                                               → stance +0.6, claim_strength 0.6  (escalation SUPPORTS "no ceasefire")
  "Both sides agree on a framework for a truce" \
                                               → stance −0.7, claim_strength 0.6  (a ceasefire approaching CONTRADICTS the negated claim)

Examples — related event: "Inflation will NOT fall below 3 percent this year":
  "CPI drops to 2.9 percent in June"           → stance −1.0, claim_strength 0.9, settled true (the inner event occurred — the negated claim is settled FALSE)
  "CPI ticks up to 4.1 percent"                → stance +0.5, claim_strength 0.5

## Alarming or critical tone is not stance direction
A quote's emotional register — danger, tragedy, outrage, criticism — is not the signal. \
Read only what the claim's content asserts: if the alarming or critical fact described IS \
the outcome the related event asks about, stance follows the content, not the tone — an \
alarming quote can carry POSITIVE stance when its content affirms the claim. Two models \
reading the identical sentence must not land on opposite stance signs; if you find yourself \
scoring a quote negative because it "sounds bad" rather than because its content argues \
against the related event, re-read the claim's exact wording and check which way the \
content actually points.

Examples — related event: "Site S remains hazardous to human habitation for at least 100 years":
  "The reactor core residue is still molten beneath the plant; its presence keeps the city \
unsafe to resettle, and will for at least the next century" \
                                               → stance +1.0, claim_strength 0.9 (alarming tone, but the content directly affirms the claim — do not score it negative because "danger" reads as bad news)

## Numeric thresholds — compare the numbers, not the sentiment
When the related event states a quantitative threshold ("more than 33 seats", \
"below $50,000", "at least 10 medals", "reaches 2800 rating"), judge each signal \
by COMPARING its number against the threshold — never by momentum or by how \
positive the news sounds for the subject. A reported or projected value on the \
wrong side of the threshold is NEGATIVE stance even when it is good news for \
the subject; general success without a number that clears the bar is at most \
weakly positive.

Examples — related event: "Likud wins more than 33 seats in the election":
  "Poll projects Likud at 31 seats"        → stance −0.6, claim_strength 0.7  (31 ≤ 33: contradicts)
  "Likud is leading in the polls"          → stance +0.2, claim_strength 0.3  (leading ≠ >33 seats)
  "Poll gives Likud 36 seats"              → stance +0.7, claim_strength 0.7  (36 > 33: supports)
  "Likud gained two seats since last poll" → stance +0.2, claim_strength 0.3  (trend, no level given)

## Multi-stage / bracket events — discount single-stage "favorite" framing
When the related event requires winning a SEQUENCE of separate future contests \
(a tournament bracket, a playoff series, a multi-round election, a series of \
confirmation votes) rather than one determination, an article's "favorite," \
"front-runner," or "strong candidate" framing about ONE upcoming stage is weak \
support for the event as a whole — it says nothing about the stages still to \
come. Advancing past one stage narrows the field but does not itself imply the \
final outcome; only raise stance as stages actually clear, and reserve high \
claim_strength for articles that address the full remaining path, not just the next \
match or round.

Examples — related event: "France wins the 2026 World Cup" (tournament bracket):
  "France is a strong favorite entering the Round of 16"     → stance +0.3, claim_strength 0.3  (one stage of several remaining)
  "France beats Paraguay to reach the quarter-finals"        → stance +0.4, claim_strength 0.5  (one stage cleared, more remain)
  "France reaches the final after a dominant semi-final win" → stance +0.6, claim_strength 0.6  (one stage left)

Examples — related event: "Judge Alvarez is confirmed to the Supreme Court" (committee vote, then floor vote):
  "Alvarez is seen as the clear favorite to be confirmed"       → stance +0.3, claim_strength 0.3  (favorite framing, no vote yet)
  "The Judiciary Committee advances Alvarez's nomination 12-10" → stance +0.4, claim_strength 0.5  (one stage cleared, floor vote remains)

Examples — related event: "Diaz wins the presidential runoff" (first round, then runoff):
  "Diaz leads first-round polling by 8 points"        → stance +0.2, claim_strength 0.3  (first round ≠ runoff win)
  "Diaz advances to the runoff after finishing first" → stance +0.4, claim_strength 0.5  (one stage cleared, runoff remains)

## Single-winner contests — a rival's win settles the claim NO
When the related event names ONE subject winning a contest that can have only one \
winner (a tournament, a race, an election to a single office), a report that a \
DIFFERENT contestant achieved that outcome — or eliminated the subject from \
contention — is not merely bad news for the subject: it settles the related event \
NEGATIVELY. The subject's outcome is now permanently impossible: stance −1.0, \
claim_strength ≥ 0.9, settled true, event_date = the date of the foreclosing result (see \
SETTLED). The stance belongs to the SUBJECT of the related event, not to whoever the \
article celebrates — never read the excitement of a decisive result as support for \
the contestant it eliminated. A defeat that does NOT eliminate the subject (a \
group-stage loss, a setback with a path remaining) is ordinary negative-lean \
evidence, never settled.

Examples — related event: "France wins the 2026 World Cup" (article dated Wednesday 2026-07-15):
  "Spain beat France 2-0 in Tuesday's semi-final to reach the final" \
                                               → stance −1.0, claim_strength 0.95, settled true, event_date "2026-07-14", event_date_reference "Tuesday's" (France eliminated — the outcome is permanently impossible; the article's subject is Spain's win, but the stance is about FRANCE)
  "France lost their opening group match 0-1" \
                                               → stance −0.4, claim_strength 0.4, settled false (a non-terminal loss — France can still advance)

Examples — related event: "England will win their World Cup semi-final on 2026-07-15":
  "Argentina stun England with a late rally to reach the final" \
                                               → stance −1.0, claim_strength 0.95, settled true, event_date "2026-07-15" (England's semi-final is decided — and lost; the triumphant tone is Argentina's, NOT support for England)

## Cited quantitative estimates — extract them as a distinct anchor
When the article itself cites an explicit modeled, polled, or market-priced \
PROBABILITY OF THE RELATED EVENT ITSELF (not a proxy stage) — e.g. "a model gives \
Team X an 18.83% chance to win the tournament", "the prediction market prices the \
deal at 33%", "the forecaster puts the odds of an agreement at 45%" — extract that \
figure into `quantitative_estimate` as a probability in [0, 1] (convert percentages: \
18.83% → 0.1883). Set `stance` to match it (`stance = 2 × quantitative_estimate − 1`) \
and `claim_strength` high (≥ 0.8) — a named model or market is a much stronger \
anchor than qualitative "favorite"/"strong candidate" framing, even when several \
qualitative articles exist alongside it. A vote share, poll share, seat count, or \
seat projection is NOT a probability of the event — "the party polls at 28%" is a \
share of the vote, not a 28% chance of winning: leave `quantitative_estimate` null \
for those, score their stance by comparing the figure against the claim's threshold \
(see Numeric thresholds above), and classify them `cited_share` below. Leave \
`quantitative_estimate` null when the article has no such explicit cited figure — \
general "leading in the polls" or "seen as the favorite" language without a stated \
number is NOT a quantitative estimate; keep using the sections above for that. Also \
leave it null for a CASUAL or CONVERSATIONAL figure of speech — a pundit, fan, coach, \
or player tossing out "I'd give it a 50-50 chance" or "there's maybe a 90% chance" is \
voicing a personal opinion, not citing a model/poll/market; only a NAMED formal \
source counts.

Examples — related event: "France wins the 2026 World Cup":
  "Simulations by Opta give France the best chance of winning the tournament, at 18.83%" \
                                               → stance −0.62, claim_strength 0.85, quantitative_estimate 0.1883
  "Betting markets rank France as favorites to lift the trophy" \
                                               → stance +0.3, claim_strength 0.3, quantitative_estimate null (no number given)
  "The team's own coach joked there's maybe a 90% chance they choke again" \
                                               → stance −0.3, claim_strength 0.3, quantitative_estimate null (casual personal opinion, not a named model/poll/market)

Examples — related event: "Likud wins more than 33 seats in the election":
  "A poll-aggregator model gives Likud a 22% chance of winning more than 33 seats" \
                                               → stance −0.56, claim_strength 0.85, quantitative_estimate 0.22
  "The latest poll puts Likud at 28% of the vote" \
                                               → stance −0.5, claim_strength 0.7, quantitative_estimate null (a vote SHARE, not a chance of the event — compare against the threshold, classify cited_share)
  "Likud is seen as gaining momentum heading into the vote" \
                                               → stance +0.2, claim_strength 0.3, quantitative_estimate null (momentum, no cited figure)

## EVIDENCE CLASS — optional; classify the KIND of evidence this claim is
Classify it independently and honestly; do not let it influence stance or
claim_strength, and vice versa. If a claim genuinely does not fit one category
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
                        of the vote, not a 28% chance of winning. A share
                        must NEVER populate `quantitative_estimate` (that
                        field is only for genuine probabilities of the
                        event itself — see Cited quantitative estimates).
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
set settled to true and use the full ±1.0 stance with claim_strength ≥ 0.9. Past-tense \
reporting of the outcome ("X won", "the deal was signed", "Y has died") is settled; \
predictions, odds, and expectations ("X is likely to win") are NOT settled, however \
confident. Do not soften a settled outcome into a likelihood — a report that the \
event happened is stance +1.0, not +0.7.

### A settlement that the event OCCURRED must be dated — no event_date, no settled
A positive settlement (stance +1.0: the event happened) asserts that THIS question's \
outcome has verifiably occurred, and that assertion must be anchored to a calendar \
date: every such claim MUST also carry event_date — the absolute date on which the \
event itself occurred, resolved exactly as described in the DATES section below. If \
the article does not let you date the occurrence, settled must be false: \
accomplished-fact language with no discoverable date is the signature of historical \
background about an earlier episode (a standing government, a title won in a past \
season, a long-ago deal), not news of this question's outcome. Report your honest \
stance, but do not settle. Never substitute the article's own publication date for \
an event the article does not actually date. \
A negative settlement (stance −1.0: the event became permanently impossible) is dated \
by the FORECLOSING event instead: when the article dates the result that made the \
outcome impossible — the rival's win, the elimination, the subject's death, the \
withdrawal — put THAT date in event_date (and its verbatim relative expression, if \
any, in event_date_reference). When the impossibility comes only from time expiring, \
or the article gives no date for the foreclosing event, leave event_date empty — \
never invent one.

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

Examples — related event: "Messi scores at least 8 goals in the tournament" \
(article dated Monday 2026-06-22):
  "Messi bagged his ninth goal of the tournament in Saturday's rout; the group stage continues" \
                                               → event_date "2026-06-20", stance +1.0, claim_strength 0.95, settled true (9 ≥ 8: already locked in, dated by the ninth goal)
  "Messi and Mbappe are tied for the tournament lead with 6 goals each, group stage ongoing" \
                                               → stance −0.3, claim_strength 0.4, settled false (6 < 8, contest still open — a tally, not a verdict)
  "The tournament concluded on Sunday; Messi finished with 7 goals" \
                                               → stance −1.0, claim_strength 0.95, settled true, event_date "2026-06-21", event_date_reference "on Sunday" (contest over, 7 < 8 is now permanent — dated by the tournament's conclusion, the event that foreclosed the 8th goal)

### Buried facts — extract settlement even when it's incidental to the article's main topic
A clear past-tense statement of the RELATED EVENT can appear as a single supporting \
clause inside an article whose main subject is something else entirely (e.g. a piece \
about downstream diplomatic fallout that mentions, in passing, that the event already \
happened). Scan the WHOLE article, not just the headline or opening paragraph — extract \
the fact and mark settled true regardless of how minor its role in the article is. \
Do not require the fact to be the article's primary subject to count it as settled. \
A statement of capability, intent, or a similar event elsewhere is not a past-tense \
report of THIS event — see the capability section above.

Examples — related event: "Peter Magyar will officially assume the role of Prime Minister \
of Hungary by December 31, 2026":
  Article mainly about Ukraine-Hungary pipeline relations, mentioning in passing: \
  "...its leader Peter Magyar became Prime Minister on May 9" \
                                               → event_date "2026-05-09", stance +1.0, claim_strength 0.95, settled true \
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
                                               → stance −0.1, claim_strength 0.4, settled false \
                                                 (background history predating the question's window — not this question's outcome)
  "The State Department formally approved the F-35 sale to Turkey on Tuesday" \
                                               → stance +1.0, claim_strength 0.95, settled true, event_date resolved from \
                                                 "Tuesday" against the article's date (this question's outcome, reported as fact)

## MATCH THE EVENT — do not credit a near-miss as the event
Before assigning |stance| >= 0.9 or settled=true, decompose the RELATED EVENT into WHO \
(the subject, including its type — a person, a party, a company, a country, an \
institution), WHAT (the exact action or outcome), and WITHIN WHAT SCOPE (threshold, \
deadline, arena). The reported fact must match ALL three — a near-miss on any one of \
them is evidence, but it is not the event. Two recurring ways a fact can miss the match, \
below: a different subject/action/arena, or the right kind of action by the wrong named \
party.

### A different subject type, action, or arena is ADJACENT evidence
A fact about:
- a DIFFERENT SUBJECT TYPE — a member of the organization when the claim is about the \
organization itself, a subsidiary when the claim is about the parent, an official when \
the claim is about the government;
- a SIMILAR BUT DIFFERENT ACTION — leaving an organization vs. the organization \
withdrawing from a contest, resigning a post vs. the body being dissolved, suspending a \
program vs. cancelling it;
- or a DIFFERENT ARENA — primaries vs. the general election, a qualifier vs. the final;
is ADJACENT evidence. Score its real bearing on likelihood honestly (typically |stance| \
<= 0.5), but it is NEVER settled and never carries the full +-1.0, no matter how \
definitively it is reported. The test: could a fact-checker cite this article alone as \
proof that the related event itself occurred? If not, it is not settled.

Examples — related event: "At least one party withdraws from the parliamentary race":
  "MK X announced he is leaving Party Y and won't run in its primaries"     → stance +0.3, claim_strength 0.5, settled false (a member leaving a party is not a party leaving the race)
  "Party Y announced it will not submit a candidate list"                   → stance +1.0, claim_strength 0.95, settled true (+ event_date of the announcement)

Examples — related event: "Company X exits the European market by year-end":
  "Company X's CEO resigned amid the European losses"                       → stance +0.2, claim_strength 0.4, settled false (leadership change is not a market exit)
  "Company X announced the closure of all European operations"              → stance +1.0, claim_strength 0.95, settled true (+ event_date of the announcement)

### A named-actor claim needs the NAMED actor and target, not just the same conflict
When the related event names SPECIFIC parties (a named country, company, person, or \
team) in specific roles — actor and/or target — a fact about a DIFFERENT party in the \
SAME broader conflict, alliance, or industry performing the identical kind of action is \
NOT evidence about the named parties, however similar the action or however clearly it \
escalates the same underlying dispute. A regional war widening to a new belligerent, a \
new company entering an industry dispute, or a different official making a similar move \
does not confirm — and barely moves — a claim that requires THESE SPECIFIC parties. \
Check the actor and target BY NAME, not by category or by "is this the same conflict": \
"the US and Iran" is not "Israel and Iran"; "Iran strikes Jordan" is not "Iran strikes \
Israel", even on the same night of the same crisis. This is NEVER settled, and its \
bearing on the named pair is weak context at most (|stance| <= 0.2, claim_strength <= 0.3) — \
a claim asking whether X and Y fight is not "satisfied" by a report that Y is fighting \
someone else.

Examples — related event: "Israel and Iran engage in direct military conflict by December 31, 2026":
  "Two US soldiers were killed in an Iranian attack on a base in Jordan"     → stance +0.15, claim_strength 0.2, settled false (the US and Jordan, not Israel — a wider war does not confirm this specific pair)
  "IRGC missiles struck US targets in Kuwait and Bahrain overnight"         → stance +0.15, claim_strength 0.2, settled false (still not Israel; regional escalation raises the odds only weakly)
  "The Israeli Air Force struck IRGC missile sites near Tehran"             → stance +1.0, claim_strength 0.95, settled true (+ event_date) (Israel and Iran, matching the claim exactly)

### A date does not excuse a near-miss — adjacency still applies to dated facts
The event_date requirement in the DATES section below is a floor for a fact that has \
ALREADY passed the match tests above, never a substitute for it. A dated fact about a \
DIFFERENT event — a predecessor's term, a different official's action, a similar event \
in another context — does not become a settlement for THIS claim just because it \
carries a clean, verifiable date. Decide the match FIRST (same subject, same action, \
same scope, same named actor/target where it applies), then check whether that matched \
fact is dated. Never work the order backwards: finding a date is not evidence that you \
found the right event, and a precisely dated adjacent fact is still adjacent, never \
settled, exactly like an undated one.

## DATES — resolve first, compare second, never assert a comparison you did not compute
Deadline claims ("by July 15", "before year-end") are decided by ARITHMETIC, not by tone. \
An article can be euphoric that the event is certain and still be evidence AGAINST the \
claim, if the date it names falls after the deadline. Certainty that the event happens \
LATE is certainty the claim is FALSE.

Two rules, in order:

1. RESOLVE. A relative date reference — "on Friday", "today", "tomorrow", "next week", \
"this weekend", "in three days" — is not a date. Convert it to an absolute calendar date \
using the article's date, given below. Put that absolute date in `event_date` (YYYY-MM-DD). \
Never carry a weekday forward as if it were a date, and never assume a weekday lands on \
the deadline: if the article says "Friday" and the deadline is the 15th, "Friday" is \
whatever date it actually is — work it out from the article's date. Whenever you resolve \
a relative reference this way, ALSO copy the article's verbatim expression into \
`event_date_reference` (e.g. "on Friday", "yesterday") — code re-does the calendar \
arithmetic from it and corrects `event_date` when the two disagree. Omit \
`event_date_reference` when the article names the absolute date outright.

2. COMPARE. Only once you have the absolute date, compare it with the claim's deadline \
(given below as "Claim deadline"). State the absolute date in the `claim` field, not the \
weekday, so the comparison is auditable.
  - event_date on or before the deadline → the claim is SUPPORTED (positive stance)
  - event_date after the deadline        → the claim is CONTRADICTED (negative stance), \
however affirmative the article sounds

Set `event_date` whenever the article gives a date for the RELATED EVENT ITSELF — omit it \
for the date of an adjacent or downstream event (the election that follows a dissolution, \
the trial that follows an indictment). It is REQUIRED on every POSITIVE settled=true \
claim (see SETTLED above): a settlement that the event occurred, which you cannot \
date, is not a settlement.

Examples — related event: "The Israeli parliament will be dissolved by July 15, 2026" \
(article dated Monday 2026-07-13):
  "The Knesset will dissolve on Friday" \
    → "Friday" is 2026-07-17 (the Friday after Monday the 13th), which is AFTER July 15 \
    → event_date "2026-07-17", event_date_reference "on Friday", stance −1.0, claim_strength 0.95 \
      claim: "The parliament will be dissolved on 2026-07-17, after the July 15 deadline" \
    (WRONG: reading "Friday" as "by July 15" and returning +1.0 — the event is certain, \
     but it is certain to happen TOO LATE, which contradicts the claim)
  "The Knesset dissolved yesterday" \
    → "yesterday" is 2026-07-12, on or before July 15 \
    → event_date "2026-07-12", event_date_reference "yesterday", stance +1.0, claim_strength 0.95, settled true

## Article language
The article may be in Hebrew, Arabic, or English. Always write the claim in English.
Quote the original language verbatim in the quote field.

## AUTHOR_LEAN — the byline author's own forecast (for scoring the author, NOT the estimate)
Separately from everything above, judge whether the BYLINE author or outlet named below \
(Source / Journalist) is THEMSELVES forecasting the related event — stating a position of \
their own — as opposed to neutrally reporting facts or relaying other people's views. This \
field exists to hold that author accountable later; it does NOT feed the event estimate, so \
keep it independent of stance and never let one influence the other.
  author_lean = the byline author's OWN directional forecast of the related event: +1 the \
author expects it to happen, -1 the author expects it will NOT happen, 0 the author \
explicitly weighs both sides and commits to neither. This is the direction the author \
expects the event to RESOLVE, not whether they welcome it: an author who condemns, warns \
against, or laments an event while treating it as happening or inevitable is still \
forecasting that it WILL happen — lean +1 toward it, never negative. Approval or alarm about \
an outcome is sentiment, not a directional forecast, and must not flip the sign. Cross-check \
against your own extracted claims below: if this author's claims already affirm the event is \
happening (a positive stance), author_lean must not read negative — and if their claims deny \
it, author_lean must not read positive. Alarm, outrage, or criticism about a DIFFERENT \
development, a related event's consequences, or this event's implications is still not a \
reason to move author_lean opposite to what the author's own claims about THIS event already \
established.
  author_lean_certainty = how firmly the author commits to that forecast (0 hedged, 1 emphatic).
Return null for both (omit them) when the byline author only reports what happened or relays \
other people's views without endorsing a direction — a straight news report has no \
author_lean. A prediction made by a QUOTED third party — an official, an analyst, a pundit \
the article cites — is that person's position, not the byline's, and must NOT be recorded \
here. Multiple sources agreeing on the same direction is still THEIR consensus, not the \
byline author's — an article that merely stacks concordant quoted forecasts has author_lean \
null unless the byline author asserts or endorses a direction in their own voice. When \
unsure whether a view is the author's own or a source's, treat it as the source's \
and return null.

## FACT_SIGNAL — what the reported FACTS alone imply (EXPERIMENTAL, shadow — separate from stance)
Separately from stance, and used by no estimate yet, record what the article's REPORTED \
FACTS on their own establish about the related event — stripped of the author's assertion, \
of quoted opinion, and of interpretive framing. Where stance may blend "what is asserted" \
with "what the facts show", fact_signal is ONLY the second: +1 the facts establish the event \
happened or is happening, -1 the facts establish it will not or cannot, 0 the facts bear on \
it but point neither way. Return null (omit fact_signal and its facets) when the prediction \
rests on opinion, advocacy, or expectation with no reported fact that bears on the event. \
WHENEVER you omit fact_signal, also record fact_signal_absent_reason so the null itself \
stays honest — a consumer must be able to tell "nothing found" from "something found that \
points the other way": opinion — the claim rests on opinion/advocacy/expectation with no \
reported fact bearing on the event; no_fact_found — nothing in the article bears on the \
event's occurrence either way; contrary_below_anchor — a reported fact DOES point against \
the event but is too weak, ambiguous, or off-dyad to anchor a graded negative value. Reserve \
contrary_below_anchor for the genuine remainder, not as an escape hatch from grading — per \
NEGATIVE PRECURSORS below, most contrary facts should be graded, not nulled.
The one EXCEPTION — DECIDER STATEMENTS: a public, on-record statement by the decider — the \
actor or authority whose own act or announcement would itself resolve the claim, including \
a senior official speaking for that authority — is itself a reported fact about intent, \
however rhetorical or dismissive its phrasing, and must never be nulled as opinion. A \
stated intent or commitment to act is a positive precursor of the event; a denial, refusal, \
or ruling-out is a negative precursor — both under the same OCCURRENCE vs PRECURSOR rule as \
any physical precursor, with weaker or hedged expressions of intent scoring smaller than \
firm commitments. For such a statement, verified means the statement itself was \
independently reported as made. An assertion about the decider's intent by an opponent, \
analyst, or unnamed source is not a decider statement and remains \
claimed-and-unverified at most.
NEGATIVE PRECURSORS — the graded scale runs in BOTH directions, and not only for decider \
statements: a reported fact that makes the event less likely — an obstacle emerging, a \
preparation reversed or abandoned, a contrary or rival development, a measured indicator \
moving against the event — bears on the event and is scored as a graded negative precursor \
(is_occurrence false, same cap), never nulled merely because it points against the claim. \
Reserve the extreme negative for facts that establish the event cannot happen, exactly as \
the extreme positive is reserved for the event itself having occurred; between the \
extremes, grade contrary facts with the same discipline as supporting ones.
DEADLINE-DEFERRED INTENT — a reported fact or statement that pushes the event's timing to a \
point you can place AFTER the claim deadline bears AGAINST a by-deadline claim, and must be \
graded like any other negative precursor, not nulled — even when the deferral is anchored to \
another named milestone or event ("after the elections", "once the review concludes") rather \
than a calendar date the DATES section's arithmetic can compare directly. You do not need a \
resolvable event_date to score this: recognizing that the referenced milestone falls at or \
after the deadline IS the fact. A firm, unconditional deferral grades stronger than a vague or \
hedged one, under the same discipline as any other negative precursor.
Discipline fact_signal by four tests, and record the facets that justify each:
  - FACET. Classify the reported fact as announcement (it establishes the event happening or \
having happened), denial (it establishes the event will not or did not happen), or neither \
(it bears on the event without asserting either polarity — a precursor, a capability, an \
escalation). A decider statement of intent or commitment is an announcement; a decider's \
denial, refusal, or ruling-out is a denial. A poll, survey, or seat/vote-share projection is \
never announcement or denial, however lopsided its numbers — it is neither, since it reports \
what respondents or a model estimate, not a decider's own statement or the event itself; \
grade its bearing on the event through fact_signal's magnitude, not through facet. Omit \
facet when fact_signal itself is omitted.
  - DYAD. Name WHO acts (event_actors) and the TARGET of the action (event_target) in the \
fact. A fact whose actor-target pair is NOT the claim's pair — a strike by a different \
country, on a different country — is context only: keep |fact_signal| small and never treat \
it as the event occurring, however forceful the fact.
  - OCCURRENCE vs PRECURSOR. Set is_occurrence true only when the fact IS the event itself \
(or its definitive outcome); set it false when the fact is a precondition, mobilisation, \
capability, or escalation that merely precedes the event. A precursor never scores as the \
event occurring, no matter how sustained, repeated, or intensifying it is — a conflict \
escalating over many days, or a preparation repeated night after night, is still not the \
discrete event happening.
  - VERIFIED vs CLAIMED. Set verified true when the fact is independently reported as having \
happened; set it false when only an interested or belligerent party CLAIMS it and no \
independent source confirms. A claimed-but-unverified event is down-weighted, not scored at \
full strength.
These facets are shadow fields for a future estimator; keep them honest and independent of \
stance — never let fact_signal pull stance, or stance pull fact_signal.

## READER_CONFIDENCE — your confidence in your OWN reading (EXPERIMENTAL, shadow)
Every field above records something about the article. This one records something about YOU: \
how far you would stand behind the reading you just produced for this span. It is NOT the \
source's hedging — that is claim_strength, and the two are independent. A flat, categorical \
sentence you had to work to interpret is high claim_strength with a LOW reader_confidence; a \
heavily hedged sentence whose direction is obvious is low claim_strength with a HIGH one. When \
you find yourself about to lower claim_strength because YOU were unsure, lower this instead.

Set reader_confidence.level by COUNTING the resolution steps between this span and the related \
event — not by how sure you feel. A step is any of: resolving a comparison, a referent, a date, \
a scope, or carrying a fact across from a neighbouring actor, target or arena.
  high   — zero steps. The span states the outcome of the related event directly.
  medium — exactly one step.
  low    — two or more steps; OR the span never mentions the related event and the whole link \
is your inference; OR two different stance signs are each defensible from this span.
Count the steps you actually took. `low` is a normal answer for a span reached by inference — \
it is a property of the distance you crossed, not an admission that you got it wrong.

Then answer separately: which ONE of these applies to your reading of THIS span? Set \
reader_confidence.trap to it, or omit trap when none of them does.
  negation                 — the meaning turns on a "not" / "no" / "fails to" / "remains \
below", or the related event is itself phrased as something not happening, and getting the \
polarity right took work.
  numeric_comparison       — you decided the direction by comparing numbers (a level against a \
threshold, a count against a target), not by reading a direction off the words.
  entity_or_event_mismatch — the span is about a neighbouring actor, target, arena or event, \
and you had to judge how far it carries to the related event as written.
  tone_vs_content          — the span's tone points one way and its factual content the other.
  inference_needed         — the span does not address the related event directly; reaching it \
required a reasoning step of your own.
  conflicting_signals      — the span carries two indications that point in opposite \
directions.

level and trap are independent: a trap does not force a low level, and no trap does not force a \
high one. Naming the trap you navigated is the useful part, and you may well have navigated it \
confidently. Report the reading you actually did — do not lower level to look cautious, or \
raise it to look decisive. Set reader_confidence on EVERY prediction, including the easy ones, \
where it is simply level high with no trap.

Examples — related event: "Force F will successfully strike Bridge K by date D":
  "Explosions damaged Bridge K's roadway on Tuesday, halting traffic"
      → level high (zero steps; trap omitted)
  "Force F ruled out striking Bridge K before the corridor talks conclude"
      → level medium, trap "negation" (one step: resolve the polarity)
  "Force F massed 40 launchers near the corridor, short of the 60 its doctrine requires"
      → level medium, trap "numeric_comparison" (one step: 40 against the 60 threshold)
  "Force F's commander said the corridor campaign is going to plan"
      → level low, trap "inference_needed" (the span never mentions Bridge K; the link to a \
strike on it is entirely inferred)
  "Force G struck Bridge K's approach road overnight"
      → level low, trap "entity_or_event_mismatch" (two steps: carry across from Force G to \
Force F, and from the approach road to the bridge)

## REPORT_KIND — a standing situation, or a step in it (EXPERIMENTAL, shadow)
For each prediction, say whether the quote reports a LEVEL or a CHANGE.
  level  — the standing situation as it is: "the reservoir is at 41% of capacity", "the \
border post remains closed", "as of today the line is still not in service". A direct \
measurement of the state.
  change — a movement in it: "the reservoir fell six points", "the border post reopened", \
"the line was extended by two stops". A step, not the state.
Omit report_kind when the quote is neither — a pure expectation about the future with no \
present state and no movement reported ("hydrologists see a further drop coming").

The test is what the sentence would still tell you a month later, not its verb tense. \
"Operators held the reservoir at 41%" is a level: the number is the point, and holding is \
the absence of a step. "Operators drew it down by six points" is a change: without knowing \
where it started, the new state is unknown.

report_kind never changes your stance. It says what KIND of report the quote is, not how \
strongly it bears on the event; a level that satisfies the question is exactly as positive \
as a change that satisfies it.

## CONSENSUS_VIEW — what the article says OTHERS expect (EXPERIMENTAL, shadow)
Once per article, not per prediction. Report what the ARTICLE says most observers — analysts, \
markets, polls, "widely expected", "few believe" — expect for the related event.
  expects_yes — the article reports that most expect it to happen.
  expects_no  — the article reports that most expect it will not.
  divided     — the article reports opinion as genuinely split.
Omit consensus_view when the article does not say what anyone else expects.

Three things this is NOT, and each of them is the common mistake:
  - It is not YOUR view. You are reporting what the article claims about other people, and \
you record it even when you think those people are wrong.
  - It is not the byline author's own forecast — that is author_lean. "Forecasters expect \
the reservoir to refill, but this is wishful thinking" is consensus_view expects_yes with a \
NEGATIVE author_lean.
  - It is not the stance of the quotes you extracted. An article may carry one discouraging \
quote and still report that most observers expect yes; record what it says about them.
Report it from the article's own words about others, not from counting your own predictions.
Like report_kind, it never changes a stance: extract the predictions exactly as you would \
have without this field.

## Output
Extract up to 5 signals. Prefer higher-certainty ones but do not omit low-certainty \
signals if they are the only content available.
The quote field must contain the sentence that implies the direction of the stance. \
If no such sentence exists in the article, the stance must be 0.0.
"""

PROMPT_SUFFIX = """\
Article:
<article>
{article_text}
</article>

Source: {source_name}
Journalist: {journalist}
Date: {article_date}
Related event: {event_name} — {event_description}
Claim deadline: {claim_deadline}

IMPORTANT: Your response must be a JSON object with a "predictions" key containing a list, \
plus OPTIONAL top-level "author_lean" (float -1 to 1) and "author_lean_certainty" (float 0 \
to 1) fields — the byline author's OWN forecast, per the AUTHOR_LEAN section. OMIT both when \
the author takes no position of their own. Also OPTIONAL at top level: "consensus_view" (one \
of expects_yes / expects_no / divided) — what the article says OTHERS expect, per the \
CONSENSUS_VIEW section; OMIT it when the article does not say.
Example: {{"predictions": [ {{...}}, {{...}} ], "author_lean": 0.6, "author_lean_certainty": \
0.5, "consensus_view": "expects_yes"}}

Each prediction has five core fields, plus several used only when applicable:
  quote (string — original language), claim (string — English), \
stance (float −1 to 1), claim_strength (float 0 to 1), settled (boolean — true only when \
the source reports the outcome as an accomplished fact), quantitative_estimate \
(float 0 to 1, OMIT this field entirely unless the source cites an explicit modeled/ \
market/polled PROBABILITY of the event itself — never a vote share or seat count, \
see the section above), \
evidence_class (one of reported_fact / cited_probability / cited_share / reporting / \
opinion, OMIT this field entirely if none fits cleanly — see the section above), \
event_date (string YYYY-MM-DD — the absolute date the article gives for the RELATED \
EVENT ITSELF, with any relative reference like "Friday" already resolved against the \
article's date; OMIT this field entirely when the article states no date for it — \
see the DATES section above), \
event_date_reference (string — the article's VERBATIM relative expression behind \
event_date, e.g. "on Friday" or "yesterday"; OMIT it when the article names the \
absolute date outright — see the DATES section above)

The following are EXPERIMENTAL shadow fields — include them together per the FACT_SIGNAL \
section whenever a reported fact bears on the event: \
fact_signal (float −1 to 1 — what the reported facts alone imply about the event), \
facet (one of announcement / denial / neither — see FACET in the FACT_SIGNAL section), \
event_actors (string — who acts in that fact), event_target (string — the target of the \
action), is_occurrence (boolean — true only when the fact IS the event itself, false for a \
precursor/precondition/escalation), verified (boolean — true when independently reported, \
false when only claimed by an interested party). \
When you OMIT fact_signal (and the facets above with it), include \
fact_signal_absent_reason instead (one of opinion / no_fact_found / contrary_below_anchor — \
see the FACT_SIGNAL section for which applies) — never omit both.

Also on every prediction: reader_confidence — \
{{"level": one of high / medium / low, "trap": one of negation / numeric_comparison / \
entity_or_event_mismatch / tone_vs_content / inference_needed / conflicting_signals, OMITTED \
when none applies}}. See the READER_CONFIDENCE section above. \
And report_kind (one of level / change — does the quote report the standing situation or a \
step in it; OMIT it when the quote reports neither, per the REPORT_KIND section).

Example — related event: "Assad regime falls in Syria":
{{
  "predictions": [
    {{
      "quote": "Syrian rebel forces pushed close on Tuesday to the major city of Hama",
      "claim": "Rebel advances toward Hama make Assad's fall increasingly likely",
      "stance": 0.7,
      "claim_strength": 0.6,
      "settled": false,
      "evidence_class": "reporting",
      "reader_confidence": {{"level": "medium", "trap": "inference_needed"}},
      "report_kind": "change"
    }},
    {{
      "quote": "Rebels seized the capital on Sunday as Assad fled to Moscow",
      "claim": "The Assad regime has fallen; rebels control Damascus",
      "stance": 1.0,
      "claim_strength": 0.95,
      "settled": true,
      "evidence_class": "reported_fact",
      "fact_signal": 1.0,
      "facet": "announcement",
      "event_actors": "Syrian rebel forces",
      "event_target": "Damascus and the Assad regime",
      "is_occurrence": true,
      "verified": true,
      "reader_confidence": {{"level": "high"}},
      "report_kind": "change"
    }}
  ],
  "consensus_view": "expects_yes"
}}

Example — related event: "France wins the 2026 World Cup" (a source citing a named model):
{{
  "predictions": [
    {{
      "quote": "Simulations by Opta indicate France has the highest chance of winning the 2026 World Cup at 18.83%",
      "claim": "Opta's model gives France an 18.83% chance to win the tournament",
      "stance": -0.62,
      "claim_strength": 0.85,
      "settled": false,
      "quantitative_estimate": 0.1883,
      "evidence_class": "cited_probability",
      "fact_signal_absent_reason": "opinion",
      "reader_confidence": {{"level": "medium", "trap": "numeric_comparison"}},
      "report_kind": "level"
    }}
  ],
  "consensus_view": "expects_no"
}}
"""


# Appended ONLY for short_form callers (retro#297). The "### Buried facts" rule above tells the
# model to scan the WHOLE article and credit a decisive clause however incidental — correct for a
# long article, and actively wrong for a terse multi-topic social post: from one IRGC post it
# minted claims about Ukrainian attrition strategy AND the S&P 500 CAPE ratio (pushed at cosine
# 0.03-0.09). A short post doesn't bury facts; either it speaks about the related event or it
# doesn't.
#
# Opt-in and append-only, exactly like gatekeeper._SHORT_FORM_OVERRIDE (#264/#266): the prompt
# every existing caller sends is byte-for-byte unchanged (test_extractor_short_form.py).
_SHORT_FORM_OVERRIDE = """

**Short-form source.** This item is a terse social-media / messaging post (e.g. a journalist's
Telegram channel), not a full news article. A post this short has one primary topic and no
buried facts: the "Buried facts" whole-article scan above is written for long articles and does
NOT apply here. Extract signals only when the post's own primary topic bears on the related
event. A passing mention, an aside, or a list item on an unrelated subject is not evidence about
the related event — if the post as a whole is about a different matter, extract nothing rather
than stretch a phrase into a claim.
"""


# Appended when the caller knows the article's language (retro#417) — same rationale and
# append-only contract as gatekeeper._LANGUAGE_HINT. The quote field already asks for the
# original language; this only tells the model up front what that language IS.
_LANGUAGE_HINT = """

**Language.** The article text is in {language}. Extract from it directly: `quote` stays \
verbatim in the original language, `claim` is your English rendering, exactly as specified \
above.
"""

# --- Conditional extraction (Phase 1 capture plan; conditional-capture-phase1.md §3.2) ---
# Lexical pre-filter: cheap check for conditional language before requesting extraction.
# Gated at the prompt level: when lexicon matches, the CONDITIONAL instruction block is
# included; when it doesn't, the block is omitted and the model is expected to null all
# 9 conditional fields. 5% bypass probe (random 5% of non-matching articles get the block
# anyway) measures the pre-filter's false-negative rate.

CONDITIONAL_LEXICON = frozenset({
    'if', 'unless', 'should', 'provided', 'were', 'in the event',
    'absent', 'barring', 'contingent', 'depends', 'assuming', 'so long as'
})

def has_conditional_language(text: str) -> bool:
    """Cheap lexical pre-filter: check if text contains conditional keywords.

    Word-boundary check (\\b) to avoid matching "if" in "life", "depends" in "independent", etc.
    Case-insensitive. Returns True if ANY keyword is found.
    """
    if not text:
        return False
    text_lower = text.lower()
    # Split into words and check for matches
    words = re.findall(r'\b\w+\b', text_lower)
    return bool(CONDITIONAL_LEXICON & set(words))

# Appended when conditional language is detected in the article (or on the 5% bypass probe).
# Instruction block for extracting conditional fields; all 9 fields are nullable, so this
# block is purely informational — when omitted, the model defaults them to null.
# PRE-RESOLUTION: these fields are recorded BEFORE the enforce_* chain (conditional-capture-phase1.md §3.3).
_CONDITIONAL_BLOCK = """

## CONDITIONAL (v1.1 — Phase 1 capture)

When this section appears, extract conditional claims: assertions whose truth or relevance is \
contingent on an antecedent. All fields below are OPTIONAL and nullable.

**is_conditional** (bool): True when this claim is conditional on an antecedent.

**antecedent_text** (string): The VERBATIM "if"-clause or conditional expression as written \
in the article, IN THE ARTICLE'S ORIGINAL LANGUAGE. Examples: "if the ceasefire collapses", \
"unless negotiations succeed", "were Trump to withdraw support". Copy exactly as it appears; \
do not paraphrase or translate here.

**antecedent_text_en** (string): RESTATE the antecedent as a standalone ENGLISH PROPOSITION, \
stated POSITIVELY. This is the ONLY field used for embedding/linking. Negation lives in \
`antecedent_polarity`, NOT in this field — one canonical form so negations cluster. Example: \
if the article says "if X does NOT happen", store `antecedent_text_en: "X happens"` and \
`antecedent_polarity: false`. If the antecedent is already affirmative ("if the election is \
held"), `antecedent_text_en: "the election is held"` and `antecedent_polarity: true` or null.

**antecedent_polarity** (bool): False when the antecedent is negated ("if X does NOT happen"). \
True or null for affirmative form. This field captures the negation so all affirmative statements \
in `antecedent_text_en` cluster together, and negation is separated into one field.

**relation** (string, enum): How the antecedent relates to the consequent. One of:
  - "raises" — the antecedent makes the consequent MORE likely (evidential).
  - "lowers" — the antecedent makes the consequent LESS likely (evidential).
  - "requires" — the antecedent is NECESSARY for the consequent (logical constraint).
  - "precludes" — the antecedent makes the consequent IMPOSSIBLE (logical constraint).
  - "unclear" — the direction or type of dependence is ambiguous.

**strength** (string, enum): Source's stated strength when no explicit probability is given. \
One of: "certain", "likely", "possible", "unlikely". Omit if the source gives a numerical \
probability instead; use `stated_probability` for those.

**stated_probability** (float, 0-1): P(consequent | antecedent) when the source explicitly \
states a number. Example: "analysts put it at 70% if the ceasefire holds". Null if the source \
does not quantify the conditional probability.

**is_counterfactual** (bool): True for "had X not happened" or "if X had happened" — past-directed, \
different epistemic object than a forward conditional. False or null for forward-looking conditionals.

**speaker** (string): Who asserted the conditional — outlet name (e.g. "Reuters", "BBC") or analyst's \
name if the claim is a direct quote (e.g. "General Smith", "Analyst John Doe"). For attribution.

### Examples

**Example 1: Causal conditional (raises)**
  Article: "If the ceasefire holds, Likud is expected to gain 15 seats in the next election, analysts say."
  Claim: "Likud gains 15 seats"
  is_conditional: true
  antecedent_text: "if the ceasefire holds"
  antecedent_text_en: "the ceasefire holds"
  antecedent_polarity: true
  relation: "raises"
  strength: null
  stated_probability: null
  is_counterfactual: false
  speaker: "Analysts"

**Example 2: Negated antecedent (lowers)**
  Article: "Unless a diplomatic breakthrough occurs, escalation looks inevitable."
  Claim: "Escalation occurs"
  is_conditional: true
  antecedent_text: "unless a diplomatic breakthrough occurs"
  antecedent_text_en: "a diplomatic breakthrough occurs"
  antecedent_polarity: false
  relation: "lowers"
  strength: null
  stated_probability: null
  is_counterfactual: false
  speaker: null

**Example 3: Explicit probability (requires)**
  Article: "A trade war would require the withdrawal of China's ambassador, sources say."
  Claim: "China withdraws ambassador"
  is_conditional: true
  antecedent_text: "if a trade war occurs"
  antecedent_text_en: "a trade war occurs"
  antecedent_polarity: true
  relation: "requires"
  strength: null
  stated_probability: null
  is_counterfactual: false
  speaker: "Sources"

When there is no conditional language, leave all nine fields null (do not include them, or \
set them to null — either is correct).
"""


async def extract_predictions(
    article_text: str,
    source_name: str,
    article_date: str,
    event_name: str,
    event_description: str,
    journalist: str = "unknown",
    claim_deadline: Optional[str] = None,
    short_form: bool = False,
    language: Optional[str] = None,
    include_conditional_block: Optional[bool] = None,
    is_single_article: bool = False,
    cache_coordinator: Optional["CacheWriteCoordinator"] = None,
    model: Optional[str] = None,
) -> tuple["ExtractionOutput", dict]:
    """Returns (ExtractionOutput, usage) where usage has prompt_tokens/completion_tokens/total_tokens.

    ``claim_deadline`` (ISO date) is rendered into the prompt so the model can compare a
    resolved ``event_date`` against it rather than hunting the deadline out of the claim's
    prose. Callers that don't classify claims may omit it — the prompt then says "not given"
    and behaviour is unchanged.

    ``short_form`` opts into scoping a social-media post to its own primary topic instead of
    the whole-article buried-facts scan (retro#297). It defaults to False and only ever
    APPENDS to the prompt, so the text every existing caller sends is byte-for-byte unchanged.

    ``language`` (retro#417) is an optional caller-supplied hint ("Hebrew", "ru", …) appended
    to the prompt so the model is told the text is non-English instead of having to notice.
    Also append-only; None keeps the prompt unchanged.

    ``include_conditional_block`` (v1.1, Phase 1 capture) optionally includes the conditional
    extraction instruction block. When None (default), the block is conditionally included based
    on lexical pre-filter (has_conditional_language). When True, always include. When False, never
    include. Append-only; None/False keeps existing behavior unchanged for backward compat.

    ``is_single_article`` (retro#564) skips prompt caching when true (single article in a request
    has no cache reuse opportunity within the 5-minute TTL). Defaults to False for backward compat.

    ``cache_coordinator`` (retro#564) when provided, gates ONLY the first extractor call in a
    request: that call writes the cache while everyone else waits, then all remaining calls
    proceed concurrently reading from the now-warm cache. Unlike a plain Semaphore(1), this does
    not serialize the whole batch — only the single write is on the critical path, preserving the
    batch's original parallelism for every call after the first. None means no coordination
    (backward compat).

    ``model`` (retro#652) overrides ``settings.extractor_model`` for this call only. None (the
    default) keeps the configured global. A per-request opt-in, not a policy — callers who want a
    different model/cost tradeoff (e.g. a benchmark harness with a wider latency budget) pass one
    in; nothing here decides what a caller should choose.
    """
    prompt = PROMPT_SUFFIX.format(
        article_text=article_text,
        source_name=source_name,
        journalist=journalist,
        article_date=article_date,
        event_name=event_name,
        event_description=event_description,
        claim_deadline=claim_deadline or "not given",
    )
    if short_form:
        prompt += _SHORT_FORM_OVERRIDE
    if language:
        prompt += _LANGUAGE_HINT.format(language=language)

    # Conditional block: lexical pre-filter gates the instruction block.
    # When None (default): check text for conditional lexicon; include if found.
    # When True: always include (used for 5% bypass probe).
    # When False: never include (backward compat; default if lexicon check fails).
    if include_conditional_block is None:
        include_conditional_block = has_conditional_language(article_text)
    if include_conditional_block:
        prompt += _CONDITIONAL_BLOCK

    async def _call_extractor():
        return await complete_structured(
            model or settings.extractor_model, ExtractionOutput, prompt, max_tokens=1200, timeout=180,
            cached_prefix=None if is_single_article else PROMPT_PREFIX,
        )

    if cache_coordinator is not None:
        return await cache_coordinator.run(_call_extractor)
    return await _call_extractor()


class CacheWriteCoordinator:
    """Per-request coordinator (retro#564): the first ``run()`` call executes immediately and
    primes the cache; every other call in the same request waits for that one to finish, then
    proceeds concurrently (no further serialization) reading from the now-warm cache.

    A plain ``asyncio.Semaphore(1)`` held around the whole call would serialize every article's
    extractor call end-to-end, turning N parallel LLM round-trips into a sequential chain — this
    only puts the *first* write on the critical path.

    Construct one instance per forecast request (state is not safe to share across requests).
    """

    def __init__(self) -> None:
        self._claim_lock = asyncio.Lock()
        self._claimed = False
        self._primed = asyncio.Event()

    async def run(self, call):
        async with self._claim_lock:
            is_first = not self._claimed
            self._claimed = True
        if is_first:
            try:
                return await call()
            finally:
                self._primed.set()
        await self._primed.wait()
        return await call()


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    """Lenient ISO-8601 date parse — accepts a bare date or a full timestamp. None on anything else."""
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except (ValueError, TypeError):
        return None


_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_DAY_WORDS = {"today": 0, "tonight": 0, "yesterday": -1, "tomorrow": 1}

_WEEKDAY_REFERENCE = re.compile(
    r"^(?:on\s+|this\s+|the\s+coming\s+|coming\s+|last\s+)?(%s)$" % "|".join(_WEEKDAYS)
)


def _resolve_relative_reference(reference: str, article: date) -> Optional[date]:
    """The absolute date a relative expression denotes, or None when out of vocabulary.

    Vocabulary is deliberately small: today/tonight/yesterday/tomorrow, and a weekday
    with an optional on/this/coming/last modifier. A bare (or on/this/coming) weekday
    means the next occurrence, same-day allowed; "last <weekday>" the previous one.
    "next <weekday>" is deliberately NOT handled — speakers disagree on whether it means
    this week's or the following week's occurrence, and a guard built on an ambiguous
    reading would inject the very errors it exists to catch. English only: a quote in
    another language falls out of vocabulary and fails open.
    """
    text = re.sub(r"[.,;:!?\"'«»]+", " ", reference.lower()).strip()
    text = re.sub(r"\s+", " ", text)
    if text in _DAY_WORDS:
        return article + timedelta(days=_DAY_WORDS[text])
    m = _WEEKDAY_REFERENCE.match(text)
    if not m:
        return None
    target = _WEEKDAYS[m.group(1)]
    if text.startswith("last "):
        back = (article.weekday() - target) % 7 or 7
        return article - timedelta(days=back)
    return article + timedelta(days=(target - article.weekday()) % 7)


def enforce_relative_date_resolution(
    predictions: list[PredictionExtraction],
    article_date: Optional[str],
) -> list[PredictionExtraction]:
    """Redo the model's relative-date arithmetic in code, and trust the code.

    The prompt asks the extractor to resolve "on Friday" against the article's date AND
    to copy the verbatim expression into ``event_date_reference``. The resolution step is
    exactly what LLMs get wrong with confidence: the Knesset incident's extractor mapped
    "Friday" (article dated Monday 2026-07-13) to the deadline itself on 5 of 5 runs, and
    post-fix still offered 2026-07-18 — a Saturday. When the copied expression is in our
    small vocabulary, this walks the calendar itself and overrides a disagreeing
    ``event_date``, which ``enforce_deadline_arithmetic`` and
    ``enforce_settlement_event_date`` then consume.

    Fail-open in every direction: no reference, no parseable article date, an
    out-of-vocabulary expression, or a missing ``event_date`` leaves the prediction
    untouched. A reference without a model-committed ``event_date`` never CREATES one —
    settlement gating must stay anchored to a date the model itself asserted.
    """
    article = _parse_iso_date(article_date)
    if article is None:
        return predictions

    for p in predictions:
        if not p.event_date_reference or not p.event_date:
            continue
        resolved = _resolve_relative_reference(p.event_date_reference, article)
        if resolved is None:
            continue
        model_date = _parse_iso_date(p.event_date)
        if model_date == resolved:
            continue
        logger.warning(
            "event=relative_date_override reference=%r article_date=%s "
            "event_date=%s -> %s claim=%r",
            p.event_date_reference, article.isoformat(),
            p.event_date, resolved.isoformat(), p.claim[:120],
        )
        p.event_date = resolved.isoformat()

    return predictions


def enforce_deadline_arithmetic(
    predictions: list[PredictionExtraction],
    claim_deadline: Optional[str],
    claim_direction: Optional[str],
) -> list[PredictionExtraction]:
    """Let arithmetic, not the LLM, decide the SIGN of a confidently-dated deadline claim.

    A deadline claim is settled by comparing two dates. LLMs are unreliable at exactly that,
    and they fail *confidently*: asked whether a parliament dissolving "on Friday" met a
    July 15 deadline, the extractor returned stance +1.0 / certainty 0.95 on 5 of 5 runs —
    once rendering the claim as "dissolved on Friday, July 15", snapping the weekday onto
    the deadline. That Friday was July 17. The same model, given an article that spelled out
    "July 17" in plain text, returned −1.0 on 4 of 4 runs. The only difference between a
    right and a wrong answer was whether the article did the arithmetic for it.

    So: the model reports the date (``event_date``), and we do the comparison here.

        arrival  ("X happens BY D"):     event_date <= D → supports (+) ; after D → contradicts (−)
        survival ("X does NOT happen by D"): mirrored.

    Only *confident* signals are corrected (|stance| >= 0.9 or settled) — those are the ones
    that pin an estimate, and a hedged "might slip past the deadline" is a genuine judgement
    we have no business overriding. Magnitude and certainty are preserved; only the sign moves.

    One carve-out: a SETTLED NEGATIVE on an ARRIVAL claim is exempt. Its ``event_date`` is
    the FORECLOSING event's (the rival's win, the elimination — see the SETTLED prompt
    section), not this claim's own occurrence, so the comparison above would read a correct
    impossibility verdict dated within the deadline as "the event occurred in time" and flip
    it to a false YES (the France-elimination trap). On a SURVIVAL claim a settled negative
    means the underlying event *did* occur — its date is the occurrence itself, and the
    arithmetic stays valid.

    Fail-open in every direction: no deadline, no ``event_date``, an unparseable date, or an
    unclassified claim leaves the prediction exactly as the model returned it.
    """
    deadline = _parse_iso_date(claim_deadline)
    if deadline is None or claim_direction not in ("arrival", "survival"):
        return predictions

    for p in predictions:
        event_date = _parse_iso_date(p.event_date)
        if event_date is None:
            continue
        if abs(p.stance) < 0.9 and not p.settled:
            continue
        if p.settled and p.stance < 0 and claim_direction == "arrival":
            # Dated foreclosure, not a dated occurrence — see docstring.
            continue

        within = event_date <= deadline
        expects_positive = within if claim_direction == "arrival" else not within
        if (p.stance > 0) == expects_positive:
            continue

        logger.warning(
            "event=deadline_arithmetic_override claim_direction=%s deadline=%s event_date=%s "
            "stance=%+.2f -> %+.2f settled=%s claim=%r",
            claim_direction, deadline.isoformat(), event_date.isoformat(),
            p.stance, -p.stance, p.settled, p.claim[:120],
        )
        p.stance = -p.stance

    return predictions


def enforce_settlement_event_date(
    predictions: list[PredictionExtraction],
    article_date: Optional[str],
    claim_created_at: Optional[str] = None,
) -> list[PredictionExtraction]:
    """A settlement vote must be anchored to a date the outcome occurred.

    ``settled=true`` is the highest-impact bit the extractor emits: once
    ``settlement_min_sources`` of them agree, the pooled estimate is pinned to
    ±settlement_stance (aggregation.py). The 2026-07-15 Netanyahu false pin rode
    on exactly two such votes — accomplished-fact language about the sitting
    64-seat coalition (formed after the PREVIOUS election) settling a claim
    about the NEXT one. Neither article dated the "outcome", because the outcome
    they described wasn't this question's: undatable past-tense language is the
    signature of historical background, which the prompt already forbids — but
    prompts are advisory, so this is the enforcement.

    A POSITIVE settlement (stance > 0: the event occurred) must date the
    occurrence itself. A NEGATIVE settlement dates the FORECLOSING event when
    the article dates it (the rival's win, the elimination — see the SETTLED
    prompt section), but the date stays optional: an impossibility that comes
    only from time expiring has nothing to date, and premature negative pins
    have their own guard (``settlement_direction_allowed``, aggregation.py —
    and, at aggregation time, per-vote revalidation). Note
    :func:`enforce_deadline_arithmetic` exempts settled arrival-claim negatives
    for exactly this reason: their date is a foreclosure, not an occurrence.

    Deterministic checks; a claim failing one keeps its stance and certainty
    (it still votes as ordinary evidence) but loses ``settled``:

      - POSITIVE with no parseable ``event_date``: unanchored — demote.
      - ``event_date`` (either sign) after the article's own date: the article
        "reports" an outcome that hadn't happened yet when it was written — a
        scheduled event, not an accomplished fact — demote.
      - ``event_date`` (either sign) BEFORE ``claim_created_at``: the outcome is
        dated to before the question was asked, so it cannot be that question's
        outcome — nobody asks whether something will happen by 2026 about an
        event that happened in 2022 — demote.

    That last check is not new policy. ``aggregation.settlement_vote_validity``
    has applied it on every archetype since 2026-08-16, under the same reason
    string (``event_before_claim_window``), and its docstring names this exact
    class: "a dated fact from before the claim existed: the 2021/2022-article
    class". What was new (retro#704) is that the rule lived ONLY at vote time, so
    the extractor kept writing ``settled=true`` on rows the pooling layer then
    silently discounted — 144 of the 215 adjacent settlements in the retro#691
    labelled set are of exactly this shape. Applying it here changes no pooled
    estimate; it makes the STORED bit agree with the vote already being cast.

    Strict ``<`` at date granularity, matching aggregation, so an event on the
    claim's creation day stays valid. Fails open on an absent or unparseable
    ``claim_created_at`` — also matching aggregation.

    Unlike :func:`enforce_deadline_arithmetic` this deliberately fails CLOSED on
    a positive settlement's missing date. The cost of a wrong demotion is a
    slower settlement pin (the estimate still moves on the stance); the cost of
    a wrong settlement is a market stuck at 97% on history — asymmetric, so the
    date is mandatory there.
    Missing/unparseable ``article_date`` skips only the future-dated check.
    """
    article = _parse_iso_date(article_date)
    created = _parse_iso_date(claim_created_at)
    for p in predictions:
        if not p.settled:
            continue
        event_date = _parse_iso_date(p.event_date)
        if p.stance > 0 and event_date is None:
            reason = "missing_event_date"
        elif event_date is not None and article is not None and event_date > article:
            reason = "event_date_after_article"
        elif event_date is not None and created is not None and event_date < created:
            # Same reason string aggregation uses, deliberately: one grep should
            # find both layers, and the two must never disagree about the rule.
            reason = "event_before_claim_window"
        else:
            continue
        logger.warning(
            "event=settlement_demoted reason=%s event_date=%s article_date=%s "
            "claim_created_at=%s stance=%+.2f certainty=%.2f claim=%r",
            reason, p.event_date, article_date, claim_created_at or "",
            p.stance, p.claim_strength, p.claim[:120],
        )
        p.settled = False

    return predictions


def enforce_precursor_cap(
    predictions: list[PredictionExtraction],
) -> list[PredictionExtraction]:
    """A precursor's ``fact_signal`` may not exceed the precursor cap.

    The OCCURRENCE-vs-PRECURSOR rule in the prompt caps a fact that merely precedes
    the event at ``|0.3|`` "no matter how sustained, repeated, or intensifying it is
    — a conflict escalating over many days, or a preparation repeated night after
    night, is still not the discrete event happening". Nothing enforced it. A live
    audit of the pool (2026-08-01, retro#367) found **269 of 1101** precursor rows —
    24.4% — above the cap, reaching ``|0.90|``; and since the stored value is the
    claim-weighted MEAN over an article's claims while ``is_occurrence`` comes from
    the single dominant one, that 24.4% is a floor on the per-claim breach rate.
    Four more over-cap emissions were observed directly during the WS5/WS5b A/B runs
    (recorded on retro#354).

    So the model reports *whether* the fact is the event; the magnitude contract is
    enforced here. Only magnitude moves — the sign a precursor points is a genuine
    judgement, how far it may push the estimate is policy (``fact_signal_precursor_cap``,
    config.py). Every other field is untouched: a clamped claim keeps its stance,
    certainty and facets and votes exactly as before in the stance lane.

    Runs before fusion, which fixes a second-order effect too: the dominant fact —
    the max-``|fact_signal|`` claim, whose facets are the ones stored for the whole
    article (forecaster.py) — can no longer be captured by an over-cap precursor
    outranking a genuine occurrence claim.

    Fail-open in every direction: a null ``fact_signal``, or an ``is_occurrence`` that
    is null (the extractor did not judge) or true (the fact IS the event), leaves the
    claim exactly as the model returned it. Only an explicitly-marked precursor is in
    scope — this never invents a judgement the model declined to make.
    """
    cap = settings.fact_signal_precursor_cap
    for p in predictions:
        if p.fact_signal is None or p.is_occurrence is not False:
            continue
        if abs(p.fact_signal) <= cap:
            continue
        clamped = cap if p.fact_signal > 0 else -cap
        logger.warning(
            "event=precursor_cap_clamped fact_signal=%+.2f -> %+.2f cap=%.2f "
            "verified=%s claim=%r",
            p.fact_signal, clamped, cap, p.verified, p.claim[:120],
        )
        p.fact_signal = clamped

    return predictions


def enforce_interested_party_stance_cap(
    predictions: list[PredictionExtraction],
) -> list[PredictionExtraction]:
    """An interested party's unverified assertion may not vote at full magnitude.

    The prompt's interested-party rule (``extractor.py``'s VERIFIED vs CLAIMED
    section) caps ``certainty`` at 0.5 on a claim only an interested party asserts,
    while stating that "full stance magnitude still applies". That is a weight-only
    discount, and a vote's location in the pool is stance ALONE
    (``stance_to_prob = (stance + 1) / 2``, aggregation.py) — so under normalization
    it cancels: N unverified interested-party claims at stance +1 still pool to
    +0.99. The one rule written to discount self-serving claims had no effect in
    precisely the case it exists for. This closes the location side (retro#368, F20).

    **Keyed on ``verified`` directly.** The issue proposed keying on the prompt's
    implied signature — high ``|stance|`` with low ``certainty`` — and warned not to
    assume it held. The prod audit (2026-08-01) found it does not, and is in fact
    inverted: on 1,418 pool rows carrying the marker, ``|stance|`` and ``certainty``
    are strongly COUPLED on unverified claims (corr **+0.68**, against +0.79 on
    verified ones), unverified rows sit LOWER on both axes (avg ``|stance|`` 0.374
    vs 0.431, avg certainty 0.456 vs 0.547), and the proposed key fired on **1 row
    in 1,418** — which was ``verified=true``. No cell in the threshold sweep was both
    precise and material. ``verified`` is the extractor's own marker and needs no
    inference.

    Only magnitude moves, and only on the stance axis. The sign an interested party
    points is a genuine judgement — a company denying a merger is evidence about the
    merger — while how far an unverified assertion may push the estimate is policy
    (``interested_party_stance_cap``, config.py). Certainty, class and every facet
    are untouched here; the certainty cap the prompt already promises is retro#378.

    Fail-open in the same asymmetry as :func:`enforce_precursor_cap`: a ``verified``
    of ``None`` — the extractor did not judge, which is the case on 87% of live pool
    rows, since the marker is populated only on extractions since 2026-07-09 and was
    never backfilled — or ``True`` leaves the claim exactly as the model returned it.
    Only an explicitly-marked unverified claim is in scope; this never invents a
    judgement the model declined to make.

    One interaction, declared: ``resolve_stance_certainty`` runs later, at
    aggregation, and re-derives stance from a cited figure for
    ``cited_probability`` claims — so it can override this clamp for a claim that is
    both ``verified=false`` and a provenance-passing ``cited_probability``. That has
    **zero live instances** (all 5 ``cited_probability`` rows in the prod pool are
    ``verified=true``), and is arguably right anyway: a checkable market figure is
    not an interested party's assertion, whoever repeated it.
    """
    cap = settings.interested_party_stance_cap
    for p in predictions:
        if p.verified is not False:
            continue
        if abs(p.stance) <= cap:
            continue
        clamped = cap if p.stance > 0 else -cap
        logger.warning(
            "event=interested_party_stance_clamped stance=%+.2f -> %+.2f cap=%.2f "
            "certainty=%.2f evidence_class=%s claim=%r",
            p.stance, clamped, cap, p.claim_strength, p.evidence_class, p.claim[:120],
        )
        p.stance = clamped

    return predictions


def enforce_interested_party_certainty(
    predictions: list[PredictionExtraction],
) -> list[PredictionExtraction]:
    """The weight-side half of the interested-party rule (retro#378, F20 family).

    The prompt's VERIFIED vs CLAIMED section says an unverified interested-party
    claim "carries claim_strength no higher than 0.5, however declaratively it reads".
    Nothing checked it. Measured on prod (2026-08-01/02, `evidence_pool_articles`
    rows carrying the marker): **56 of 185 ``verified=false`` rows — 30.3% —
    exceed the cap**, max 0.733, with ten sitting at exactly 0.70 carrying an
    average ``|stance|`` of 0.76. As with :func:`enforce_precursor_cap` (24.4%),
    a numeric the prompt has taught for months is simply not held by the prompt.

    That 30.3% is a **floor** on the per-claim rate. The stored ``certainty`` is
    the article-level reduction while ``verified`` is the dominant claim's, so a
    single over-cap interested-party claim diluted by in-contract siblings never
    shows up in that count. Per-claim data is persisted as of retro#364, so the
    real rate becomes measurable going forward.

    Why this is a separate function from
    :func:`enforce_interested_party_stance_cap` rather than two lines in one:
    they act on different axes with different consequences and must produce
    separately attributable R8 movement. Stance is a vote's LOCATION in the pool;
    certainty is its WEIGHT. The audit separated them on evidence — a
    weight-only discount fully cancels under normalization in a unanimous pool,
    which is why the stance half was needed at all, and the unanimous-pool case
    has zero live instances (0 of 71 multi-article predictions have an entirely
    unverified pool; 38 are mixed, 33 all-verified). So in current traffic this
    cap DOES bite, in those 38 mixed pools, where an unverified claim competes
    against verified ones and its weight decides how far it pulls the mean.

    Unlike the stance cap, the number here is not new policy: it is the prompt's
    own literal, and ``test_extractor_prompt.py`` pins the two together so they
    cannot drift.

    Fail-open in the same asymmetry as the rest of the chain: ``verified`` of
    ``None`` — the extractor did not judge, true of ~87% of live pool rows, since
    the marker is populated only on extractions since 2026-07-09 and was never
    backfilled — or ``True`` leaves the claim untouched. This cap is therefore a
    no-op on historical rows and can only be validated forward, never
    retrospectively.

    One ordering note: ``resolve_stance_certainty`` runs later, at aggregation,
    and can re-derive BOTH stance and certainty (flooring it at 0.9) for a
    provenance-passing ``cited_probability`` claim — so it can override this cap
    for a claim that is both ``verified=false`` and a checkable cited figure.
    Zero live instances (all ``cited_probability`` rows in the prod pool are
    ``verified=true``), and defensible on the same grounds as there: a checkable
    market figure is not an interested party's assertion, whoever repeated it.
    """
    cap = settings.interested_party_certainty_cap
    for p in predictions:
        if p.verified is not False:
            continue
        if p.claim_strength <= cap:
            continue
        logger.warning(
            "event=interested_party_certainty_clamped certainty=%.2f -> %.2f "
            "cap=%.2f stance=%+.2f evidence_class=%s claim=%r",
            p.claim_strength, cap, cap, p.stance, p.evidence_class, p.claim[:120],
        )
        p.claim_strength = cap

    return predictions


def enforce_decider_intent_stance_cap(
    predictions: list[PredictionExtraction],
) -> list[PredictionExtraction]:
    """A decider's stated future intent may not vote at full stance magnitude.

    The fact lane already treats a decider's on-record commitment as a capped
    precursor: the FACET rule in the prompt marks "a decider statement of intent
    or commitment" as an ``announcement`` and "a denial, refusal, or ruling-out"
    as a ``denial``, and :func:`enforce_precursor_cap` clamps its ``fact_signal``
    to ±0.3. The stance lane — the lane a vote's location actually comes from —
    had no guardrail for the same rows: :func:`enforce_interested_party_stance_cap`
    keys on ``verified=false``, and a decider's own statement is usually
    ``verified=true`` (the statement demonstrably happened) or unjudged. So
    "I intend to win" was ±0.3 evidence in one lane and up to ±0.85 in the other
    (retro#518, surfaced by the Netanyahu/Le Monde case, elections#141).

    Prod audit (2026-08-15): 119 pool rows carry ``is_occurrence=false`` with
    facet ``announcement``/``denial``; 71 of them — 60% — vote above the cap,
    reaching |0.85|, and every over-cap row sits against a ``fact_signal``
    already clamped to ±0.3. The population is dominated by exactly the contract
    shape: ministers' "we will not withdraw" statements at 0.69–0.83,
    "Netanyahu stated there will be no Palestinian state" at 0.70.

    Keyed on the extractor's own markers (``is_occurrence`` + ``facet``), never
    an inferred signature — the same lesson as F20 (the high-|stance|/
    low-certainty key was tested and defeated, retro#368). Known collateral,
    measured and accepted: the model sometimes labels polls and measured
    indicators ``announcement``/``denial`` where the facet contract says
    ``neither`` (~4 of the 14 worst over-cap rows in the audit); the fix for
    that is facet labeling in the prompt, not a narrower key here.

    Forward-only by decision: ``facet`` exists on 1.9% of stored pool rows
    (shipped 2026-08-10, never backfilled), so this covers new extractions and
    legacy rows age out — the ``verified``-marker precedent.

    Only stance moves, sign preserved — a decider committing to act is genuine
    evidence about the event; how far their say-so may push the estimate is
    policy (``decider_intent_stance_cap``, config.py — deliberately equal to
    ``fact_signal_precursor_cap``, and a separate constant from the fact-lane
    ``decider_statement_*_cap`` knobs reserved for retro#486). Fail-open in the
    same asymmetry as the siblings: ``is_occurrence`` of None (unjudged) or True
    (the fact IS the event), or a facet of None/``neither``, leaves the claim
    exactly as the model returned it.
    """
    cap = settings.decider_intent_stance_cap
    for p in predictions:
        if p.is_occurrence is not False:
            continue
        if p.facet not in ("announcement", "denial"):
            continue
        if abs(p.stance) <= cap:
            continue
        clamped = cap if p.stance > 0 else -cap
        logger.warning(
            "event=decider_intent_stance_clamped stance=%+.2f -> %+.2f cap=%.2f "
            "facet=%s verified=%s certainty=%.2f claim=%r",
            p.stance, clamped, cap, p.facet, p.verified, p.claim_strength, p.claim[:120],
        )
        p.stance = clamped

    return predictions


def _names_allowlisted_source(text: str) -> Optional[str]:
    """The allowlisted source named in ``text``, or None. Word-boundary and
    case-insensitive; internal spaces match any run of whitespace so a name
    broken across a line still matches."""
    for name in settings.cited_probability_source_allowlist:
        pattern = r"\b" + re.escape(name).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            return name
    return None


def enforce_anchor_provenance(
    predictions: list[PredictionExtraction],
) -> list[PredictionExtraction]:
    """``cited_probability`` must name a source whose figure could be verified.

    That class carries the largest weight in the table (4.0) and authorizes the
    stance rewrite (``resolve_stance_certainty``), and nothing checks where the
    number came from. So it is the cheapest thing in the system to fabricate:
    land one sentence of "a market prices this at 80%" in any article we crawl
    and you have bought the strongest evidence class there is. The prompt's own
    canonical example — "a poll-aggregator model gives Likud a 22% chance" —
    names nobody, which is precisely the shape at issue.

    Prod audit (2026-08-01, retro#369) over all 16 ``cited_probability`` rows,
    mean evidence_weight 2.34, the highest of any class: about ten genuinely
    name a checkable source (Opta ×5, Kalshi ×4, Polymarket), and about six are
    not cited probabilities at all — Goldman Sachs' "$110 Brent by year-end",
    Citigroup's "$82,000 price target", and two carrying no figure whatsoever
    ("Fed rate hikes are increasingly likely"). One check does both jobs: a
    claim naming no verifiable source is not an anchor, whether because the
    number was invented or because there was never a probability there at all.

    Deliberately fails CLOSED — the same asymmetry :func:`enforce_settlement_event_date`
    applies to an undated positive settlement. An unverifiable premium is the
    exposure itself, so absence of provenance costs the premium; the claim keeps
    its stance and certainty and still votes as ordinary evidence.

    Permanent, not interim. This was written expecting R5's provenance axis to
    make "who stands behind this" a stored field, at which point the function
    would be deleted rather than migrated. That axis was rejected on 2026-08-10
    (retro#479, closing #402): `evidence_class` stays a flat 5-class enum, so the
    text scan IS the mechanism.

    Which makes its known false-*retention* path a permanent accepted limit, not
    a countdown: the scan reads the whole quote, so a quote naming an allowlisted
    source *alongside* an unrelated percentage passes and keeps both the 4.0 and
    the stance rewrite. Closing that means tightening this scan — proximity
    between the name and the number — not waiting for a field.

    Shadow by default. With ``anchor_provenance_enforced`` off, the check runs
    and logs every claim it *would* demote but changes nothing — which is what
    produces the firing rate the demotion target should be chosen against.
    """
    for p in predictions:
        if p.evidence_class != "cited_probability":
            continue
        source = _names_allowlisted_source(p.quote or p.claim or "")
        if source is not None:
            continue
        logger.warning(
            "event=anchor_provenance_unattributed enforced=%s demote_to=%s "
            "qe=%s stance=%+.2f claim=%r",
            settings.anchor_provenance_enforced,
            settings.unattributed_probability_class,
            p.quantitative_estimate, p.stance, p.claim[:120],
        )
        if settings.anchor_provenance_enforced:
            p.evidence_class = settings.unattributed_probability_class

    return predictions


# A claim text asserting a deontic/certainty marker ("is mandatory", "must") reads as
# supporting its own occurrence; one asserting an explicit negation/refusal reads as
# opposing it. Deliberately small and literal — this is an observability signal, not a
# semantic verifier (retro#298 explicitly scoped a full fix as needing a second LLM
# pass or a verifier stage; the general "does this stance follow from this claim" case
# stays unimplemented). It will miss subtler mismatches like a demand read as adversarial
# when it is actually a climb-down (retro#298's own row 6451) — only literal marker
# clashes are in scope.
_CLAIM_SUPPORT_MARKERS = [
    re.compile(p) for p in (
        r"\bis mandatory\b", r"\bmust\b", r"\bis required\b", r"\bis obligated\b",
        r"\bis guaranteed\b", r"\bis inevitable\b",
    )
]
_CLAIM_OPPOSE_MARKERS = [
    re.compile(p) for p in (
        r"\bwill not\b", r"\bwon'?t\b", r"\brefuses to\b", r"\brejects\b",
        r"\bdenies\b", r"\bis impossible\b", r"\bruled out\b",
    )
]


def flag_claim_stance_sign_conflicts(
    predictions: list[PredictionExtraction],
) -> list[PredictionExtraction]:
    """Log (never correct) claims whose own text and stance sign disagree — retro#298.

    retro#298 found rows where the extracted ``claim`` reads as supporting the related
    event while ``stance`` is negative, or vice versa — e.g. a claim stating a
    withdrawal "is mandatory" scored stance -0.136. The issue's own "cheap partial"
    suggestion: flag rows where the claim text contains an explicit support/oppose
    marker and the stance sign disagrees, purely as an observability signal, before
    committing to an LLM verifier stage. Predictions are returned unchanged.
    """
    for p in predictions:
        claim_lower = p.claim.lower()
        has_support = any(m.search(claim_lower) for m in _CLAIM_SUPPORT_MARKERS)
        has_oppose = any(m.search(claim_lower) for m in _CLAIM_OPPOSE_MARKERS)
        if has_support and not has_oppose and p.stance < -0.1:
            logger.warning(
                "event=claim_stance_sign_conflict marker=support stance=%+.2f claim=%r",
                p.stance, p.claim[:160],
            )
        elif has_oppose and not has_support and p.stance > 0.1:
            logger.warning(
                "event=claim_stance_sign_conflict marker=oppose stance=%+.2f claim=%r",
                p.stance, p.claim[:160],
            )
    return predictions


# ── winner-entity consistency (retro#401) ───────────────────────────────────
#
# The England–Argentina incident (retro#360) inverted stance because nothing
# checked the extracted evidence against the question's own subject — the
# model's judgement was the only check there was. #313 shipped exactly the
# facets a deterministic check needs (event_actors/event_target/is_occurrence)
# but nothing downstream ever read them. This is that check.
#
# Deliberately narrow. A free-text `question` cannot be parsed into named
# entities without NER, so the patterns below only fire on a small number of
# explicit "X (will) VERB [against/vs/over] Y" shapes — a two-word "vs"
# question with no verb, or a subject named some other way, is left alone
# (no match => no-op). False negatives are free (today's LLM-only behaviour);
# false positives are not, so precision is chosen over recall throughout.

# Sentence-initial words that are capitalized but are never the subject of a
# versus question — without this, "Will England beat Argentina" reads "Will"
# as a one-word entity, since it starts with a capital letter like any name.
_VERSUS_ENTITY_STOPWORDS = frozenset({
    "will", "would", "does", "did", "is", "are", "can", "the", "a", "an",
    "their", "his", "her", "its", "this", "that", "these", "those", "who",
    "what", "when", "after", "before", "and", "or", "so", "if",
})

# A "named entity" here is one-to-four consecutive capitalized words — good
# enough for team/country/candidate names ("England", "Real Madrid", "New
# Zealand"), not a real NER model. An optional leading "the"/"the" article is
# absorbed separately so "the Lakers" still yields entity "Lakers".
_ENTITY = r"[A-Z][A-Za-z0-9&'\-]*(?:\s+[A-Z][A-Za-z0-9&'\-]*){0,3}"
_ARTICLE = r"(?:(?i:the)\s+)?"

# Transitive win verbs take the rival as a direct object — "England beat
# Argentina" — no preposition needed. "win" is deliberately excluded here: it
# is almost always intransitive ("win the match"), so it is handled by the
# prepositional pattern below instead, which requires an explicit versus
# marker and so is far less prone to misfiring on a non-versus sentence.
_WIN_VERB_TRANSITIVE = (
    r"(?:beat|defeat|overcome|upset|edge|down|sink|stun|top|rout|thrash|crush)"
)
_VERSUS_MARKER = r"(?:against|vs\.?|v\.?|over)"

_VERSUS_TRANSITIVE_RE = re.compile(
    rf"{_ARTICLE}(?P<subject>{_ENTITY})\s+(?:will\s+)?(?i:{_WIN_VERB_TRANSITIVE})\w*\s+"
    rf"{_ARTICLE}(?P<rival>{_ENTITY})"
)
_VERSUS_PREPOSITIONAL_RE = re.compile(
    rf"{_ARTICLE}(?P<subject>{_ENTITY})\s+(?:will\s+)?(?i:win)\w*"
    rf"(?:\s+[\w-]+){{0,8}}?\s+(?i:{_VERSUS_MARKER})\s+"
    rf"{_ARTICLE}(?P<rival>{_ENTITY})"
)
_LEADING_INTERROGATIVE_RE = re.compile(r"^\s*(?:will|would)\s+", re.I)


def _match_versus_question(question: str) -> Optional[tuple[str, str]]:
    """Return ``(subject, rival)`` for a two-named-actor versus question, or
    ``None`` when the question does not confidently match that shape.

    Tries the prepositional pattern ("X will win ... against/vs/over Y")
    before the transitive one ("X (will) beat/defeat/... Y") only because the
    former is the shape of the incident that motivated this function; either
    can match a given question and only the first hit is used.
    """
    stripped = _LEADING_INTERROGATIVE_RE.sub("", question)
    for pattern in (_VERSUS_PREPOSITIONAL_RE, _VERSUS_TRANSITIVE_RE):
        for m in pattern.finditer(stripped):
            subject, rival = m.group("subject").strip(), m.group("rival").strip()
            if subject.split()[0].lower() in _VERSUS_ENTITY_STOPWORDS:
                continue
            if rival.split()[0].lower() in _VERSUS_ENTITY_STOPWORDS:
                continue
            if subject.lower() == rival.lower():
                continue
            return subject, rival
    return None


def _mentions_entity(field_text: Optional[str], entity: str) -> bool:
    """Case-insensitive whole-phrase containment: does ``field_text`` name
    ``entity`` (e.g. does event_actors="Argentina's players" mention "Argentina")?
    """
    if not field_text:
        return False
    return re.search(rf"\b{re.escape(entity)}\b", field_text, re.I) is not None


# retro#545 slice (ii), 2026-08-24 precision review: audit_named_entity_dyad_mismatch's
# _mentions_entity check false-positives whenever the same entity is named differently —
# "Donald Trump" vs "Trump administration", "Israel" vs "Israeli government". Anchoring on
# the entity's last (most distinctive) word as a prefix-stem closes both without a curated
# alias table. A small closed exclusion list keeps generic multi-word org names ("X Party",
# "X Coalition") from loose-matching on their common trailing noun, and a length floor keeps
# short acronyms (US/UK/EU/UN) from matching unrelated words that merely start the same way.
_GENERIC_ENTITY_ANCHOR_WORDS = frozenset({
    "party", "administration", "government", "movement", "coalition",
    "authority", "committee", "council", "commission", "forces",
})
_STEM_MATCH_MIN_LEN = 4


def _mentions_entity_stem(field_text: Optional[str], entity: str) -> bool:
    """Looser than ``_mentions_entity``: also matches when ``field_text`` contains a word
    that starts with ``entity``'s last word — "Trump" as a prefix inside "Trump
    administration" (entity "Donald Trump"), "Israel" as a prefix of "Israeli" inside
    "Israeli government". Audit-only: deliberately NOT used by
    ``enforce_winner_entity_consistency``, which stays on the stricter exact-phrase check —
    this only ever turns a "fire" into a "no-op", never the reverse, so it's safe to loosen
    here without touching that guard's enforcing behavior.

    Does not solve metonyms ("the Kremlin" for "Russia"), acronym expansion ("WHO" for
    "World Health Organization"), or a question whose extracted "subject" is a topic/event
    noun rather than an actor (see the known limitation noted on
    ``_extract_named_entities``) — those need real alias data or a different fix, not a
    string primitive.
    """
    if _mentions_entity(field_text, entity):
        return True
    if not field_text:
        return False
    anchor = entity.split()[-1]
    if len(anchor) < _STEM_MATCH_MIN_LEN or anchor.lower() in _GENERIC_ENTITY_ANCHOR_WORDS:
        return False
    return re.search(rf"\b{re.escape(anchor)}\w*\b", field_text, re.I) is not None


def enforce_winner_entity_consistency(
    predictions: list[PredictionExtraction],
    question: str,
) -> list[PredictionExtraction]:
    """Deterministic guard for versus/sports questions: does the dominant
    fact's actor→target dyad actually support the stance sign it carries?

    retro#360 (prod, 2026-07-15): four articles plainly reporting "Argentina
    beat England" were extracted as stance +1.0 for "England will win", and
    the false-unanimous settlement pinned the estimate at 97% — Brier 0.94,
    one of the two worst misses in the resolved corpus. retro#313 shipped
    ``event_actors``/``event_target``/``is_occurrence`` for exactly this
    check, but nothing ever read them (retro#401). This does:

      1. Parse the question for a two-named-actor "X (will) beat/win against Y"
         shape (:func:`_match_versus_question`). No match => no-op — this
         never invents a subject/rival the question doesn't name plainly.
      2. For each claim whose dyad IS the event itself (``is_occurrence`` is
         exactly ``True`` — a precursor cannot tell us who won, and an
         unjudged ``None`` is left to :func:`enforce_precursor_cap` and the
         model's own stance), check whether ``event_actors`` names the RIVAL
         (not the subject) acting on a ``event_target`` naming the SUBJECT
         (not the rival) — "Argentina [actor] beat England [target]" — or the
         mirror image, the SUBJECT acting on the RIVAL.
      3. A rival-beats-subject dyad with a positive stance, or a
         subject-beats-rival dyad with a negative stance, is exactly the
         incident's shape: the dyad and the stance sign disagree about who
         won.

    Conservative by design (retro#401 asks for this explicitly): the dyad
    match tells us the SIGN is wrong with reasonable confidence, but nothing
    here is confident enough in the correct magnitude to assert a flipped
    value, so a caught claim is NEUTRALISED, not inverted — ``stance`` is
    zeroed (no directional vote) and ``settled`` is stripped (mirrors
    :func:`enforce_settlement_event_date`'s "loses settled, keeps voting as
    ordinary evidence" demotion — except here even the ordinary vote is
    silenced, because the one thing this function is sure of is that the
    existing sign is untrustworthy). ``certainty``/``evidence_class``/facets
    are untouched, so the claim still contributes to weight and is still
    fully auditable in ``claims_detail``.

    Every step fails open: a question that doesn't parse, a claim missing
    either facet, an unjudged ``is_occurrence``, or a dyad that names neither
    "the rival acting on the subject" nor "the subject acting on the rival"
    leaves the claim exactly as extracted.
    """
    match = _match_versus_question(question)
    if match is None:
        return predictions
    subject, rival = match

    for p in predictions:
        if p.event_actors is None or p.event_target is None:
            continue
        if p.is_occurrence is not True:
            continue

        actor_is_rival = (
            _mentions_entity(p.event_actors, rival)
            and not _mentions_entity(p.event_actors, subject)
        )
        actor_is_subject = (
            _mentions_entity(p.event_actors, subject)
            and not _mentions_entity(p.event_actors, rival)
        )
        target_is_subject = (
            _mentions_entity(p.event_target, subject)
            and not _mentions_entity(p.event_target, rival)
        )
        target_is_rival = (
            _mentions_entity(p.event_target, rival)
            and not _mentions_entity(p.event_target, subject)
        )

        rival_beats_subject = actor_is_rival and target_is_subject
        subject_beats_rival = actor_is_subject and target_is_rival
        if not (rival_beats_subject or subject_beats_rival):
            continue

        wrong_signed = (
            (rival_beats_subject and p.stance > 0)
            or (subject_beats_rival and p.stance < 0)
        )
        if not wrong_signed:
            continue

        logger.warning(
            "event=winner_entity_sign_conflict subject=%r rival=%r actors=%r "
            "target=%r stance=%+.2f settled=%s claim=%r",
            subject, rival, p.event_actors, p.event_target, p.stance, p.settled,
            p.claim[:120],
        )
        p.settled = False
        p.stance = 0.0

    return predictions


# retro#545 slice (ii): does a strong-stance claim about ONE specific named
# actor land on a fact whose dyad never names them? The wrong-entity examples
# in the issue (a Yoaz Hendel claim scored against an Almog Cohen article, an
# Oren Smadja article, ...) aren't the "X vs Y" shape `_match_versus_question`
# targets — there's no rival, just a single named actor the article isn't
# about. Extracting that actor from free text still isn't real NER, so this
# reuses the same "1-4 capitalized words, minus stopwords" heuristic as the
# versus check above, applied to the whole question instead of a fixed
# grammar. Audit only (retro#545 comment, 2026-08-22): coverage is partial
# (event_actors/event_target populate on ~38% of the strong-stance band,
# prod audit 2026-08-22) and precision on this shape is unmeasured, so this
# logs rather than mutates — the Gate-0/evidence-window (#558) precedent for
# shipping an unvalidated detector shadow-first.
_ENTITY_DYAD_AUDIT_STANCE_GATE = 0.7
_ENTITY_DYAD_AUDIT_CERTAINTY_GATE = 0.7


def _extract_named_entities(question: str) -> list[str]:
    """Named-entity-shaped substrings in ``question``, deduped case-insensitively.

    Same heuristic as ``_ENTITY``/``_VERSUS_ENTITY_STOPWORDS`` above (1-4
    consecutive capitalized words, minus common sentence-initial words) —
    not real NER, just enough to catch a plainly-named person/org/place.

    Known limitation (retro#644): this can't tell an actor-shaped span from a
    topic/event-shaped one — "Ebola" is just as capitalized as "Yoaz Hendel".
    Not fixed here — this function is shared with the *enforcing*
    ``enforce_winner_entity_consistency``, so a span-selection change needs
    its own review, not a bundled fix. ``audit_named_entity_dyad_mismatch``
    instead uses ``_extract_actor_shaped_entities``, an audit-only sibling
    that filters out the reviewed topic-modifier shape.
    """
    seen: set[str] = set()
    entities: list[str] = []
    for m in re.finditer(_ENTITY, question):
        # Consecutive capitalized words match as ONE candidate (e.g. "Will Yoaz
        # Hendel"), so a leading stopword must be stripped from the candidate,
        # not used to reject it outright — otherwise "Will Yoaz Hendel" loses
        # "Yoaz Hendel" entirely instead of just "Will".
        words = m.group(0).split()
        while words and words[0].lower() in _VERSUS_ENTITY_STOPWORDS:
            words = words[1:]
        if not words:
            continue
        candidate = " ".join(words)
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(candidate)
    return entities


# retro#644, 2026-08-24 review: the residual false-positive shape after
# stem-matching (#645) — "Ebola" is grabbed as the subject of "the Ebola
# outbreak", but "Ebola" itself is a topic modifier, not the actor who could
# plausibly show up in event_actors/event_target ("WHO", "Africa CDC" are a
# correctly-extracted, genuinely different dyad). A candidate immediately
# followed by one of these topic-head nouns is a modifier inside a larger
# topic noun phrase, not a standalone named actor, so it's dropped as an
# audit subject. A small closed list, same style as
# ``_GENERIC_ENTITY_ANCHOR_WORDS`` above — not real NER, just enough to catch
# the reviewed shape. Audit-only: deliberately a SEPARATE function from
# ``_extract_named_entities``, which stays untouched and shared with the
# *enforcing* ``enforce_winner_entity_consistency`` guard (see that
# function's docstring for why a span-selection change there needs its own
# review).
_TOPIC_HEAD_NOUNS = frozenset({
    "outbreak", "pandemic", "epidemic", "crisis", "war", "conflict",
    "election", "elections", "referendum", "deal", "agreement", "summit",
    "case", "cases", "death", "deaths", "toll", "campaign", "scandal",
    "controversy", "shutdown", "strike", "protest", "protests", "rally",
    "boycott", "recession", "inflation", "drought", "wildfire", "wildfires",
    "earthquake", "storm", "hurricane", "virus", "disease", "vaccine",
    "treaty", "ceasefire", "dispute", "crackdown", "uprising", "coup",
    "insurgency", "famine", "flood", "floods",
})


def _extract_actor_shaped_entities(question: str) -> list[str]:
    """Like ``_extract_named_entities``, but only ever considers the FIRST
    capitalized candidate — the only one ``audit_named_entity_dyad_mismatch``
    reads (``entities[0]``) — and returns ``[]`` instead of that candidate
    when it's immediately followed by a ``_TOPIC_HEAD_NOUNS`` word
    (retro#644): "Ebola" in "the Ebola outbreak" is a topic modifier, not the
    actor a dyad check should compare against.

    Deliberately does NOT fall through to a later capitalized token when the
    first one is filtered — a second-or-later token was never validated as
    the question's primary subject in the first place (it's exactly as
    likely to be a date, location, or other generic noun as a real actor;
    see ``audit_named_entity_dyad_mismatch``'s docstring on why only
    ``entities[0]`` is trusted). Filtering the first candidate to ``[]``
    keeps the caller's existing "no entities parses out of question => no-op"
    fail-open branch, rather than adding a new one.

    Duplicates ``_extract_named_entities``'s loop rather than filtering its
    output, because the topic-head check needs the candidate's position in
    ``question`` (to inspect the word right after it), which the plain
    string list doesn't carry. The duplication is deliberate: it keeps this
    audit-only path fully separate from the span-selection logic
    ``enforce_winner_entity_consistency`` relies on via
    ``_extract_named_entities``.
    """
    for m in re.finditer(_ENTITY, question):
        words = m.group(0).split()
        while words and words[0].lower() in _VERSUS_ENTITY_STOPWORDS:
            words = words[1:]
        if not words:
            continue
        candidate = " ".join(words)
        tail = question[m.end():].lstrip()
        next_word = tail.split(maxsplit=1)[0].strip(".,?!:;\"'") if tail else ""
        if next_word.lower() in _TOPIC_HEAD_NOUNS:
            return []
        return [candidate]
    return []


def audit_named_entity_dyad_mismatch(
    predictions: list[PredictionExtraction],
    question: str,
) -> list[PredictionExtraction]:
    """Log-only: flag a strong-stance claim whose fact dyad never names the
    question's own primary (first-mentioned) named actor.

    Only the FIRST extracted entity is checked, not "any" of them — a
    question routinely names a location or organisation alongside its real
    subject ("Yoaz Hendel ... in the 26th Knesset"), and those generic nouns
    trivially co-occur in most articles on the same topic, which would mask
    exactly the wrong-entity shape this is meant to catch (an article about a
    different named PERSON, same Knesset). The subject is almost always
    named first in these claims ("X will...", "X runs...", "X is..."), the
    same assumption `_match_versus_question`'s subject/rival ordering makes.

    Fails open throughout, matching every sibling guard: no entity parses out
    of ``question`` => no-op; either dyad field missing on a claim => skip it;
    below the stance/certainty gate => skip; the subject entity appears in
    EITHER ``event_actors`` or ``event_target`` => skip. Never mutates
    ``stance``/``certainty``/``settled``/anything else — pure logging, same
    contract as ``evidence_window_outside`` (PR #558).

    retro#545 slice (ii), 2026-08-24 precision review (244 sampled events, ~0%
    real precision): the comparison uses ``_mentions_entity_stem`` (not the
    stricter ``_mentions_entity``) so an institutional-alias ("Donald Trump" /
    "Trump administration") or adjectival-form ("Israel" / "Israeli
    government") reference no longer false-positives. retro#644 closed the
    third reviewed shape (topic-vs-actor, e.g. "Ebola" in "the Ebola
    outbreak") by extracting the subject with ``_extract_actor_shaped_entities``
    instead of ``_extract_named_entities`` — see that function's docstring.
    Neither fix is exhaustive (curated word lists, not real NER); any further
    shape belongs in a new issue with its own repro.

    Also logs one ``event=entity_dyad_mismatch_shadow`` summary line per call,
    regardless of outcome, with ``eligible``/``fired`` counts — matching the
    ``evidence_window_shadow`` convention (PR #558) — so a future precision
    review has a real trigger-rate denominator instead of only a raw hit count.
    """
    entities = _extract_actor_shaped_entities(question)
    if not entities:
        logger.info(
            "event=entity_dyad_mismatch_shadow question_subject=None eligible=0 fired=0 n=%d",
            len(predictions),
        )
        return predictions
    subject = entities[0]

    eligible = 0
    fired = 0
    for p in predictions:
        if p.event_actors is None or p.event_target is None:
            continue
        if abs(p.stance) < _ENTITY_DYAD_AUDIT_STANCE_GATE:
            continue
        if p.claim_strength < _ENTITY_DYAD_AUDIT_CERTAINTY_GATE:
            continue
        eligible += 1

        if _mentions_entity_stem(p.event_actors, subject) or _mentions_entity_stem(p.event_target, subject):
            continue
        fired += 1

        logger.warning(
            "event=entity_dyad_mismatch question_subject=%r actors=%r target=%r "
            "stance=%+.2f certainty=%.2f settled=%s claim=%r",
            subject, p.event_actors, p.event_target, p.stance, p.claim_strength,
            p.settled, p.claim[:120],
        )

    logger.info(
        "event=entity_dyad_mismatch_shadow question_subject=%r eligible=%d fired=%d n=%d",
        subject, eligible, fired, len(predictions),
    )
    return predictions


# retro#602 sizing (2026-08-23): at |stance|>=0.7 & certainty>=0.7 with a
# populated fact_signal, 44/531 (8.3%) prod rows disagree in sign with their
# own stance. Hand-checked 20: 18 genuine sign-inversions (the tracked Burnham
# and Israel-coalition clusters), 1 borderline, 1 a near-zero fact_signal that
# isn't really a polarity flip — hence the |fact_signal|>=0.3 floor below to
# drop that last case. ~90% per-row precision, unlike audit_named_entity_dyad_
# mismatch's ~0% on the same "promote?" question (not promoted). Log-only for
# now per that same recommendation: this covers every claim (not just
# settled=True, which enforce_settlement_fact_signal_agreement already
# enforces at the stricter 0.5 anchor) and is meant to surface *new* instances
# of the defect class before deciding whether to enforce here too.
_FACT_SIGNAL_SIGN_STANCE_GATE = 0.7
_FACT_SIGNAL_SIGN_CERTAINTY_GATE = 0.7
_FACT_SIGNAL_SIGN_MAGNITUDE_GATE = 0.3


def audit_fact_signal_sign_mismatch(
    predictions: list[PredictionExtraction],
) -> list[PredictionExtraction]:
    """Log-only: flag a strong-stance claim whose fact_signal points the
    opposite way (retro#545/#602).

    Fails open like every sibling guard: a missing or near-zero fact_signal
    (legitimately omitted for opinion/advocacy rows, or a value that bears on
    the event without establishing it) never fires, and below the stance/
    certainty gate is skipped. Never mutates ``stance``/``fact_signal``/
    ``settled``/anything else — pure observability, same contract as
    ``audit_named_entity_dyad_mismatch``.
    """
    for p in predictions:
        if p.fact_signal is None or abs(p.fact_signal) < _FACT_SIGNAL_SIGN_MAGNITUDE_GATE:
            continue
        if abs(p.stance) < _FACT_SIGNAL_SIGN_STANCE_GATE:
            continue
        if p.claim_strength < _FACT_SIGNAL_SIGN_CERTAINTY_GATE:
            continue
        if (p.fact_signal > 0) == (p.stance > 0):
            continue

        logger.warning(
            "event=fact_signal_sign_mismatch stance=%+.2f fact_signal=%+.2f "
            "certainty=%.2f settled=%s claim=%r",
            p.stance, p.fact_signal, p.claim_strength, p.settled, p.claim[:120],
        )

    return predictions


# retro#326 sizing (2026-08-25): a prod sweep of author_lean rows added since
# the PR#314 sentiment-leak fix deployed (2026-07-24 11:34 UTC) found ~26-30
# rows (of 1467, ~2%) where author_lean's sign disagrees with the article's
# own claim-weighted stance at a real (non-near-zero) stance magnitude — not
# the single narrow "Behrendt-class" residual PR#314 tracked, but a broader,
# ongoing recurrence: strong evaluative/alarmed language (outrage, "strategic
# disaster") sitting next to a clearly affirmed factual claim, e.g. a byline
# explicitly declaring Israel will NOT withdraw from Lebanon still scored
# author_lean=-0.9 against stance=+1.0. Spot-checked 3/3 high-stance hits as
# genuine leaks (Bermant/FP, Behrendt/tagesschau, a hnaftali.com piece) —
# same gate shape and thresholds as `audit_fact_signal_sign_mismatch`
# (retro#602) since that guard's 0.7/0.7/0.3 bar is the only precision-sized
# precedent in this codebase; a broader hand-check to size a looser bound (the
# way #602 did with a 20-row sample) is left as follow-up. Log-only, same
# contract as every sibling audit in this module: never mutates author_lean,
# author_lean_certainty, or stance.
_AUTHOR_LEAN_SIGN_STANCE_GATE = 0.7
_AUTHOR_LEAN_SIGN_CERTAINTY_GATE = 0.7
_AUTHOR_LEAN_SIGN_MAGNITUDE_GATE = 0.3


def audit_author_lean_sign_mismatch(
    author_lean: Optional[float],
    author_lean_certainty: Optional[float],
    avg_stance: float,
    avg_certainty: float,
    *,
    url: Optional[str] = None,
) -> None:
    """Log-only: flag an author_lean whose sign disagrees with the article's
    own claim-weighted stance (retro#326).

    ``author_lean`` is the BYLINE author's own directional forecast — per the
    AUTHOR_LEAN prompt section it must track whether the author expects the
    event to happen, never their sentiment about it. When the article's own
    extracted claims (``avg_stance``, the same claim-weighted aggregate used
    for the live estimate) affirm the event at real magnitude and confidence
    but author_lean reads the opposite sign, that is very likely a sentiment-
    vs-forecast leak, not a genuine author/fact disagreement — a genuine
    disagreement would show up as a *negative* avg_stance too (the author's
    own claims would reflect their doubt), which this gate leaves alone.

    Fails open like every sibling guard: a null author_lean (no position
    taken — most reporting) never fires, and below the stance/certainty gate
    or the author_lean magnitude floor is skipped. Never mutates anything —
    pure observability, same contract as ``audit_fact_signal_sign_mismatch``.
    """
    if author_lean is None or abs(author_lean) < _AUTHOR_LEAN_SIGN_MAGNITUDE_GATE:
        return
    if abs(avg_stance) < _AUTHOR_LEAN_SIGN_STANCE_GATE:
        return
    if avg_certainty < _AUTHOR_LEAN_SIGN_CERTAINTY_GATE:
        return
    if (author_lean > 0) == (avg_stance > 0):
        return

    logger.warning(
        "event=author_lean_sign_mismatch author_lean=%+.2f author_lean_certainty=%s "
        "avg_stance=%+.2f avg_certainty=%.2f url=%r",
        author_lean,
        f"{author_lean_certainty:.2f}" if author_lean_certainty is not None else None,
        avg_stance, avg_certainty, url,
    )


# A settlement whose own fact lane points the other way is contradicting itself.
# 0.5 is the anchoring bar the FACT_SIGNAL prompt already uses for a graded
# reading: below it a value is "bears on the event" rather than "establishes it",
# and `enforce_precursor_cap` has already clamped precursor rows to ±0.3, so a
# fact_signal that survives at |0.5| or more is an announcement/denial-grade
# reading of the event itself — the only kind that can contradict a settlement.
_SETTLEMENT_FACT_SIGNAL_ANCHOR = 0.5


def enforce_settlement_fact_signal_agreement(
    predictions: list[PredictionExtraction],
) -> list[PredictionExtraction]:
    """A settlement vote may not contradict its own fact lane.

    retro#545's sign-error class: the extractor commits to a strong settled
    stance whose direction is the opposite of what the article reports. The
    flagship case is live and public — 41 pool rows on the ACTIVE "Andy Burnham
    will REMAIN Prime Minister until 2028" forecast, every one of them
    ``stance=-1.00 settled`` off articles reporting that he *took office*, which
    is evidence for the claim, not the foreclosure of it. Settle-pinned rows
    clamp the published probability to floor/ceiling rather than nudging a
    weighted mean, so each of those 41 is a wrong number on a public page.

    The check costs no LLM call because the model already tells us it disagrees
    with itself. ``fact_signal`` is the fact-lane counterpart of ``stance`` on
    the *same* axis — "+1 the facts establish the event happened or is
    happening, -1 the facts establish it will not or cannot" — so for a
    ``settled`` claim, which asserts an accomplished fact rather than a reading
    of one, the two must carry the same sign. Opposite signs mean one of the two
    is mis-signed, and nothing here can tell which.

    So a caught claim is NEUTRALISED, not inverted, exactly as in
    :func:`enforce_winner_entity_consistency`: ``settled`` is stripped and
    ``stance`` zeroed, because the single thing this function is sure of is that
    the existing sign is untrustworthy. ``certainty``, ``fact_signal``,
    ``evidence_class`` and the facets are untouched, so the row keeps its weight
    and stays fully auditable in ``claims_detail``.

    Prod audit (2026-08-19, head rows, COMPLETE): 230 settled rows carry a
    ``fact_signal``; **46 of them oppose their own stance** at |fact_signal| >=
    0.5, across just **3 ACTIVE forecasts** — 41 Burnham, 3 "no Arab ministers",
    2 "Netanyahu will be PM on 31 Dec". Every one of the 46 already sits at
    |stance| >= 0.7, so no separate strong-stance gate is needed; the population
    is strong by construction. 45 of the 46 were written in the last 30 days,
    and 96% of settled rows written in that window carry a ``fact_signal``, so
    coverage on current traffic is effectively complete even though the field
    exists on only 58% of settled rows historically (it is a shadow field that
    was never backfilled — the ``verified``-marker precedent, forward-only).

    Fail-open in the same asymmetry as its siblings: a claim that is not
    ``settled``, has no ``fact_signal`` (legitimately omitted for opinion and
    advocacy rows), carries one below the anchor, or has a zero stance is left
    exactly as the model returned it. Runs last, after
    :func:`enforce_settlement_event_date` has demoted undated settlements and
    :func:`enforce_precursor_cap` has clamped precursor fact_signals below the
    anchor — both of which correctly keep their rows out of this net.
    """
    for p in predictions:
        if p.settled is not True:
            continue
        if p.fact_signal is None or abs(p.fact_signal) < _SETTLEMENT_FACT_SIGNAL_ANCHOR:
            continue
        if p.stance == 0.0:
            continue
        if (p.fact_signal > 0) == (p.stance > 0):
            continue

        logger.warning(
            "event=settlement_fact_signal_conflict stance=%+.2f fact_signal=%+.2f "
            "certainty=%.2f facet=%s verified=%s claim=%r",
            p.stance, p.fact_signal, p.claim_strength, p.facet, p.verified, p.claim[:120],
        )
        p.settled = False
        p.stance = 0.0

    return predictions


_QUOTE_PROVENANCE_MIN_LEN = 20


def _normalize_for_provenance_compare(text: str) -> str:
    """Casefold, collapse whitespace, strip enclosing punctuation/quote marks —
    enough to catch a ``quote`` that is ``event_name``/``event_description``
    restated with only formatting differences, without touching real content."""
    collapsed = re.sub(r"\s+", " ", text.strip().casefold())
    return collapsed.strip(" .,;:!?\"'“”‘’")


def audit_quote_provenance_mismatch(
    predictions: list[PredictionExtraction],
    event_name: str,
    event_description: str,
) -> list[PredictionExtraction]:
    """Log-only: flag a claim whose ``quote`` is actually the event's own
    ``event_name``/``event_description`` restated, not real article text.

    retro#545's 2026-08-25 cross-model survey (700 matched-quote comparisons,
    78 model-pairs x 50 real prod articles) found 2 of 10 flagged articles
    carried a fabricated quote of exactly this shape: the article had nothing
    to do with its assigned event, and the model's ``quote`` field was
    verbatim the event_name/event_description rather than anything from the
    article body. No existing guard checks quote provenance at all —
    ``extract_predictions``'s prompt tells the model to quote verbatim, but
    nothing verifies it did.

    Deliberately narrow: compares ``quote`` against ``event_name``/
    ``event_description`` only, not a quote-vs-article_text substring check.
    The two known real examples were exact restatements of the event, which
    this catches precisely; a full article-body check would need real
    normalization work (translation, HTML artifacts, elided quotes) to avoid
    drowning in noise — deferred until this narrower signal's precision is
    measured, the same incremental-scoping approach
    ``audit_named_entity_dyad_mismatch`` took.

    Fails open like every sibling: a ``quote`` shorter than
    ``_QUOTE_PROVENANCE_MIN_LEN`` is skipped — below that length, an on-topic
    quote could coincidentally overlap the event text without being
    fabricated. Never mutates ``stance``/``claim``/anything else — pure
    logging, same contract as ``audit_named_entity_dyad_mismatch``.

    Also logs one ``event=quote_provenance_mismatch_shadow`` summary line per
    call with ``eligible=``/``fired=`` counts, matching the
    ``entity_dyad_mismatch_shadow`` convention, so a future precision review
    has a real trigger-rate denominator.
    """
    targets = {
        _normalize_for_provenance_compare(t)
        for t in (event_name, event_description)
        if t
    }
    targets.discard("")

    eligible = 0
    fired = 0
    for p in predictions:
        quote = (p.quote or "").strip()
        if len(quote) < _QUOTE_PROVENANCE_MIN_LEN:
            continue
        eligible += 1

        if _normalize_for_provenance_compare(quote) not in targets:
            continue
        fired += 1

        logger.warning(
            "event=quote_provenance_mismatch event_name=%r event_description=%r "
            "quote=%r stance=%+.2f certainty=%.2f settled=%s claim=%r",
            event_name, event_description, quote, p.stance, p.claim_strength,
            p.settled, p.claim[:120],
        )

    logger.info(
        "event=quote_provenance_mismatch_shadow eligible=%d fired=%d n=%d",
        eligible, fired, len(predictions),
    )
    return predictions
