"""Basic model validation tests — no LLM calls."""

import pytest
from pydantic import ValidationError

from tm.models import (
    ClaimActor,
    GatekeeperOutput,
    PredictionExtraction,
    ExtractionOutput,
    PredictionType,
    MatrixState,
    CellStatus,
    Grounds,
    Quantity,
    ReaderConfidence,
    Voice,
)


def test_gatekeeper_output():
    out = GatekeeperOutput(is_prediction=True, reason="Contains forecast", prediction_count_estimate=2)
    assert out.is_prediction is True
    assert out.prediction_count_estimate == 2


def test_gatekeeper_output_unwraps_properties_envelope():
    """Nova Lite (MD_JSON mode) intermittently wraps output as {"properties": {...}}
    instead of the flat shape — retro#306."""
    out = GatekeeperOutput.model_validate({
        "properties": {"is_prediction": True, "reason": "Contains forecast", "prediction_count_estimate": 2}
    })
    assert out.is_prediction is True
    assert out.prediction_count_estimate == 2


def test_prediction_extraction_clamps():
    pred = PredictionExtraction(
        quote="test",
        claim="test claim",
        stance=0.8,
        sentiment=0.5,
        certainty=0.9,
        specificity=0.7,
        hedge_ratio=0.1,
        conditionality=0.0,
        magnitude=0.6,
        time_horizon="months",
        time_horizon_days=90,
        prediction_type=PredictionType.binary,
        source_authority=0.8,
    )
    assert pred.stance == 0.8
    assert pred.time_horizon_days == 90


def test_prediction_extraction_quantitative_estimate_defaults_to_none():
    pred = PredictionExtraction(quote="q", claim="c", stance=0.0, certainty=0.5)
    assert pred.quantitative_estimate is None


def test_prediction_extraction_quantitative_estimate_accepts_valid_probability():
    pred = PredictionExtraction(
        quote="q", claim="c", stance=-0.62, certainty=0.85, quantitative_estimate=0.1883,
    )
    assert pred.quantitative_estimate == 0.1883


def test_prediction_extraction_quantitative_estimate_rejects_out_of_range():
    with pytest.raises(ValidationError):
        PredictionExtraction(quote="q", claim="c", stance=0.0, certainty=0.5, quantitative_estimate=1.5)


def test_prediction_extraction_fact_signal_facets_default_to_none():
    pred = PredictionExtraction(quote="q", claim="c", stance=0.0, certainty=0.5)
    assert pred.fact_signal is None
    assert pred.event_actors is None
    assert pred.event_target is None
    assert pred.is_occurrence is None
    assert pred.verified is None


def test_prediction_extraction_fact_signal_facets_accept_valid_values():
    pred = PredictionExtraction(
        quote="q", claim="c", stance=0.8, certainty=0.9,
        fact_signal=0.3, event_actors="United States", event_target="Iran",
        is_occurrence=False, verified=True,
    )
    assert pred.fact_signal == 0.3
    assert pred.event_actors == "United States"
    assert pred.event_target == "Iran"
    assert pred.is_occurrence is False
    assert pred.verified is True


def test_prediction_extraction_fact_signal_rejects_out_of_range():
    with pytest.raises(ValidationError):
        PredictionExtraction(quote="q", claim="c", stance=0.0, certainty=0.5, fact_signal=1.5)
    with pytest.raises(ValidationError):
        PredictionExtraction(quote="q", claim="c", stance=0.0, certainty=0.5, fact_signal=-1.5)


def test_prediction_extraction_unwraps_properties_envelope():
    pred = PredictionExtraction.model_validate({
        "properties": {"quote": "q", "claim": "c", "stance": 0.4, "certainty": 0.6}
    })
    assert pred.quote == "q"
    assert pred.stance == 0.4


def test_extraction_output_unwraps_properties_envelope():
    out = ExtractionOutput.model_validate({
        "properties": {"predictions": [], "author_lean": 0.3, "author_lean_certainty": 0.7}
    })
    assert out.author_lean == 0.3
    assert out.author_lean_certainty == 0.7


def test_extraction_output_author_lean_defaults_to_none():
    out = ExtractionOutput(predictions=[])
    assert out.author_lean is None
    assert out.author_lean_certainty is None


def test_extraction_output_author_lean_accepts_valid_values():
    out = ExtractionOutput(predictions=[], author_lean=-0.6, author_lean_certainty=0.5)
    assert out.author_lean == -0.6
    assert out.author_lean_certainty == 0.5


def test_extraction_output_author_lean_rejects_out_of_range():
    with pytest.raises(ValidationError):
        ExtractionOutput(predictions=[], author_lean=1.5)
    with pytest.raises(ValidationError):
        ExtractionOutput(predictions=[], author_lean_certainty=-0.1)


def test_matrix_state_tracking():
    state = MatrixState()

    # Default is pending
    cell = state.get("A01", "ynet")
    assert cell.status == CellStatus.pending

    # Update status
    state.set_status("A01", "ynet", CellStatus.done, prediction_count=3)
    assert state.get("A01", "ynet").status == CellStatus.done
    assert state.get("A01", "ynet").prediction_count == 3

    # Stats
    stats = state.stats()
    assert stats["done"] == 1


def test_matrix_state_key():
    state = MatrixState()
    assert state.key("B01", "haaretz") == "B01:haaretz"


class TestClaimStrengthCertaintyAlias:
    """`certainty` was renamed to `claim_strength` in Oracle 1.5 Phase 1 (retro#680).

    The old name stays live as a WIRE alias for one schema cycle: inbound it is
    accepted, outbound it is still emitted. That keeps every already-persisted
    atlas row and every reader that indexes the literal key ``certainty``
    (`tm.utils.split_scored_predictions`, `tm.scorer`, `tm.backtest`,
    `tm.render_atlas`) working unchanged across the deploy, rather than having
    them silently reclassify new rows as malformed.

    These tests are the contract. When the alias is dropped next cycle, the ones
    asserting the old name are what should fail first.
    """

    def test_accepts_the_old_name_inbound(self):
        pred = PredictionExtraction(quote="q", claim="c", stance=0.1, certainty=0.65)
        assert pred.claim_strength == 0.65

    def test_accepts_the_new_name_inbound(self):
        pred = PredictionExtraction(quote="q", claim="c", stance=0.1, claim_strength=0.65)
        assert pred.claim_strength == 0.65

    def test_emits_both_names_with_the_same_value(self):
        dumped = PredictionExtraction(
            quote="q", claim="c", stance=0.1, claim_strength=0.65
        ).model_dump()
        assert dumped["claim_strength"] == 0.65
        assert dumped["certainty"] == 0.65

    def test_the_old_name_survives_a_dump_validate_round_trip(self):
        pred = PredictionExtraction(quote="q", claim="c", stance=0.1, certainty=0.65)
        assert PredictionExtraction.model_validate(pred.model_dump()).claim_strength == 0.65

    def test_the_alias_is_not_offered_to_the_llm(self):
        """The alias must live on the wire only, never in the elicitation schema.

        A `@computed_field` would also emit both names — but it would additionally
        publish `certainty` into the JSON schema instructor sends to the model,
        re-teaching the model the name this rename exists to retire, and inviting
        it to fill in two fields that must never disagree. `@model_serializer`
        keeps the alias strictly outbound; this test is what tells the two apart.
        """
        props = PredictionExtraction.model_json_schema()["properties"]
        assert "claim_strength" in props
        assert "certainty" not in props


class TestReaderConfidence:
    """`reader_confidence` {level, trap} — the READER's confidence in its own
    reading, split from the SOURCE's `claim_strength` (Oracle 1.5 Phase 1,
    retro#681).

    Shadow: populated and persisted, read by nothing. What these tests pin is
    that it can be populated *honestly* — an omitted field, a level with no
    trap, and a value the model garbled are three different states, and the
    field is only worth harvesting if the three stay distinguishable.
    """

    @staticmethod
    def _pred(**over):
        return PredictionExtraction(**{
            "quote": "q", "claim": "c", "stance": 0.1, "claim_strength": 0.65, **over
        })

    def test_a_level_and_a_trap_both_land(self):
        pred = self._pred(reader_confidence={"level": "low", "trap": "numeric_comparison"})
        assert pred.reader_confidence.level == "low"
        assert pred.reader_confidence.trap == "numeric_comparison"

    def test_a_level_with_no_trap_is_the_ordinary_case(self):
        """Most spans trip no trap, and the prompt asks for the field anyway.
        A required trap would force the model to invent one."""
        pred = self._pred(reader_confidence={"level": "high"})
        assert pred.reader_confidence.level == "high"
        assert pred.reader_confidence.trap is None

    def test_an_absent_field_stays_absent(self):
        """Every row extracted before v5 has no reader_confidence, and so does
        every row where the model simply didn't answer. Both must parse, and
        both must read as None rather than as some default level."""
        assert self._pred().reader_confidence is None

    def test_the_model_itself_is_strict(self):
        """A half-answer is not data. `level` is what Phase 4 gates on, and the
        trap names are the contract with the detectors that score them
        (retro#657 negation, the PR#671 numeric cases, stance_tone_conflation) —
        a free-text trap could not be checked against anything."""
        with pytest.raises(ValidationError):
            ReaderConfidence.model_validate({"trap": "negation"})
        with pytest.raises(ValidationError):
            ReaderConfidence.model_validate({"level": "high", "trap": "vibes"})
        with pytest.raises(ValidationError):
            ReaderConfidence.model_validate({"level": "very high"})

    @pytest.mark.parametrize("bad", [
        {"trap": "negation"},                    # no level
        {"level": "high", "trap": "vibes"},      # trap outside the enum
        {"level": "very high"},                  # level outside the enum
        {"level": ["high"]},                     # wrong type entirely
    ])
    def test_a_malformed_answer_is_dropped_and_the_prediction_survives(self, bad, caplog):
        """Strict model, tolerant boundary. `complete_structured` runs instructor
        with max_retries=1, so a still-malformed value would raise out of
        ExtractionOutput and drop the whole ARTICLE from the forecast. Paying a
        real article for a field nothing reads is the wrong trade."""
        import logging

        with caplog.at_level(logging.WARNING, logger="tm.models"):
            pred = self._pred(reader_confidence=bad)
        assert pred.reader_confidence is None
        assert pred.claim_strength == 0.65, "the rest of the prediction must be untouched"
        assert "event=reader_confidence_malformed" in caplog.text, (
            "a silent drop would make 'answered badly' look like 'did not answer', "
            "which is the one distinction the fill rate exists to draw"
        )

    def test_a_double_serialized_object_is_parsed(self):
        """Some models in TOOLS mode emit a nested object as a JSON *string*
        (`ExtractionOutput._deserialize_string_predictions` already handles the
        same thing one level up). reader_confidence is the first nested field
        inside a prediction, so it needs the same unwrapping — otherwise the
        field reads as 0% filled on exactly the models most likely to be
        studied for fill rate."""
        pred = PredictionExtraction.model_validate({
            "quote": "q", "claim": "c", "stance": 0.1, "claim_strength": 0.65,
            "reader_confidence": '{"level": "medium", "trap": "tone_vs_content"}',
        })
        assert pred.reader_confidence == ReaderConfidence(level="medium", trap="tone_vs_content")

    def test_an_unparseable_string_is_dropped_loudly(self, caplog):
        """The coercion above must not become a silent swallow-everything. A
        model that answers in prose and a model that does not answer are
        different findings, and the fill rate exists to tell them apart — so
        the drop is logged, exactly like any other malformed value."""
        import logging

        with caplog.at_level(logging.WARNING, logger="tm.models"):
            pred = PredictionExtraction.model_validate({
                "quote": "q", "claim": "c", "stance": 0.1, "claim_strength": 0.65,
                "reader_confidence": "pretty confident",
            })
        assert pred.reader_confidence is None
        assert "event=reader_confidence_malformed" in caplog.text

    def test_it_survives_a_dump_validate_round_trip(self):
        """daatan persists this inside claims_detail verbatim (daatan#1235)."""
        pred = self._pred(reader_confidence={"level": "low", "trap": "conflicting_signals"})
        assert PredictionExtraction.model_validate(
            pred.model_dump()
        ).reader_confidence == pred.reader_confidence

    def test_the_llm_schema_offers_the_object_and_every_trap_name(self):
        """The prompt names six traps; the schema must accept exactly those six.
        A drift either way means the model is told to answer something it cannot
        express, or can express something no detector scores."""
        schema = PredictionExtraction.model_json_schema()
        assert "reader_confidence" in schema["properties"]
        trap = schema["$defs"]["ReaderConfidence"]["properties"]["trap"]
        names = {v for branch in trap["anyOf"] for v in branch.get("enum", [])}
        assert names == {
            "negation", "numeric_comparison", "entity_or_event_mismatch",
            "tone_vs_content", "inference_needed", "conflicting_signals",
        }

    def test_no_llm_facing_model_pays_for_a_docstring(self):
        """Pydantic copies a model's docstring into its JSON schema `description`,
        and that schema IS the tool definition sent on every extraction call — so a
        docstring here is not documentation, it is prompt text billed per request.
        Every model reachable from `ExtractionOutput` keeps its rationale in `#`
        comments for this reason — the assertion below enumerates them from the
        schema itself rather than from a list someone has to remember to extend.

        Pinned rather than left to review: the first draft of that class used a
        docstring and shipped 1,283 characters of retro#664 rationale to the model
        on every call, 18% of the whole schema and larger than any field in it.
        """
        schema = ExtractionOutput.model_json_schema()
        # Walk the whole reachable schema rather than a hand-kept tuple. The tuple version
        # of this test passed while `ClaimActor` (retro#697) shipped a 378-char docstring,
        # because a new nested model is exactly the thing nobody remembers to add to a
        # list. `$defs` holds every model the extraction schema reaches, so a future field
        # cannot arrive outside the assertion's scope.
        offenders = {
            name: d["description"]
            for name, d in {"ExtractionOutput": schema, **schema.get("$defs", {})}.items()
            if d.get("description")
        }
        assert not offenders, (
            "These LLM-facing models have docstrings, sent to the model on every call as "
            "schema description: "
            + ", ".join(f"{n} ({len(v)} chars)" for n, v in sorted(offenders.items()))
            + ". Move each to a `#` comment above the class."
        )

    def test_it_is_independent_of_claim_strength(self):
        """The whole point of retro#680's split. retro#664's Kenya case is a
        flat, categorical span — maximum source commitment — that Nova Lite
        read badly; the two numbers have to be able to move opposite ways or
        the reader's confusion goes on being billed to the source."""
        pred = self._pred(
            claim_strength=1.0, reader_confidence={"level": "low", "trap": "numeric_comparison"}
        )
        assert pred.claim_strength == 1.0
        assert pred.reader_confidence.level == "low"


class TestReportKindAndConsensusView:
    """`report_kind` (per claim) and `consensus_view` (per article) — the two
    fields unparked from retro#673 in retro#686 (Oracle 1.5 Phase 1).

    Shadow: populated, read by nothing. `report_kind` is persisted for free
    (daatan's `claims_detail` is a JSON column); `consensus_view` is per-source
    and has no column yet, so for now it lives only on the `/forecast` response
    and in the A/B harness's article-level fill. Both are flat enums
    rather than scalars on purpose — #673's own caveat is that a graded field
    is a fresh site for the #394 pathology, where a scalar collapses onto its
    band labels. What these tests pin is the same thing #681's do: that an
    omitted value, a filled value and a garbled one stay three distinguishable
    states, because the fill rate is the entire product of a shadow field.
    """

    @staticmethod
    def _pred(**over):
        return PredictionExtraction(**{
            "quote": "q", "claim": "c", "stance": 0.1, "claim_strength": 0.65, **over
        })

    # --- report_kind ---

    def test_both_members_land(self):
        assert self._pred(report_kind="level").report_kind == "level"
        assert self._pred(report_kind="change").report_kind == "change"

    def test_an_absent_field_stays_absent(self):
        """Every row extracted before v8 has no report_kind, and so does every
        row where the model judged that neither member fits — the prompt asks
        it to omit rather than guess. Both must read as None, not as a default
        member: a defaulted 'level' would be indistinguishable from a real one
        and would quietly inflate exactly the statistic this field is for."""
        assert self._pred().report_kind is None

    def test_it_is_independent_of_stance(self):
        """The distinction the field exists to draw. 'unemployment is at 7%'
        and 'unemployment rose to 7%' carry the same stance toward a
        'above 6%?' question and are not the same evidence — a level restates
        a standing situation a prior article may already have supplied, a
        change is new movement. Nothing about the stance sign predicts which."""
        rising = self._pred(stance=0.8, report_kind="change")
        standing = self._pred(stance=0.8, report_kind="level")
        assert rising.stance == standing.stance == 0.8
        assert rising.report_kind != standing.report_kind

    @pytest.mark.parametrize("bad", ["levels", "Level", "change_level", "", 1, ["level"]])
    def test_a_garbled_value_is_dropped_and_the_prediction_survives(self, bad, caplog):
        """Strict field, tolerant boundary — the retro#681 trade, applied to a
        flat enum. `complete_structured` runs instructor with max_retries=1, so
        a model that keeps answering 'levels' would raise out of
        ExtractionOutput and drop the whole ARTICLE from the forecast. Paying a
        real article for a field nothing reads is the wrong trade, and a brand
        new enum name is precisely where a model gets it wrong.

        Note `Level` is in this list: the drop is case-SENSITIVE. A silent
        .lower() would be a second, undocumented way to answer, and the fill
        rate could no longer say whether the model returned the name it was
        given.
        """
        import logging

        with caplog.at_level(logging.WARNING, logger="tm.models"):
            pred = self._pred(report_kind=bad)
        assert pred.report_kind is None
        assert pred.claim_strength == 0.65, "the rest of the prediction must be untouched"
        assert "event=report_kind_malformed" in caplog.text, (
            "a silent drop would make 'answered badly' look like 'did not answer', "
            "which is the one distinction the fill rate exists to draw"
        )

    def test_it_survives_a_dump_validate_round_trip(self):
        """daatan persists this inside claims_detail verbatim (daatan#1235)."""
        pred = self._pred(report_kind="change")
        assert PredictionExtraction.model_validate(pred.model_dump()).report_kind == "change"

    def test_the_llm_schema_offers_exactly_two_members(self):
        """The prompt names two; the schema must accept exactly those two. A
        third member added here without the prose block — or vice versa — is
        how a field starts being asked for something it cannot express."""
        prop = PredictionExtraction.model_json_schema()["properties"]["report_kind"]
        names = {v for branch in prop["anyOf"] for v in branch.get("enum", [])}
        assert names == {"level", "change"}

    def test_it_is_appended_after_every_pre_existing_field(self):
        """retro#680 measured that perturbing the MIDDLE of this schema costs
        Nova Lite the whole fact_signal block (fill 42% -> 25%). Both new
        fields go at the tail so every existing field keeps its neighbours and
        its position in the text the model reads."""
        props = list(PredictionExtraction.model_json_schema()["properties"])
        assert props.index("reader_confidence") < props.index("report_kind")
        # Every later field group took the tail after these two, by the same rule:
        # `quantity` (retro#683), then `tone` + `voice` (retro#684). Pinned as the
        # whole ordered tail rather than one index each — the rule is about the
        # ORDER of the block, and an index-per-field assertion goes on passing while
        # a new field is quietly inserted between two of them.
        assert props[props.index("reader_confidence"):] == [
            "reader_confidence", "report_kind", "quantity", "tone", "voice", "grounds",
        ]

    # --- consensus_view ---

    def test_every_member_lands(self):
        for v in ("expects_yes", "expects_no", "divided"):
            assert ExtractionOutput(predictions=[], consensus_view=v).consensus_view == v

    def test_an_absent_consensus_view_stays_absent(self):
        """Most articles never say what anyone else expects. Null is the
        ordinary answer here, not a failure — a default would make the field
        look filled on exactly the articles that carry no consensus at all."""
        assert ExtractionOutput(predictions=[]).consensus_view is None

    def test_it_is_independent_of_author_lean(self):
        """The field's entire reason for existing, and its kill criterion. An
        article whose byline expects the event while reporting that everyone
        else does not is the interesting case — if consensus_view could only
        agree with author_lean it would be a copy of a field we already have."""
        out = ExtractionOutput(predictions=[], author_lean=0.8, consensus_view="expects_no")
        assert out.author_lean == 0.8
        assert out.consensus_view == "expects_no"

    @pytest.mark.parametrize("bad", ["expects yes", "yes", "EXPECTS_YES", "split", 0, {"v": "divided"}])
    def test_a_garbled_consensus_view_is_dropped_and_the_article_survives(self, bad, caplog):
        """Same trade as report_kind, one level up — and it matters more here,
        because this value rides on the ExtractionOutput itself: a raise takes
        every prediction in the article with it, not one field of one claim."""
        import logging

        with caplog.at_level(logging.WARNING, logger="tm.models"):
            out = ExtractionOutput.model_validate({
                "predictions": [], "author_lean": 0.4, "consensus_view": bad,
            })
        assert out.consensus_view is None
        assert out.author_lean == 0.4, "the rest of the extraction must be untouched"
        assert "event=consensus_view_malformed" in caplog.text

    def test_the_drop_survives_the_properties_envelope(self, caplog):
        """Nova Lite's spurious {"properties": {...}} wrapper (retro#306) is
        unwrapped before field validation. The guard has to run after that
        unwrap, or the one model most likely to garble a new enum is also the
        one model the guard never protects."""
        import logging

        with caplog.at_level(logging.WARNING, logger="tm.models"):
            out = ExtractionOutput.model_validate({
                "properties": {"predictions": [], "consensus_view": "unsure"}
            })
        assert out.consensus_view is None
        assert "event=consensus_view_malformed" in caplog.text

    def test_consensus_view_survives_a_dump_validate_round_trip(self):
        out = ExtractionOutput(predictions=[], consensus_view="divided")
        assert ExtractionOutput.model_validate(out.model_dump()).consensus_view == "divided"

    def test_the_llm_schema_offers_exactly_three_members(self):
        prop = ExtractionOutput.model_json_schema()["properties"]["consensus_view"]
        names = {v for branch in prop["anyOf"] for v in branch.get("enum", [])}
        assert names == {"expects_yes", "expects_no", "divided"}

    def test_the_article_level_tail_is_in_order(self):
        """Article-level and next to author_lean because it is the same shape of
        question — what the AUTHOR thinks vs what the author says EVERYONE ELSE
        thinks — and at the tail for the retro#680 reason above. retro#697's
        three follow it under the same rule.

        The whole ordered tail, not `props[-1] == "consensus_view"` plus two
        index comparisons, which is what this was. That form pins only whichever
        field happens to have landed last: it fails the moment a new field is
        appended CORRECTLY, and says nothing about the order of the ones before
        it. This version fails when something is inserted into the middle, which
        is the thing retro#680 actually costs us.
        """
        props = list(ExtractionOutput.model_json_schema()["properties"])
        assert props[props.index("author_lean"):] == [
            "author_lean", "author_lean_certainty", "consensus_view",
            "claim_actor", "claim_predicate", "claim_scope",
        ]

    def test_neither_field_pays_for_a_docstring(self):
        """Pydantic copies a model docstring into the JSON schema description,
        and that schema IS the tool definition sent on every call (retro#700).
        Re-pinned here because both new fields landed on models that already
        carry their rationale in `#` comments for exactly this reason."""
        for model in (ExtractionOutput, PredictionExtraction):
            assert model.model_json_schema().get("description", "") == ""


class TestClaimDecomposition:
    """`claim_actor` {name, type}, `claim_predicate`, `claim_scope` — the WHO /
    WHAT / WITHIN WHAT SCOPE of the RELATED EVENT (Oracle 1.5 Phase 1, retro#697).

    `PROMPT_PREFIX` § MATCH THE EVENT has required this decomposition since v1 and
    given the model nowhere to put it, so the reasoning was demanded, discarded and
    unverifiable. These pin the shape of the answer, not its quality — the A/B
    measures whether the model answers, and `settlement_semantic` measures whether
    the answer is worth anything.
    """

    def test_all_three_default_to_none(self):
        """The prompt asks for all three on every call, so None means "did not
        answer" and must never be a default masquerading as a value."""
        out = ExtractionOutput(predictions=[])
        assert out.claim_actor is None
        assert out.claim_predicate is None
        assert out.claim_scope is None

    def test_it_accepts_the_decomposition(self):
        out = ExtractionOutput.model_validate({
            "predictions": [],
            "claim_actor": {"name": "Party Y", "type": "party"},
            "claim_predicate": "withdraws from the parliamentary race",
            "claim_scope": "at least one party, before the election",
        })
        assert out.claim_actor.name == "Party Y"
        assert out.claim_actor.type == "party"
        assert out.claim_predicate == "withdraws from the parliamentary race"
        assert out.claim_scope == "at least one party, before the election"

    def test_the_actor_type_enum_is_closed(self):
        """Closed because MATCH THE EVENT's first adjacency rule is stated in terms
        of TYPE — a member when the event is about the party — so a free-text type
        would leave that comparison uncodable, which is the reason the field is
        elicited at all rather than left as internal reasoning."""
        defs = ExtractionOutput.model_json_schema()["$defs"]["ClaimActor"]
        assert set(defs["properties"]["type"]["enum"]) == {
            "person", "party", "company", "country", "institution", "other",
        }

    def test_both_actor_keys_are_required(self):
        """Unlike Quantity/Voice there is no useful half of this object: a name
        with no type cannot answer the different-subject-type test, and a type with
        no name cannot answer the named-actor one."""
        with pytest.raises(ValidationError):
            ClaimActor(name="Party Y")
        with pytest.raises(ValidationError):
            ClaimActor(type="party")

    def test_a_bare_string_is_dropped_not_raised(self, caplog):
        """The cheapest way for a model to fail here, and the one that must not
        cost an article: `complete_structured` runs instructor with
        `max_retries=1`, so a raise out of ExtractionOutput drops a real article
        from a real forecast. Logged rather than silent, because a silent None
        makes "answered badly" indistinguishable from "did not answer" — the one
        distinction a shadow field's fill rate exists to draw.
        """
        with caplog.at_level("WARNING"):
            out = ExtractionOutput.model_validate({
                "predictions": [], "claim_actor": "Party Y",
            })
        assert out.claim_actor is None
        assert "event=claim_actor_malformed" in caplog.text

    def test_an_out_of_enum_type_is_dropped_not_raised(self, caplog):
        with caplog.at_level("WARNING"):
            out = ExtractionOutput.model_validate({
                "predictions": [], "claim_actor": {"name": "Party Y", "type": "coalition"},
            })
        assert out.claim_actor is None
        assert "event=claim_actor_malformed" in caplog.text

    def test_a_malformed_actor_leaves_its_siblings_alone(self):
        """The guard nulls one field, not the decomposition. Two of three answers
        is still two more than the regex proxy has."""
        out = ExtractionOutput.model_validate({
            "predictions": [],
            "claim_actor": "Party Y",
            "claim_predicate": "withdraws from the parliamentary race",
            "claim_scope": "before the election",
        })
        assert out.claim_actor is None
        assert out.claim_predicate == "withdraws from the parliamentary race"
        assert out.claim_scope == "before the election"

    def test_a_double_serialized_actor_is_parsed(self):
        """Some models in TOOLS mode emit a nested object as a JSON string —
        `_coerce_nested_json_string`, the same treatment reader_confidence,
        quantity and voice get one level down."""
        out = ExtractionOutput.model_validate({
            "predictions": [], "claim_actor": '{"name": "Turkey", "type": "country"}',
        })
        assert out.claim_actor.name == "Turkey"
        assert out.claim_actor.type == "country"

    def test_constructing_the_model_directly_still_raises(self):
        """Strictness kept where it belongs. The guard buys tolerance on the LLM
        boundary only; a stored row or a caller building the object gets the
        validation error."""
        with pytest.raises(ValidationError):
            ClaimActor(name="Party Y", type="coalition")


class TestQuantity:
    """`quantity` {value, unit, comparator, value_hi, as_of} — the number the
    article gives, so that whether it satisfies the question can be decided in
    code (Oracle 1.5 Phase 1, retro#683).

    Shadow: populated and persisted, read by nothing. What these pin is the
    boundary between a strict model and a tolerant call site — the field is only
    worth harvesting if "the model did not answer", "the model answered badly"
    and "the model answered" stay three distinguishable states, and only worth
    comparing in code if a half-answer never reaches the comparison.
    """

    @staticmethod
    def _pred(**over):
        base = dict(quote="q", claim="c", stance=0.4, claim_strength=0.65)
        return PredictionExtraction(**{**base, **over})

    def test_a_stated_level_round_trips(self):
        pred = self._pred(quantity={"value": 8.75, "unit": "percent", "comparator": "="})
        assert (pred.quantity.value, pred.quantity.unit) == (8.75, "percent")
        assert pred.quantity.comparator == "="
        assert pred.quantity.value_hi is None and pred.quantity.as_of is None

    def test_a_range_carries_both_bounds(self):
        pred = self._pred(quantity={
            "value": 1.8, "unit": "million containers", "comparator": "between", "value_hi": 2.2,
        })
        assert (pred.quantity.value, pred.quantity.value_hi) == (1.8, 2.2)

    def test_it_defaults_to_none(self):
        """Every row extracted before v9 has no quantity, and so does every
        prediction that reports no figure — the two look the same on purpose."""
        assert self._pred().quantity is None

    @pytest.mark.parametrize("bad", [
        {"value": 36},                                                   # no unit
        {"unit": "seats", "comparator": "="},                            # no value
        {"value": 36, "unit": "seats", "comparator": "at least"},        # not an enum member
        {"value": 2, "unit": "percent", "comparator": "between"},        # between, no upper bound
        {"value": 3, "unit": "percent", "comparator": "between", "value_hi": 2},   # inverted
        {"value": 9, "unit": "percent", "comparator": "<=", "value_hi": 12},       # stray bound
        "about nine percent",
        ["9", "percent"],
    ])
    def test_a_malformed_answer_is_dropped_and_the_prediction_survives(self, bad, caplog):
        """Strict model, tolerant boundary — the retro#681/#686 trade, applied to
        a field with more ways to be wrong than either: five keys, a closed enum
        and a conditional requirement between two of them. `complete_structured`
        runs instructor with max_retries=1, so a still-malformed value would raise
        out of ExtractionOutput and drop the whole ARTICLE from the forecast."""
        import logging

        with caplog.at_level(logging.WARNING, logger="tm.models"):
            pred = self._pred(quantity=bad)
        assert pred.quantity is None
        assert pred.claim_strength == 0.65, "the rest of the prediction must be untouched"
        assert "event=quantity_malformed" in caplog.text, (
            "a silent drop would make 'answered badly' look like 'did not answer', "
            "which is the one distinction the fill rate exists to draw"
        )

    def test_a_double_serialized_object_is_parsed(self):
        """Some models in TOOLS mode emit a nested object as a JSON string
        (retro#306). `quantity` is the second nested field inside a prediction and
        needs the same coercion `reader_confidence` gets."""
        pred = self._pred(quantity='{"value": 214, "unit": "daily departures", "comparator": "="}')
        assert pred.quantity == Quantity(value=214, unit="daily departures", comparator="=")

    def test_constructing_it_directly_still_raises(self):
        """Strictness is relaxed at the LLM boundary, not in the type. Code that
        builds a Quantity is code that is about to compare it, and a comparison
        against a half-answer produces a confident wrong sign, not an abstention."""
        with pytest.raises(ValidationError):
            Quantity(value=36)
        with pytest.raises(ValueError):
            Quantity(value=2, unit="percent", comparator="between")

    def test_it_survives_a_dump_validate_round_trip(self):
        """daatan persists this inside claims_detail verbatim (daatan#1235/#1645)."""
        pred = self._pred(quantity={
            "value": 40, "unit": "million tonnes", "comparator": "<", "as_of": "2026-06-30",
        })
        assert PredictionExtraction.model_validate(pred.model_dump()).quantity == pred.quantity

    def test_the_llm_schema_offers_exactly_six_comparators(self):
        """The prompt names six; the schema must accept exactly those six. A
        drift either way means the model is told to answer something it cannot
        express, or can express something the code-side comparison cannot read."""
        schema = PredictionExtraction.model_json_schema()
        assert "quantity" in schema["properties"]
        comparator = schema["$defs"]["Quantity"]["properties"]["comparator"]
        assert set(comparator["enum"]) == {"=", "<", "<=", ">", ">=", "between"}

    def test_it_does_not_pay_for_a_docstring(self):
        """Pydantic copies a model docstring into the JSON schema description,
        and that schema IS the tool definition sent on every call (retro#700).
        `ReaderConfidence`'s first draft shipped 1,283 chars of rationale this
        way; `Quantity` keeps its own in `#` comments above the class."""
        assert Quantity.model_json_schema().get("description", "") == ""

    def test_it_is_independent_of_quantitative_estimate(self):
        """The two are near neighbours and the prompt spends a paragraph keeping
        them apart, so the schema must let a claim carry one without the other:
        a seat count is not a probability (retro#362), and a cited probability is
        not a level."""
        share = self._pred(quantity={"value": 28, "unit": "percent", "comparator": "="})
        assert share.quantitative_estimate is None
        probability = self._pred(quantitative_estimate=0.22)
        assert probability.quantity is None


class TestTone:
    """`tone` — approve | neutral | alarm, the quote's own register (Oracle 1.5
    Phase 1, retro#684).

    The leak retro#326 and retro#657 keep patching by prompt is a PROJECTION
    problem: an evaluation read as a direction. These pin the separation the
    field exists to create — a tone value never moves stance, and the two vary
    independently — because a `tone` that merely echoes the sign of `stance`
    would pass every fill-rate check while carrying no information at all.
    """

    @staticmethod
    def _pred(**over):
        base = dict(quote="q", claim="c", stance=0.4, claim_strength=0.65)
        return PredictionExtraction(**{**base, **over})

    @pytest.mark.parametrize("value", ["approve", "neutral", "alarm"])
    def test_every_member_lands(self, value):
        assert self._pred(tone=value).tone == value

    def test_it_defaults_to_none(self):
        """Every row extracted before v10 has none. The prompt asks for `tone` on
        every prediction, `neutral` included, so on a v10 row None means the model
        did not answer — not that the quote was even-handed. That distinction is
        the whole fill-rate measurement, and a default would erase it."""
        assert self._pred().tone is None

    @pytest.mark.parametrize("bad", ["alarming", "positive", "ALARM", "", 3, {"tone": "alarm"}])
    def test_an_out_of_enum_answer_is_dropped_and_the_prediction_survives(self, bad, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="tm.models"):
            pred = self._pred(tone=bad)
        assert pred.tone is None
        assert pred.claim_strength == 0.65, "the rest of the prediction must be untouched"
        assert "event=tone_malformed" in caplog.text

    def test_it_is_independent_of_stance(self):
        """The field's reason for existing and its acceptance test. A quote can
        WARN about something while being strong evidence FOR it — "a catastrophic
        breach that hands them the city" is alarm at stance +1.0 — and the whole
        point of eliciting the evaluation separately is that rating can then
        project it out. A schema that could not express that pair would be
        measuring the same axis twice."""
        alarmed_but_positive = self._pred(stance=1.0, tone="alarm")
        assert (alarmed_but_positive.stance, alarmed_but_positive.tone) == (1.0, "alarm")
        approving_but_negative = self._pred(stance=-0.9, tone="approve")
        assert (approving_but_negative.stance, approving_but_negative.tone) == (-0.9, "approve")

    def test_the_llm_schema_offers_exactly_three_values(self):
        """The TONE block names three; the schema must accept exactly those. A
        drift either way means the model is told to answer something it cannot
        express, or can express something no consumer was told to expect."""
        schema = PredictionExtraction.model_json_schema()
        members = next(
            b["enum"] for b in schema["properties"]["tone"]["anyOf"] if "enum" in b
        )
        assert set(members) == {"approve", "neutral", "alarm"}


class TestVoice:
    """`voice` {kind, attributed_to} — whose assertion the quote is (retro#684).

    A wire carried by thirty outlets is one observation. What these pin is that
    the informative half survives a partial answer: `kind` alone is already a
    usable observation, so the guards must never let a stray `attributed_to`
    take it down with it.
    """

    @staticmethod
    def _pred(**over):
        base = dict(quote="q", claim="c", stance=0.4, claim_strength=0.65)
        return PredictionExtraction(**{**base, **over})

    @pytest.mark.parametrize("kind", [
        "byline", "quoted_person", "institution", "wire", "unattributed",
    ])
    def test_every_kind_lands(self, kind):
        assert self._pred(voice={"kind": kind}).voice.kind == kind

    def test_a_named_source_carries_its_name(self):
        pred = self._pred(voice={"kind": "wire", "attributed_to": "Reuters"})
        assert (pred.voice.kind, pred.voice.attributed_to) == ("wire", "Reuters")

    def test_it_defaults_to_none(self):
        assert self._pred().voice is None

    def test_a_name_beside_a_byline_is_kept_not_rejected(self):
        """Deliberate, and the reason `Voice` has no cross-field validator where
        `Quantity` does. The prompt tells the model to omit `attributed_to` on a
        byline; a model that ignores that has still answered the question that
        matters. Rejecting the pair would hand `_drop_malformed_voice` a raise,
        and the guard nulls the WHOLE object — trading a stray string for a lost
        `kind`, which is a lost observation."""
        pred = self._pred(voice={"kind": "byline", "attributed_to": "the reporter"})
        assert pred.voice.kind == "byline"

    @pytest.mark.parametrize("bad", [
        {},                                              # no kind
        {"attributed_to": "Reuters"},                    # the name without the kind
        {"kind": "editorial"},                           # not an enum member
        {"kind": "Wire"},                                # right member, wrong case
        "Reuters",                                       # the bare string, not the object
        ["wire", "Reuters"],
    ])
    def test_a_malformed_answer_is_dropped_and_the_prediction_survives(self, bad, caplog):
        """`complete_structured` runs instructor with max_retries=1, so a
        still-malformed value would raise out of ExtractionOutput and drop the
        whole ARTICLE from the forecast — far too much to pay for a field that
        nothing reads yet."""
        import logging

        with caplog.at_level(logging.WARNING, logger="tm.models"):
            pred = self._pred(voice=bad)
        assert pred.voice is None
        assert pred.claim_strength == 0.65, "the rest of the prediction must be untouched"
        assert "event=voice_malformed" in caplog.text, (
            "a silent drop would make 'answered badly' look like 'did not answer'"
        )

    def test_a_double_serialized_object_is_parsed(self):
        """Some models in TOOLS mode emit a nested object as a JSON string
        (retro#306). `voice` is the third nested field inside a prediction and
        needs the same coercion `reader_confidence` and `quantity` get."""
        pred = self._pred(voice='{"kind": "institution", "attributed_to": "the Bank of Israel"}')
        assert pred.voice == Voice(kind="institution", attributed_to="the Bank of Israel")

    def test_constructing_it_directly_still_raises(self):
        """Strictness is relaxed at the LLM boundary, not in the type."""
        with pytest.raises(ValidationError):
            Voice(attributed_to="Reuters")
        with pytest.raises(ValidationError):
            Voice(kind="editorial")

    def test_it_survives_a_dump_validate_round_trip(self):
        """daatan persists this inside claims_detail verbatim (daatan#1235/#1645)."""
        pred = self._pred(voice={"kind": "quoted_person", "attributed_to": "Minister X"})
        assert PredictionExtraction.model_validate(pred.model_dump()).voice == pred.voice

    def test_it_does_not_pay_for_a_docstring(self):
        """Pydantic copies a model docstring into the JSON schema description, and
        that schema IS the tool definition sent on every call (retro#700).
        `Voice` keeps its rationale in `#` comments above the class, like
        `Quantity`."""
        assert Voice.model_json_schema().get("description", "") == ""

    def test_the_llm_schema_offers_exactly_five_kinds(self):
        schema = PredictionExtraction.model_json_schema()
        assert "voice" in schema["properties"]
        assert set(schema["$defs"]["Voice"]["properties"]["kind"]["enum"]) == {
            "byline", "quoted_person", "institution", "wire", "unattributed",
        }


class TestGrounds:
    """`grounds` {kind, basis} — what the position rests on (Oracle 1.5 Phase 1,
    retro#763, unparked from retro#673 §1).

    The pool counts articles where it means to count reasons. What these pin is
    the same boundary `Voice` has: the closed pick survives a partial answer, a
    malformed answer costs the field and never the claim, and the field is a
    separate axis from `evidence_class` rather than an extension of it.
    """

    KINDS = (
        "observed_milestone", "official_statement", "market_or_poll_figure",
        "analyst_inference", "precedent_or_base_rate", "authors_judgement",
    )

    @staticmethod
    def _pred(**over):
        base = dict(quote="q", claim="c", stance=0.4, claim_strength=0.65)
        return PredictionExtraction(**{**base, **over})

    @pytest.mark.parametrize("kind", KINDS)
    def test_every_kind_lands(self, kind):
        assert self._pred(grounds={"kind": kind}).grounds.kind == kind

    def test_the_basis_rides_with_the_kind(self):
        pred = self._pred(grounds={"kind": "official_statement",
                                   "basis": "the ministry's 12 March statement"})
        assert (pred.grounds.kind, pred.grounds.basis) == (
            "official_statement", "the ministry's 12 March statement")

    def test_it_defaults_to_none(self):
        """Every row extracted before v12 has none, and the prompt asks for it on
        every prediction, so on a v12 row None means the model did not answer."""
        assert self._pred().grounds is None

    def test_a_kind_without_a_basis_is_kept(self):
        """`basis` is optional on purpose: the pick is what the n_eff is taken
        over, and a model that answers it and skips the phrase has answered the
        question the consumer asks first. Requiring the phrase would hand the drop
        guard a raise and null the pick with it."""
        assert self._pred(grounds={"kind": "authors_judgement"}).grounds.basis is None

    def test_it_is_a_separate_axis_from_evidence_class(self):
        """The issue's own rule: class is the ROUTE the information took, grounds
        is WHAT WAS SEEN. A columnist reasoning from a poll is `opinion` by class
        and `market_or_poll_figure` by grounds, and the schema must let the two
        vary independently — a field that merely re-labelled `evidence_class`
        would fill at 100% and carry nothing."""
        pred = self._pred(evidence_class="opinion",
                          grounds={"kind": "market_or_poll_figure", "basis": "the 41% Ipsos figure"})
        assert pred.evidence_class == "opinion"
        assert pred.grounds.kind == "market_or_poll_figure"

    @pytest.mark.parametrize("bad", [
        {},                                        # no kind
        {"basis": "the ministry said so"},         # the phrase without the pick
        {"kind": "rumour"},                        # not an enum member
        {"kind": "Official_Statement"},            # right member, wrong case
        "the ministry said so",                    # the bare string, not the object
        ["official_statement", "the ministry"],
    ])
    def test_a_malformed_answer_is_dropped_and_the_prediction_survives(self, bad, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="tm.models"):
            pred = self._pred(grounds=bad)
        assert pred.grounds is None
        assert pred.claim_strength == 0.65, "the rest of the prediction must be untouched"
        assert "event=grounds_malformed" in caplog.text, (
            "a silent drop would make 'answered badly' look like 'did not answer'"
        )

    def test_a_double_serialized_object_is_parsed(self):
        """retro#306: some models emit a nested object as a JSON string."""
        pred = self._pred(grounds='{"kind": "precedent_or_base_rate", "basis": "no incumbent has lost since 1992"}')
        assert pred.grounds == Grounds(kind="precedent_or_base_rate",
                                       basis="no incumbent has lost since 1992")

    def test_constructing_it_directly_still_raises(self):
        with pytest.raises(ValidationError):
            Grounds(basis="the ministry said so")
        with pytest.raises(ValidationError):
            Grounds(kind="rumour")

    def test_it_survives_a_dump_validate_round_trip(self):
        pred = self._pred(grounds={"kind": "observed_milestone", "basis": "the line opened on 3 May"})
        assert PredictionExtraction.model_validate(pred.model_dump()).grounds == pred.grounds

    def test_it_does_not_pay_for_a_docstring(self):
        """retro#700: a model docstring lands in the schema billed on every call."""
        assert Grounds.model_json_schema().get("description", "") == ""

    def test_the_llm_schema_offers_exactly_six_kinds(self):
        schema = PredictionExtraction.model_json_schema()
        assert "grounds" in schema["properties"]
        assert set(schema["$defs"]["Grounds"]["properties"]["kind"]["enum"]) == set(self.KINDS)


class TestEvidenceClassGuard:
    """An out-of-enum `evidence_class` costs the class, not the article (retro#763).

    Measured on v12's first A/B: both raters wrote the GROUNDS kind
    `official_statement` into `evidence_class` on four corpus cases, 5/5 runs
    each. `evidence_class` had always been strict, so every one of those runs
    raised out of ExtractionOutput — on a live forecast that is the article
    gone, deterministically, for a field confusion the prompt can only lower
    and never rule out.
    """

    @staticmethod
    def _pred(**over):
        base = dict(quote="q", claim="c", stance=0.4, claim_strength=0.65)
        return PredictionExtraction(**{**base, **over})

    @pytest.mark.parametrize("value", [
        "reported_fact", "cited_probability", "cited_share", "reporting", "opinion",
    ])
    def test_every_real_class_still_lands(self, value):
        assert self._pred(evidence_class=value).evidence_class == value

    def test_a_grounds_kind_written_as_the_class_is_dropped_and_the_claim_survives(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="tm.models"):
            pred = self._pred(evidence_class="official_statement",
                              grounds={"kind": "official_statement", "basis": "the minister's announcement"})
        assert pred.evidence_class is None
        assert pred.grounds.kind == "official_statement", "the right field keeps the answer"
        assert pred.claim_strength == 0.65
        assert "event=evidence_class_malformed" in caplog.text

    def test_none_and_absent_are_untouched(self):
        assert self._pred().evidence_class is None
        assert self._pred(evidence_class=None).evidence_class is None
