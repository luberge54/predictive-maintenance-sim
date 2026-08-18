"""Tests for the decision layer.

This is the module that produces the project's headline number, so the arithmetic behind
it gets checked directly rather than inferred from a plausible-looking report. Each
replay is tested on a fleet small enough that the expected cost can be worked out by
hand and written into the assertion.
"""

import pandas as pd
import pytest

from src import config, decision

NOMINAL_LIFE = 200.0
COST_RATIO = 5.0


@pytest.fixture
def costs():
    """One failure costs 5 overhauls; scrapping 200 cycles of life costs 1."""
    return decision.CostModel.from_ratio(COST_RATIO, NOMINAL_LIFE)


def fleet_with_predictions(engines: dict[int, list[float]]):
    """Build a fleet from per-cycle predictions.

    Each engine runs to failure, so its true remaining life counts down to zero on its
    last recorded cycle. `engines` maps an engine id to what the model predicted at each
    of its cycles, in order.
    """
    rows, predicted = [], []
    for unit, predictions in engines.items():
        lifetime = len(predictions)
        for offset, value in enumerate(predictions):
            cycle = offset + 1
            rows.append(
                {
                    config.UNIT_COLUMN: unit,
                    config.CYCLE_COLUMN: cycle,
                    config.RUL_COLUMN: lifetime - cycle,
                }
            )
            predicted.append(value)

    frame = pd.DataFrame(rows)
    return frame, pd.Series(predicted, index=frame.index)


# --------------------------------------------------------------------------------------
# What things cost
# --------------------------------------------------------------------------------------


def test_cost_model_is_built_from_the_single_ratio_the_user_chooses(costs):
    # Assert — everything is relative to one overhaul, so only the ratios matter
    assert costs.preventive == 1.0
    assert costs.failure == 5.0
    assert costs.per_wasted_cycle == pytest.approx(1.0 / NOMINAL_LIFE)


def test_scrapping_a_full_engine_life_costs_exactly_one_overhaul(costs):
    # Assert — this is the derivation, not a tuned constant
    assert costs.per_wasted_cycle * NOMINAL_LIFE == pytest.approx(costs.preventive)


def test_cost_model_refuses_a_ratio_outside_the_offered_range():
    # Act / Assert
    with pytest.raises(ValueError, match="cost_ratio must be between"):
        decision.CostModel.from_ratio(config.COST_RATIO_MAX + 1, NOMINAL_LIFE)


def test_total_cost_adds_the_three_terms(costs):
    # Act — 2 overhauls, 1 failure, 100 cycles thrown away
    total = costs.total(interventions=2, failures=1, wasted_cycles=100)

    # Assert — 2*1 + 1*5 + 100*0.005
    assert total == pytest.approx(7.5)


# --------------------------------------------------------------------------------------
# Replaying policies
# --------------------------------------------------------------------------------------


def test_run_to_failure_breaks_every_engine(costs):
    # Arrange
    frame, _ = fleet_with_predictions({1: [30, 20, 10], 2: [40, 30, 20, 10]})

    # Act
    outcome = decision.replay_run_to_failure(frame, costs)

    # Assert
    assert outcome.failures == 2
    assert outcome.interventions == 0
    assert outcome.wasted_cycles == 0
    assert outcome.total_cost == pytest.approx(2 * costs.failure)


def test_condition_based_pulls_the_engine_at_the_first_crossing(costs):
    # Arrange — the prediction dips to 20 at cycle 4 of a 5-cycle life
    frame, predictions = fleet_with_predictions({1: [50, 40, 30, 20, 10]})

    # Act
    outcome = decision.replay_condition_based(frame, predictions, 20, costs)

    # Assert — pulled at cycle 4, where 1 cycle of real life was still left
    assert outcome.interventions == 1
    assert outcome.failures == 0
    assert outcome.wasted_cycles == 1


def test_condition_based_ignores_later_crossings(costs):
    # Arrange — the prediction crosses the threshold at cycle 2 and stays below it
    frame, predictions = fleet_with_predictions({1: [50, 20, 15, 10, 5]})

    # Act
    outcome = decision.replay_condition_based(frame, predictions, 20, costs)

    # Assert — an engine is pulled once, on the first trigger, not the last
    assert outcome.wasted_cycles == 3


def test_condition_based_lets_an_engine_that_never_triggers_break(costs):
    # Arrange — the model never predicts anything low enough
    frame, predictions = fleet_with_predictions({1: [90, 80, 70]})

    # Act
    outcome = decision.replay_condition_based(frame, predictions, 20, costs)

    # Assert
    assert outcome.failures == 1
    assert outcome.total_cost == pytest.approx(costs.failure)


def test_condition_based_refuses_unsorted_rows(costs):
    # Arrange — shuffled cycles would make "the first crossing" meaningless
    frame, predictions = fleet_with_predictions({1: [50, 40, 30, 20, 10]})
    shuffled = frame.sample(frac=1.0, random_state=0)

    # Act / Assert
    with pytest.raises(ValueError, match="must be sorted"):
        decision.replay_condition_based(shuffled, predictions[shuffled.index], 20, costs)


def test_calendar_wastes_the_life_left_at_the_overhaul_date(costs):
    # Arrange — one engine lives 10 cycles, pulled on cycle 4
    frame, _ = fleet_with_predictions({1: [0] * 10})

    # Act
    outcome = decision.replay_calendar(frame, interval=4, costs=costs)

    # Assert
    assert outcome.interventions == 1
    assert outcome.wasted_cycles == 6


def test_calendar_loses_an_engine_that_dies_before_the_date(costs):
    # Arrange — the engine lives 3 cycles, the calendar says overhaul at 10
    frame, _ = fleet_with_predictions({1: [0] * 3})

    # Act
    outcome = decision.replay_calendar(frame, interval=10, costs=costs)

    # Assert — this is the failure mode of maintaining on a fixed schedule
    assert outcome.failures == 1
    assert outcome.wasted_cycles == 0


# --------------------------------------------------------------------------------------
# Finding the threshold
# --------------------------------------------------------------------------------------


def test_best_threshold_never_returns_less_notice_than_the_shop_needs(costs):
    # Arrange — a fleet whose unconstrained optimum is far below the lead time
    frame, predictions = fleet_with_predictions(
        {unit: list(range(60, 0, -1)) for unit in range(1, 6)}
    )

    # Act
    threshold = decision.best_threshold(frame, predictions, costs, minimum_lead_time=20)

    # Assert
    assert threshold >= 20


def test_best_threshold_can_be_asked_for_the_unconstrained_optimum(costs):
    # Arrange
    frame, predictions = fleet_with_predictions(
        {unit: list(range(60, 0, -1)) for unit in range(1, 6)}
    )

    # Act — passing the grid floor removes the operational constraint
    unconstrained = decision.best_threshold(
        frame, predictions, costs, minimum_lead_time=config.THRESHOLD_GRID_MIN
    )
    constrained = decision.best_threshold(frame, predictions, costs, minimum_lead_time=20)

    # Assert — the constraint can only ever push the threshold up
    assert unconstrained <= constrained


def test_best_threshold_beats_every_other_candidate_it_considered(costs):
    # Arrange
    frame, predictions = fleet_with_predictions(
        {1: list(range(40, 0, -1)), 2: list(range(70, 0, -1))}
    )

    # Act
    chosen = decision.best_threshold(frame, predictions, costs, minimum_lead_time=5)
    chosen_cost = decision.replay_condition_based(frame, predictions, chosen, costs).total_cost

    # Assert — no candidate above the floor may be cheaper
    for candidate in range(5, config.THRESHOLD_GRID_MAX + 1):
        rival = decision.replay_condition_based(frame, predictions, candidate, costs)
        assert rival.total_cost >= chosen_cost - 1e-9


def test_saving_is_negative_when_a_policy_is_worse_than_its_baseline():
    # Arrange
    expensive = decision.FleetOutcome(10, 0, 10, 0, total_cost=50.0)
    cheap = decision.FleetOutcome(10, 10, 0, 0, total_cost=10.0)

    # Act / Assert — a saving must be able to report bad news
    assert expensive.saving_versus(cheap) < 0
    assert cheap.saving_versus(expensive) == pytest.approx(0.8)


# --------------------------------------------------------------------------------------
# Turning a prediction into an instruction
# --------------------------------------------------------------------------------------


@pytest.fixture
def policy(costs):
    """A policy with round numbers, so the band edges can be asserted exactly."""
    return decision.DecisionPolicy(
        act_threshold=20, watch_threshold=33, costs=costs, model_error=13.0
    )


@pytest.mark.parametrize(
    ("predicted_rul", "expected"),
    [
        (5, decision.ACTION_INTERVENE),
        (20, decision.ACTION_INTERVENE),
        (21, decision.ACTION_MONITOR),
        (33, decision.ACTION_MONITOR),
        (34, decision.ACTION_HEALTHY),
        (200, decision.ACTION_HEALTHY),
    ],
)
def test_each_band_produces_its_own_instruction(policy, predicted_rul, expected):
    # Act
    recommendation = policy.recommend(predicted_rul)

    # Assert — the boundaries are inclusive on the lower side
    assert recommendation.action == expected


def test_a_healthy_engine_is_told_when_to_come_back(policy):
    # Act
    recommendation = policy.recommend(133)

    # Assert — halfway to the point where uncertainty starts to matter
    assert recommendation.recheck_in == 50


def test_a_barely_healthy_engine_is_never_told_to_come_back_in_zero_cycles(policy):
    # Act
    recommendation = policy.recommend(policy.watch_threshold + 1)

    # Assert
    assert recommendation.recheck_in >= 1


def test_the_same_engine_always_produces_the_same_sentence(policy):
    # Act — the reason explanations are rule-generated rather than model-generated
    first = policy.recommend(18).explain()
    second = policy.recommend(18).explain()

    # Assert
    assert first == second


def test_every_explanation_states_the_assumption_it_rests_on(policy):
    # Act / Assert — a recommendation without its cost assumption cannot be argued with
    for predicted_rul in (10, 25, 120):
        text = policy.recommend(predicted_rul).explain()
        assert "5.0x" in text
        assert str(policy.act_threshold) in text


def test_explanations_survive_a_console_that_cannot_print_unicode(policy):
    # Act / Assert — operator text lands in terminals whose encoding we do not control
    for predicted_rul in (10, 25, 120):
        text = policy.recommend(predicted_rul).explain()
        assert text.isascii()


def test_the_watch_band_is_exactly_one_model_error_wide(costs):
    # Arrange
    frame, predictions = fleet_with_predictions(
        {unit: list(range(60, 0, -1)) for unit in range(1, 6)}
    )

    # Act
    fitted = decision.DecisionPolicy.fit(frame, predictions, costs, model_error=13.4)

    # Assert — the band is the model's own uncertainty, not a number someone picked
    assert fitted.watch_threshold - fitted.act_threshold == 13


# --------------------------------------------------------------------------------------
# Replaying once and pricing many times
# --------------------------------------------------------------------------------------


def test_the_threshold_grid_agrees_with_replaying_one_threshold(costs):
    """The fast path and the obvious path must never disagree.

    The grid exists so the dashboard can reprice thousands of candidates without
    replaying them. That is only safe while it produces identical outcomes.
    """
    # Arrange
    frame, predictions = fleet_with_predictions(
        {1: list(range(40, 0, -1)), 2: list(range(70, 0, -1)), 3: [90] * 20}
    )
    grid = decision.replay_threshold_grid(frame, predictions)

    # Act / Assert
    for threshold in (1, 5, 20, 60, 150):
        direct = decision.replay_condition_based(frame, predictions, threshold, costs)
        from_grid = decision.outcome_at(grid, threshold, costs)
        assert from_grid == direct


def test_the_calendar_grid_agrees_with_replaying_one_interval(costs):
    # Arrange
    frame, _ = fleet_with_predictions({1: [0] * 120, 2: [0] * 260, 3: [0] * 40})
    grid = decision.replay_calendar_grid(frame)

    # Act / Assert
    for interval in (10, 50, 137, 300):
        direct = decision.replay_calendar(frame, interval, costs)
        assert decision.outcome_at(grid, interval, costs) == direct


def test_pricing_a_grid_matches_pricing_a_single_outcome(costs):
    # Arrange
    frame, predictions = fleet_with_predictions({1: list(range(30, 0, -1))})
    grid = decision.replay_threshold_grid(frame, predictions)

    # Act
    priced = decision.price_grid(grid, costs)

    # Assert
    for threshold in (2, 15, 90):
        row = grid.loc[threshold]
        expected = costs.total(
            int(row["interventions"]), int(row["failures"]), int(row["wasted_cycles"])
        )
        assert priced.loc[threshold] == pytest.approx(expected)


def test_an_engine_appears_exactly_once_at_every_threshold(costs):
    # Arrange — every engine is either pulled or lost, never both, never neither
    frame, predictions = fleet_with_predictions(
        {1: list(range(40, 0, -1)), 2: [90] * 20, 3: list(range(10, 0, -1))}
    )

    # Act
    grid = decision.replay_threshold_grid(frame, predictions)

    # Assert
    assert (grid["interventions"] + grid["failures"] == 3).all()
