"""Turn a predicted remaining life into a maintenance decision, and price it.

Run with:  python -m src.decision

This is the part of the project that has an opinion. The model says "about 23 cycles
left"; a plant manager needs "pull it now" or "it can wait". Bridging those two is not a
modelling problem, it is a cost problem, and the cost assumption is stated rather than
hidden (`docs/00-scoping.md`).

**The threshold is not chosen, it is found.** Every candidate threshold is replayed
across the whole fleet, the total cost of each is computed, and the cheapest wins. The
cost ratio is a slider rather than a constant because it is an assumption, and an
assumption should be visible and arguable.

What the ratio actually moves, on this dataset, is not the threshold. Every engine's
prediction eventually drops near zero, so a very low threshold already catches the whole
fleet and raising it prevents nothing further; the cost-optimal threshold barely responds
to the ratio at all. What the ratio moves is **whether condition-based monitoring is
worth doing** — from an 8% loss against a calendar at 1x, to a 51% saving at 20x. See
`docs/03-decision.md`, section 6.

Three policies are priced the same way so they can be compared honestly:

- **run to failure** — repair only after a breakdown
- **calendar** — overhaul every N cycles regardless of condition, with N optimised by
  the same search, because beating a badly tuned baseline would prove nothing
- **condition-based** — this project

Every explanation is generated from rules. Same inputs, same sentence, every run, with a
line of code behind each clause. Text that is rephrased on each call cannot be audited,
and a maintenance instruction that cannot be audited is a liability.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config, data_loader, features, model as model_module

ACTION_INTERVENE = "intervene now"
ACTION_MONITOR = "monitor closely"
ACTION_HEALTHY = "healthy"


# --------------------------------------------------------------------------------------
# What things cost
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """What each outcome costs, in units of one planned preventive overhaul."""

    failure: float
    per_wasted_cycle: float
    cost_ratio: float
    preventive: float = config.COST_PREVENTIVE

    @classmethod
    def from_ratio(cls, cost_ratio: float, nominal_life: float) -> CostModel:
        """Build the cost model from the one number the user actually chooses.

        Args:
            cost_ratio: how many planned overhauls one unplanned failure costs.
            nominal_life: median engine lifetime, read from the fleet.

        Raises:
            ValueError: if the ratio is outside the range the dashboard offers.
        """
        return cls(
            failure=config.cost_of_failure(cost_ratio),
            per_wasted_cycle=config.cost_per_wasted_cycle(nominal_life),
            cost_ratio=cost_ratio,
        )

    def total(self, interventions: int, failures: int, wasted_cycles: int) -> float:
        """Total cost of a fleet outcome."""
        return (
            interventions * self.preventive
            + failures * self.failure
            + wasted_cycles * self.per_wasted_cycle
        )


@dataclass(frozen=True)
class FleetOutcome:
    """What happened to a fleet under one policy, and what it cost."""

    engines: int
    interventions: int
    failures: int
    wasted_cycles: int
    total_cost: float

    @property
    def cost_per_engine(self) -> float:
        return self.total_cost / self.engines

    def saving_versus(self, baseline: FleetOutcome) -> float:
        """Fraction of `baseline`'s cost avoided. Negative means this policy is worse."""
        return 1.0 - (self.total_cost / baseline.total_cost)


# --------------------------------------------------------------------------------------
# Replaying a policy across a fleet
# --------------------------------------------------------------------------------------


def replay_condition_based(
    frame: pd.DataFrame, predicted_rul: pd.Series, threshold: int, costs: CostModel
) -> FleetOutcome:
    """Pull each engine the first cycle its predicted life drops to `threshold`.

    An engine that never triggers runs until it breaks. One that triggers is overhauled
    on our terms, throwing away whatever life it genuinely had left.
    """
    features.require_chronological_order(frame)

    triggered = frame[predicted_rul <= threshold]
    first_trigger = triggered.groupby(config.UNIT_COLUMN, sort=False).first()

    engines = frame[config.UNIT_COLUMN].nunique()
    interventions = len(first_trigger)
    wasted = int(first_trigger[config.RUL_COLUMN].sum())

    return _outcome(engines, interventions, wasted, costs)


def replay_calendar(frame: pd.DataFrame, interval: int, costs: CostModel) -> FleetOutcome:
    """Overhaul every engine at cycle `interval`, whatever its condition.

    An engine that does not survive to that cycle has already failed. This is what most
    plants actually do, and it is the baseline that matters.
    """
    lifetimes = _lifetimes(frame)
    survived_to_the_date = lifetimes >= interval

    interventions = int(survived_to_the_date.sum())
    wasted = int((lifetimes[survived_to_the_date] - interval).sum())

    return _outcome(len(lifetimes), interventions, wasted, costs)


def replay_run_to_failure(frame: pd.DataFrame, costs: CostModel) -> FleetOutcome:
    """Never intervene. Every engine breaks, nothing is wasted, everything is unplanned."""
    engines = frame[config.UNIT_COLUMN].nunique()
    return _outcome(engines, interventions=0, wasted_cycles=0, costs=costs)


def replay_threshold_grid(frame: pd.DataFrame, predicted_rul: pd.Series) -> pd.DataFrame:
    """Fleet outcome at every candidate threshold, before any cost is applied.

    What happens to a fleet does not depend on the cost model; only the price of what
    happened does. Separating the two is what makes the cost ratio cheap to move: replay
    the fleet once, then price the same outcomes as many times as the reader drags the
    slider. Without it, drawing the saving curve across twenty cost ratios would mean
    replaying several thousand fleets.

    Returns:
        One row per threshold, indexed by threshold, with the intervention, failure and
        wasted-cycle counts.
    """
    features.require_chronological_order(frame)
    engines = frame[config.UNIT_COLUMN].nunique()

    rows = []
    for threshold in range(config.THRESHOLD_GRID_MIN, config.THRESHOLD_GRID_MAX + 1):
        triggered = frame[predicted_rul <= threshold]
        first_trigger = triggered.groupby(config.UNIT_COLUMN, sort=False).first()
        interventions = len(first_trigger)
        rows.append(
            {
                "threshold": threshold,
                "interventions": interventions,
                "failures": engines - interventions,
                "wasted_cycles": int(first_trigger[config.RUL_COLUMN].sum()),
            }
        )

    return pd.DataFrame(rows).set_index("threshold")


def replay_calendar_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Fleet outcome at every candidate calendar interval, before any cost is applied."""
    lifetimes = _lifetimes(frame)

    rows = []
    for interval in range(
        config.CALENDAR_INTERVAL_GRID_MIN, config.CALENDAR_INTERVAL_GRID_MAX + 1
    ):
        survived = lifetimes >= interval
        interventions = int(survived.sum())
        rows.append(
            {
                "interval": interval,
                "interventions": interventions,
                "failures": len(lifetimes) - interventions,
                "wasted_cycles": int((lifetimes[survived] - interval).sum()),
            }
        )

    return pd.DataFrame(rows).set_index("interval")


def price_grid(grid: pd.DataFrame, costs: CostModel) -> pd.Series:
    """Total cost of every candidate in a replayed grid."""
    return (
        grid["interventions"] * costs.preventive
        + grid["failures"] * costs.failure
        + grid["wasted_cycles"] * costs.per_wasted_cycle
    )


def outcome_at(grid: pd.DataFrame, candidate: int, costs: CostModel) -> FleetOutcome:
    """The priced outcome of one candidate already present in a replayed grid."""
    row = grid.loc[candidate]
    interventions = int(row["interventions"])
    failures = int(row["failures"])
    wasted = int(row["wasted_cycles"])
    return FleetOutcome(
        engines=interventions + failures,
        interventions=interventions,
        failures=failures,
        wasted_cycles=wasted,
        total_cost=costs.total(interventions, failures, wasted),
    )


def cheapest_threshold(
    grid: pd.DataFrame,
    costs: CostModel,
    minimum_lead_time: int = config.MINIMUM_LEAD_TIME,
) -> int:
    """The cheapest threshold in an already-replayed grid that respects the lead time."""
    floor = max(config.THRESHOLD_GRID_MIN, minimum_lead_time)
    affordable_notice = grid[grid.index >= floor]
    return int(price_grid(affordable_notice, costs).idxmin())


def best_threshold(
    frame: pd.DataFrame,
    predicted_rul: pd.Series,
    costs: CostModel,
    minimum_lead_time: int = config.MINIMUM_LEAD_TIME,
) -> int:
    """The cheapest intervention threshold that still gives the shop time to act.

    Nothing is hand-picked: every candidate is replayed across the fleet and priced.

    `minimum_lead_time` floors the search. Left unconstrained, the cost curve is a cliff
    followed by a gentle slope — once the threshold is high enough to catch every engine,
    raising it further only wastes good life — so the optimum lands on the cliff edge and
    recommends intervening a handful of cycles before failure. That is arithmetically
    correct and operationally useless: no shop sources a part in five flights. The floor
    is where that operational reality enters the arithmetic.

    Args:
        minimum_lead_time: cycles of notice the maintenance organisation needs. Pass
            `config.THRESHOLD_GRID_MIN` to see the unconstrained optimum.
    """
    grid = replay_threshold_grid(frame, predicted_rul)
    return cheapest_threshold(grid, costs, minimum_lead_time)


def best_calendar_interval(frame: pd.DataFrame, costs: CostModel) -> int:
    """The calendar interval that costs least on this fleet.

    The baseline gets the same optimisation as our own policy. Beating a deliberately
    badly tuned calendar would prove nothing.
    """
    return int(price_grid(replay_calendar_grid(frame), costs).idxmin())


# --------------------------------------------------------------------------------------
# The decision itself
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Recommendation:
    """One engine's instruction, plus everything needed to justify it."""

    action: str
    predicted_rul: float
    act_threshold: int
    watch_threshold: int
    cost_ratio: float
    model_error: float
    recheck_in: int | None

    def explain(self) -> str:
        """Two or three sentences a non-technical reader can act on.

        Rule-generated, so the same engine always produces the same words.
        """
        if self.action == ACTION_INTERVENE:
            return (
                f"Intervene now. Predicted remaining life is {self.predicted_rul:.0f} "
                f"cycles, at or below the {self.act_threshold}-cycle action threshold. "
                f"That threshold is where an unplanned failure costing "
                f"{self.cost_ratio:.1f}x a planned overhaul makes pulling this engine "
                f"cheaper on average than running it further. Typical model error is "
                f"plus or minus {self.model_error:.0f} cycles, so treat this as a range, "
                f"not a date."
            )
        if self.action == ACTION_MONITOR:
            return (
                f"Monitor closely. Predicted remaining life is "
                f"{self.predicted_rul:.0f} cycles, above the "
                f"{self.act_threshold}-cycle action threshold that the assumed "
                f"{self.cost_ratio:.1f}x cost ratio sets, but by less than the model's "
                f"typical error of {self.model_error:.0f} cycles. The model calls this "
                f"engine safe and is not confident enough to be believed. Re-check "
                f"every cycle; do not schedule work yet."
            )
        return (
            f"No action. Predicted remaining life is {self.predicted_rul:.0f} cycles, "
            f"{self.predicted_rul - self.watch_threshold:.0f} clear of the point where "
            f"the model's uncertainty starts to matter. At the assumed "
            f"{self.cost_ratio:.1f}x cost ratio the action threshold sits at "
            f"{self.act_threshold} cycles. Re-check in {self.recheck_in} cycles."
        )


@dataclass(frozen=True)
class DecisionPolicy:
    """The two thresholds that turn a prediction into an instruction."""

    act_threshold: int
    watch_threshold: int
    costs: CostModel
    model_error: float

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        predicted_rul: pd.Series,
        costs: CostModel,
        model_error: float,
        minimum_lead_time: int = config.MINIMUM_LEAD_TIME,
    ) -> DecisionPolicy:
        """Derive both thresholds from the cost model and the model's own error.

        The action threshold is the cheapest one found by replaying the fleet, subject to
        leaving the shop enough notice to act. The watch threshold sits exactly one
        typical model error above it: the band where the model says "safe" but is
        routinely wrong by that much. Making the uncertainty visible beats hiding it
        behind a single number.
        """
        act_threshold = best_threshold(frame, predicted_rul, costs, minimum_lead_time)
        return cls(
            act_threshold=act_threshold,
            watch_threshold=act_threshold + round(model_error),
            costs=costs,
            model_error=model_error,
        )

    def recommend(self, predicted_rul: float) -> Recommendation:
        """Classify one engine's predicted remaining life."""
        if predicted_rul <= self.act_threshold:
            action, recheck_in = ACTION_INTERVENE, None
        elif predicted_rul <= self.watch_threshold:
            action, recheck_in = ACTION_MONITOR, None
        else:
            action = ACTION_HEALTHY
            headroom = predicted_rul - self.watch_threshold
            recheck_in = max(1, int(headroom * config.RECHECK_SAFETY_FACTOR))

        return Recommendation(
            action=action,
            predicted_rul=predicted_rul,
            act_threshold=self.act_threshold,
            watch_threshold=self.watch_threshold,
            cost_ratio=self.costs.cost_ratio,
            model_error=self.model_error,
            recheck_in=recheck_in,
        )


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------


def _outcome(
    engines: int, interventions: int, wasted_cycles: int, costs: CostModel
) -> FleetOutcome:
    """Assemble a fleet outcome; every engine either gets pulled or breaks."""
    failures = engines - interventions
    return FleetOutcome(
        engines=engines,
        interventions=interventions,
        failures=failures,
        wasted_cycles=wasted_cycles,
        total_cost=costs.total(interventions, failures, wasted_cycles),
    )


def _lifetimes(frame: pd.DataFrame) -> pd.Series:
    """How many cycles each engine survived."""
    return frame.groupby(config.UNIT_COLUMN)[config.CYCLE_COLUMN].max()


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def main() -> None:
    """Tune on one set of engines, then price every policy on engines never seen."""
    trained = model_module.load()
    split = features.split_by_engine(data_loader.load_training_data())
    costs = CostModel.from_ratio(config.DEFAULT_COST_RATIO, trained.nominal_life)

    tuning_predictions = trained.predict(split.tuning)
    policy = DecisionPolicy.fit(
        split.tuning, tuning_predictions, costs, trained.validation_rmse
    )
    calendar_interval = best_calendar_interval(split.tuning, costs)

    _print_thresholds(split, costs, policy, tuning_predictions, calendar_interval)
    _print_outcomes(split.evaluation, trained, policy, costs, calendar_interval)
    _print_examples(split.evaluation, trained, policy)


def _print_thresholds(split, costs, policy, tuning_predictions, calendar_interval) -> None:
    """Show both optima, because the gap between them is the point."""
    unconstrained = best_threshold(
        split.tuning, tuning_predictions, costs, config.THRESHOLD_GRID_MIN
    )

    print(
        f"Tuned on {split.tuning[config.UNIT_COLUMN].nunique()} engines at a "
        f"{costs.cost_ratio:.1f}x cost ratio.\n"
        f"\n"
        f"  cheapest threshold, ignoring lead time  {unconstrained:>3} cycles  "
        f"<- optimal and unusable\n"
        f"  action threshold                        {policy.act_threshold:>3} cycles  "
        f"(floored at {config.MINIMUM_LEAD_TIME} cycles of notice)\n"
        f"  watch threshold                         {policy.watch_threshold:>3} cycles  "
        f"(+{round(policy.model_error)} for model error)\n"
        f"  calendar interval                       {calendar_interval:>3} cycles  "
        f"(baseline, tuned the same way)"
    )


def _print_outcomes(evaluation, trained, policy, costs, calendar_interval) -> None:
    """Price every policy on the engines held back for exactly this."""
    run_to_failure = replay_run_to_failure(evaluation, costs)
    calendar = replay_calendar(evaluation, calendar_interval, costs)
    condition_based = replay_condition_based(
        evaluation, trained.predict(evaluation), policy.act_threshold, costs
    )

    engines = evaluation[config.UNIT_COLUMN].nunique()
    print(f"\nPriced on {engines} engines that shaped neither the model nor the threshold:")
    print(f"\n  {'policy':34s} {'overhauls':>9s} {'failures':>8s} {'wasted':>8s} {'cost':>8s}")
    print("  " + "-" * 71)
    for name, outcome in (
        ("run to failure", run_to_failure),
        (f"calendar, every {calendar_interval} cycles", calendar),
        ("condition-based (this project)", condition_based),
    ):
        print(
            f"  {name:34s} {outcome.interventions:>9d} {outcome.failures:>8d} "
            f"{outcome.wasted_cycles:>8d} {outcome.total_cost:>8.1f}"
        )

    print(
        f"\n  Saving versus the tuned calendar baseline "
        f"{condition_based.saving_versus(calendar):>7.1%}\n"
        f"  Saving versus run to failure              "
        f"{condition_based.saving_versus(run_to_failure):>7.1%}"
    )


def _print_examples(evaluation, trained, policy) -> None:
    """One worked recommendation per action, so the text is shown, not just described."""
    print("\nWhat an operator reads:")
    predictions = trained.predict(evaluation)
    seen = set()

    for index, predicted in predictions.items():
        recommendation = policy.recommend(float(predicted))
        if recommendation.action in seen:
            continue
        seen.add(recommendation.action)
        engine = evaluation.loc[index, config.UNIT_COLUMN]
        cycle = evaluation.loc[index, config.CYCLE_COLUMN]
        print(f"\n  Engine {engine}, cycle {cycle}\n  {recommendation.explain()}")


if __name__ == "__main__":
    main()
