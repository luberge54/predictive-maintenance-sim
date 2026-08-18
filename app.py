"""Streamlit dashboard for the Predictive Maintenance Simulator.

Run with:  streamlit run app.py

Lives at the project root rather than under `src/` so that `from src import ...` resolves
without a sys.path hack: Streamlit puts the script's own directory on the path.

The page has one job the report in `python -m src.decision` cannot do: let the reader move
the cost assumption and watch the recommendation move with it. Everything else here is
presentation.

Two fleets are shown, and they are deliberately not the same engines:

- **the backtest** — evaluation engines from the training file, which ran to failure, so a
  policy can be replayed on them and priced
- **the live fleet** — the NASA test engines, whose recording stops before they fail. That
  is exactly the operational situation: still flying, outcome unknown. A policy cannot be
  scored on them, but a recommendation can be made for them.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, data_loader, decision, features, model as model_module

SENSORS_CHARTED_BY_DEFAULT = 3
URGENT_ENGINES_SHOWN = 12

ACTION_ICONS = {
    decision.ACTION_INTERVENE: "🔴 intervene now",
    decision.ACTION_MONITOR: "🟠 monitor closely",
    decision.ACTION_HEALTHY: "🟢 healthy",
}


# --------------------------------------------------------------------------------------
# Loading, cached so moving a slider does not retrain or re-predict anything
# --------------------------------------------------------------------------------------


@st.cache_resource
def load_trained_model():
    """The fitted model. `cache_resource` because it is a shared object, not data."""
    return model_module.load()


@st.cache_data
def load_backtest_fleet() -> tuple[pd.DataFrame, pd.Series]:
    """Engines held back to price policies: never fitted on, never used to tune."""
    split = features.split_by_engine(data_loader.load_training_data())
    return split.evaluation, load_trained_model().predict(split.evaluation)


@st.cache_data
def load_tuning_fleet() -> tuple[pd.DataFrame, pd.Series]:
    """Engines the thresholds are derived from."""
    split = features.split_by_engine(data_loader.load_training_data())
    return split.tuning, load_trained_model().predict(split.tuning)


@st.cache_data
def load_live_fleet() -> tuple[pd.DataFrame, pd.Series]:
    """The NASA test engines: still running, outcome not yet reached."""
    fleet = data_loader.load_test_data()
    return fleet, load_trained_model().predict(fleet)


@st.cache_data
def tuning_grids() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every candidate replayed on the tuning fleet, before any cost is applied."""
    tuning, predictions = load_tuning_fleet()
    return (
        decision.replay_threshold_grid(tuning, predictions),
        decision.replay_calendar_grid(tuning),
    )


@st.cache_data
def backtest_grids() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same, on the engines the saving is reported against."""
    fleet, predictions = load_backtest_fleet()
    return (
        decision.replay_threshold_grid(fleet, predictions),
        decision.replay_calendar_grid(fleet),
    )


def derive_policy(cost_ratio: float, lead_time: int) -> dict:
    """Re-derive both thresholds for one cost assumption.

    Cheap, because the fleets were replayed once and only the pricing changes. Moving a
    slider re-prices a stored table; it does not re-run 130 fleet replays.
    """
    trained = load_trained_model()
    thresholds, calendar = tuning_grids()
    costs = decision.CostModel.from_ratio(cost_ratio, trained.nominal_life)

    act_threshold = decision.cheapest_threshold(thresholds, costs, lead_time)
    return {
        "act_threshold": act_threshold,
        "watch_threshold": act_threshold + round(trained.validation_rmse),
        "unconstrained_threshold": decision.cheapest_threshold(
            thresholds, costs, config.THRESHOLD_GRID_MIN
        ),
        "calendar_interval": int(decision.price_grid(calendar, costs).idxmin()),
    }


@st.cache_data
def assumption_sweep(lead_time: int) -> pd.DataFrame:
    """Re-derive and re-price the policy across the whole range of cost ratios.

    The scoping document asks for this explicitly: the honest answer to "where does your
    cost ratio come from?" is "it is yours to set — here is the curve", including the
    part of the curve where this tool is not worth using.
    """
    trained = load_trained_model()
    tuning_thresholds, tuning_calendar = tuning_grids()
    backtest_thresholds, backtest_calendar = backtest_grids()

    rows = []
    ratio = config.COST_RATIO_MIN
    while ratio <= config.COST_RATIO_MAX:
        costs = decision.CostModel.from_ratio(ratio, trained.nominal_life)
        threshold = decision.cheapest_threshold(tuning_thresholds, costs, lead_time)
        interval = int(decision.price_grid(tuning_calendar, costs).idxmin())

        ours = decision.outcome_at(backtest_thresholds, threshold, costs)
        calendar = decision.outcome_at(backtest_calendar, interval, costs)
        rows.append(
            {
                "cost ratio": ratio,
                # Stored as a percentage, not a fraction: the chart axis shows this
                # value verbatim, and "0.5" reads worse than "50" beside a % label.
                "saving vs calendar": ours.saving_versus(calendar) * 100,
                "intervention threshold": threshold,
            }
        )
        ratio += config.COST_RATIO_STEP

    return pd.DataFrame(rows).set_index("cost ratio")


@st.cache_data
def most_informative_sensors() -> list[str]:
    """Sensors whose raw readings track degradation most closely."""
    train = data_loader.load_training_data()
    usable = features.usable_sensor_columns(train)
    correlations = train[usable].corrwith(train[config.RUL_COLUMN]).abs()
    return correlations.sort_values(ascending=False).index.tolist()


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------


def sidebar_assumptions() -> tuple[float, int]:
    """The two numbers that belong to whoever runs the fleet, not to the data."""
    st.sidebar.header("Your assumptions")

    cost_ratio = st.sidebar.slider(
        "An unplanned failure costs this many planned overhauls",
        min_value=config.COST_RATIO_MIN,
        max_value=config.COST_RATIO_MAX,
        value=config.DEFAULT_COST_RATIO,
        step=config.COST_RATIO_STEP,
        help=(
            "Nothing in the data sets this — it is yours. On this fleet it barely moves "
            "the threshold, because a low threshold already catches every engine. What "
            "it moves is whether monitoring is worth doing at all."
        ),
    )
    lead_time = st.sidebar.slider(
        "Cycles of notice the shop needs to act",
        min_value=config.THRESHOLD_GRID_MIN,
        max_value=60,
        value=config.MINIMUM_LEAD_TIME,
        help=(
            "Time to source the part, book the slot, roster the crew. Drag this to 1 to "
            "see the unconstrained cost optimum — cheapest on paper, impossible in a "
            "hangar."
        ),
    )

    trained = load_trained_model()
    st.sidebar.divider()
    st.sidebar.caption(
        f"Model: gradient boosting on {len(trained.sensor_columns)} sensors  \n"
        f"Typical error: ±{trained.validation_rmse:.1f} cycles  \n"
        f"Median engine life: {trained.nominal_life:.0f} cycles  \n"
        f"One wasted cycle costs "
        f"{config.cost_per_wasted_cycle(trained.nominal_life):.5f} overhauls"
    )
    return cost_ratio, lead_time


def section_thresholds(policy: decision.DecisionPolicy, derived: dict) -> None:
    st.subheader("The thresholds this assumption implies")

    columns = st.columns(4)
    columns[0].metric("Intervene at", f"{policy.act_threshold} cycles")
    columns[1].metric(
        "Monitor from",
        f"{policy.watch_threshold} cycles",
        delta=f"+{round(policy.model_error)} for model error",
        delta_color="off",
    )
    columns[2].metric(
        "Cheapest, ignoring lead time",
        f"{derived['unconstrained_threshold']} cycles",
        delta="optimal and unusable",
        delta_color="off",
    )
    columns[3].metric(
        "Calendar baseline", f"every {derived['calendar_interval']} cycles"
    )

    st.caption(
        "No threshold here was chosen. Every candidate from "
        f"{config.THRESHOLD_GRID_MIN} to {config.THRESHOLD_GRID_MAX} cycles is replayed "
        "across the fleet and priced; the cheapest wins. The calendar baseline gets the "
        "same search, because beating a badly tuned baseline would prove nothing."
    )


def section_backtest(policy: decision.DecisionPolicy, calendar_interval: int) -> None:
    st.subheader("What each policy would have cost")

    fleet, predictions = load_backtest_fleet()
    run_to_failure = decision.replay_run_to_failure(fleet, policy.costs)
    calendar = decision.replay_calendar(fleet, calendar_interval, policy.costs)
    ours = decision.replay_condition_based(
        fleet, predictions, policy.act_threshold, policy.costs
    )

    table = pd.DataFrame(
        [
            _outcome_row("Run to failure", run_to_failure),
            _outcome_row(f"Calendar, every {calendar_interval} cycles", calendar),
            _outcome_row("Condition-based (this project)", ours),
        ]
    )
    st.dataframe(table, width="stretch", hide_index=True)

    left, right = st.columns(2)
    left.metric(
        "Saving versus the calendar baseline", f"{ours.saving_versus(calendar):.1%}"
    )
    right.metric(
        "Saving versus running to failure",
        f"{ours.saving_versus(run_to_failure):.1%}",
    )
    st.caption(
        f"Priced on {fleet[config.UNIT_COLUMN].nunique()} engines that shaped neither "
        "the model nor the threshold. Costs are in units of one planned overhaul."
    )


def section_assumption_sweep(cost_ratio: float, lead_time: int) -> None:
    """The whole curve, including the part where this tool is not worth using."""
    st.subheader("How much your assumption matters")

    sweep = assumption_sweep(lead_time)
    left, right = st.columns(2)

    with left:
        st.markdown("**Saving versus the calendar baseline**")
        st.line_chart(
            sweep[["saving vs calendar"]],
            x_label="an unplanned failure costs this many overhauls",
            y_label="% of the calendar's cost avoided",
        )
    with right:
        st.markdown("**Intervention threshold**")
        st.line_chart(
            sweep[["intervention threshold"]],
            x_label="an unplanned failure costs this many overhauls",
            y_label="cycles of predicted life left",
        )

    breakeven = sweep[sweep["saving vs calendar"] > 0]
    breakeven_ratio = breakeven.index.min() if len(breakeven) else None

    st.markdown(
        f"**Read the left chart first.** At your current assumption of "
        f"**{cost_ratio:.1f}x**, condition-based monitoring saves "
        f"**{sweep.loc[cost_ratio, 'saving vs calendar']:.1f}%**. "
        + (
            f"Below about **{breakeven_ratio:.1f}x** it saves nothing at all — a fixed "
            f"calendar is the better answer, and this tool should say so rather than "
            f"sell itself."
            if breakeven_ratio is not None
            else "There is no ratio in this range where it pays off."
        )
    )
    st.markdown(
        "**The right chart is flatter than expected, and that is a finding.** Every "
        "engine's predicted life eventually drops below about 5 cycles, so a very low "
        "threshold already catches the whole fleet; raising it prevents no further "
        "failures and only scraps good life. On this dataset the cost ratio therefore "
        "barely moves the *threshold* — what it moves is whether monitoring is worth "
        "doing at all. The threshold you see is set by the lead-time slider, not by the "
        "cost slider."
    )


def section_live_fleet(policy: decision.DecisionPolicy) -> pd.DataFrame:
    """The fleet as it stands now, most urgent first."""
    st.subheader("The fleet right now")

    status = current_fleet_status(policy)
    counts = status["Recommendation"].value_counts()
    columns = st.columns(3)
    for column, action in zip(columns, ACTION_ICONS.values()):
        column.metric(action, int(counts.get(action, 0)))

    chosen = st.multiselect(
        "Show", list(ACTION_ICONS.values()), default=list(ACTION_ICONS.values())
    )
    visible = status[status["Recommendation"].isin(chosen)]
    st.dataframe(visible.head(URGENT_ENGINES_SHOWN * 10), width="stretch", hide_index=True)

    st.caption(
        "The NASA test engines: recording stops before they fail, which is exactly the "
        "situation a plant is in. **Actual RUL** is known only because this is a "
        "benchmark — the model never sees it, and neither would an operator."
    )
    return status


def section_engine_detail(policy: decision.DecisionPolicy, status: pd.DataFrame) -> None:
    st.subheader("One engine, in detail")

    engine = st.selectbox(
        "Engine", status["Engine"].tolist(), format_func=lambda unit: f"Engine {unit}"
    )
    fleet, predictions = load_live_fleet()
    history = fleet[fleet[config.UNIT_COLUMN] == engine]
    engine_predictions = predictions.loc[history.index]

    recommendation = policy.recommend(float(engine_predictions.iloc[-1]))
    st.info(recommendation.explain(), icon="🛠️")

    st.markdown("**Predicted remaining life, cycle by cycle**")
    st.line_chart(
        pd.DataFrame(
            {
                "predicted remaining life": engine_predictions.to_numpy(),
                "intervene at or below": policy.act_threshold,
                "monitor at or below": policy.watch_threshold,
            },
            index=history[config.CYCLE_COLUMN].to_numpy(),
        ),
        x_label="cycle",
        y_label="cycles of life left",
    )

    ranked_sensors = most_informative_sensors()
    charted = st.multiselect(
        "Sensors",
        ranked_sensors,
        default=ranked_sensors[:SENSORS_CHARTED_BY_DEFAULT],
        help="Ordered by how strongly each one tracks degradation across the fleet.",
    )
    if charted:
        st.markdown("**Raw sensor readings**")
        st.line_chart(
            history.set_index(config.CYCLE_COLUMN)[charted],
            x_label="cycle",
            y_label="reading",
        )


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def current_fleet_status(policy: decision.DecisionPolicy) -> pd.DataFrame:
    """One row per live engine, at its most recent cycle, most urgent first."""
    fleet, predictions = load_live_fleet()
    latest_cycle = fleet.groupby(config.UNIT_COLUMN)[config.CYCLE_COLUMN].transform("max")
    latest = fleet[fleet[config.CYCLE_COLUMN] == latest_cycle]

    rows = [
        {
            "Engine": int(row[config.UNIT_COLUMN]),
            "Cycles run": int(row[config.CYCLE_COLUMN]),
            "Predicted RUL": round(float(predictions.loc[index]), 1),
            "Recommendation": ACTION_ICONS[
                policy.recommend(float(predictions.loc[index])).action
            ],
            "Actual RUL": int(row[config.RUL_COLUMN]),
        }
        for index, row in latest.iterrows()
    ]
    return pd.DataFrame(rows).sort_values("Predicted RUL").reset_index(drop=True)


def _outcome_row(name: str, outcome: decision.FleetOutcome) -> dict:
    return {
        "Policy": name,
        "Planned overhauls": outcome.interventions,
        "Unplanned failures": outcome.failures,
        "Good cycles scrapped": outcome.wasted_cycles,
        "Total cost": round(outcome.total_cost, 1),
    }


def show_setup_error(problem: Exception) -> None:
    """Anything missing is a setup step, not a crash. Say which one."""
    st.error("This dashboard needs the dataset and a trained model.")
    st.code(str(problem), language="text")
    st.markdown(
        "```powershell\n"
        "# 1. download the C-MAPSS .txt files into data/raw/\n"
        f"#    {config.DATASET_DOWNLOAD_URL}\n"
        "# 2. train the model\n"
        ".\\.venv\\Scripts\\python.exe -m src.model\n"
        "```"
    )


def main() -> None:
    st.set_page_config(
        page_title="Predictive Maintenance Simulator", page_icon="🛠️", layout="wide"
    )
    st.title("Predictive Maintenance Simulator")
    st.markdown(
        "Turns an engine-failure prediction into an **explicit maintenance decision**, "
        "driven by a cost trade-off you set. The model is deliberately ordinary; the "
        "argument is in what happens to it next."
    )

    try:
        cost_ratio, lead_time = sidebar_assumptions()
        derived = derive_policy(cost_ratio, lead_time)
    except (FileNotFoundError, ValueError) as problem:
        show_setup_error(problem)
        return

    trained = load_trained_model()
    policy = decision.DecisionPolicy(
        act_threshold=derived["act_threshold"],
        watch_threshold=derived["watch_threshold"],
        costs=decision.CostModel.from_ratio(cost_ratio, trained.nominal_life),
        model_error=trained.validation_rmse,
    )

    section_thresholds(policy, derived)
    st.divider()
    section_backtest(policy, derived["calendar_interval"])
    st.divider()
    section_assumption_sweep(cost_ratio, lead_time)
    st.divider()
    status = section_live_fleet(policy)
    st.divider()
    section_engine_detail(policy, status)


main()
