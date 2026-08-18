"""Print the dataset facts that `docs/01-data.md` is built on.

Run with:  python -m src.explore

Every number in that document comes from here, so a reader can re-derive the claims
instead of trusting them. Read-only: this module never writes a file.
"""

from __future__ import annotations

import pandas as pd

from src import config, data_loader

SECTION_WIDTH = 78
TOP_SENSORS_SHOWN = 6


def main() -> None:
    """Print the full exploration report."""
    train = data_loader.load_training_data()
    test = data_loader.load_test_data()

    report_volume(train, test)
    report_engine_lifetimes(train)
    report_rul_sanity(train, test)
    dead_sensors = report_dead_sensors(train)
    report_sensor_correlations(train, dead_sensors)
    report_operating_conditions(train)


def report_volume(train: pd.DataFrame, test: pd.DataFrame) -> None:
    _heading("1. Volume")
    for name, frame in (("training", train), ("test", test)):
        engines = frame[config.UNIT_COLUMN].nunique()
        print(f"  {name:9s} {len(frame):>7,} rows | {engines:>3} engines")


def report_engine_lifetimes(train: pd.DataFrame) -> None:
    _heading("2. Engine lifetimes (training)")
    lifetimes = train.groupby(config.UNIT_COLUMN)[config.CYCLE_COLUMN].max()
    for label, value in [
        ("shortest", lifetimes.min()),
        ("25th percentile", lifetimes.quantile(0.25)),
        ("median", lifetimes.median()),
        ("mean", lifetimes.mean()),
        ("75th percentile", lifetimes.quantile(0.75)),
        ("longest", lifetimes.max()),
    ]:
        print(f"  {label:16s} {value:>7.1f} cycles")

    nominal_life = data_loader.median_engine_life(train)
    wasted_cycle_cost = config.cost_per_wasted_cycle(nominal_life)
    print(f"\n  NOMINAL_LIFE      = {nominal_life:.0f} cycles")
    print(f"  cost per wasted cycle = {wasted_cycle_cost:.6f} preventive overhauls")
    print(f"  longest / shortest    = {lifetimes.max() / lifetimes.min():.1f}x spread")


def report_rul_sanity(train: pd.DataFrame, test: pd.DataFrame) -> None:
    _heading("3. RUL sanity checks")
    train_min = train[config.RUL_COLUMN].min()
    rul_at_cutoff = test.groupby(config.UNIT_COLUMN)[config.RUL_COLUMN].min()

    print(f"  training RUL reaches zero  : {train_min} (expected 0, engines run to failure)")
    print(
        f"  test RUL at last record    : min {rul_at_cutoff.min()}, "
        f"max {rul_at_cutoff.max()} (expected > 0, cut off before failure)"
    )


def report_dead_sensors(train: pd.DataFrame) -> list[str]:
    """Print and return the sensors that never move — they carry no information."""
    _heading("4. Sensors with zero variance")
    spread = train[config.SENSOR_COLUMNS].std()
    dead = spread[spread == 0].index.tolist()

    print(f"  {len(dead)} constant sensors, to be dropped: {dead}")
    print(f"  {len(config.SENSOR_COLUMNS) - len(dead)} usable sensors")
    return dead


def report_sensor_correlations(train: pd.DataFrame, dead_sensors: list[str]) -> None:
    _heading("5. Sensors most tied to degradation")
    usable = [name for name in config.SENSOR_COLUMNS if name not in dead_sensors]
    correlations = (
        train[usable + [config.RUL_COLUMN]].corr()[config.RUL_COLUMN].drop(config.RUL_COLUMN)
    )
    ranked = correlations.reindex(correlations.abs().sort_values(ascending=False).index)

    for name, value in ranked.head(TOP_SENSORS_SHOWN).items():
        trend = "rises" if value < 0 else "falls"
        print(f"  {name:12s} {value:+.3f}   ({trend} as the engine wears)")
    print(f"  weakest of the {len(ranked)}: {ranked.abs().min():.3f}")


def report_operating_conditions(train: pd.DataFrame) -> None:
    _heading("6. Operating conditions")
    for name in config.OPERATIONAL_SETTING_COLUMNS:
        column = train[name]
        verdict = "constant" if column.std() == 0 else "varies"
        print(f"  {name:14s} {verdict:8s} min {column.min():>8.4f}  max {column.max():>8.4f}")


def _heading(title: str) -> None:
    print()
    print("=" * SECTION_WIDTH)
    print(title)
    print("=" * SECTION_WIDTH)


if __name__ == "__main__":
    main()
