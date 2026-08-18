"""Tests for feature building and splitting.

Three of these guard against mistakes that do not announce themselves: a rolling window
that reads the future, a window that bleeds across engines, and a split that puts the
same engine on both sides. All three produce a model that scores well and fails in
service, so each gets an explicit test rather than a code comment.
"""

import pandas as pd
import pytest

from conftest import make_fleet
from src import config, features

DEAD_SENSORS = ("sensor_1", "sensor_10")


# --------------------------------------------------------------------------------------
# Choosing sensors
# --------------------------------------------------------------------------------------


def test_constant_sensors_are_excluded():
    # Arrange
    fleet = make_fleet({1: 20, 2: 20}, constant_sensors=DEAD_SENSORS)

    # Act
    usable = features.usable_sensor_columns(fleet)

    # Assert
    assert set(DEAD_SENSORS).isdisjoint(usable)
    assert len(usable) == len(config.SENSOR_COLUMNS) - len(DEAD_SENSORS)


# --------------------------------------------------------------------------------------
# Rolling windows
# --------------------------------------------------------------------------------------


def test_rolling_features_only_look_backwards():
    # Arrange — the same engine, once complete and once truncated early
    fleet = make_fleet({1: 40})
    usable = features.usable_sensor_columns(fleet)

    # Act
    from_full_history = features.build_features(fleet, usable).head(10)
    from_truncated_history = features.build_features(fleet.head(10), usable)

    # Assert — cycle 10's features cannot depend on cycles that have not happened yet
    assert from_full_history.equals(from_truncated_history)


def test_rolling_windows_do_not_leak_between_engines():
    # Arrange — engine 1 reads 100.0 throughout, engine 2 reads 0.0 throughout
    hot = make_fleet({1: 10}, sensor_value=100.0)
    cold = make_fleet({2: 10}, sensor_value=0.0)
    fleet = pd.concat([hot, cold], ignore_index=True)
    usable = [config.SENSOR_COLUMNS[0]]

    # Act
    built = features.build_features(fleet, usable)
    first_row_of_cold_engine = built[fleet[config.UNIT_COLUMN] == 2].iloc[0]

    # Assert — engine 2 starts from nothing, not from engine 1's readings
    for window in config.ROLLING_WINDOWS:
        assert first_row_of_cold_engine[f"{usable[0]}_mean_{window}"] == 0.0


def test_unsorted_rows_are_refused():
    # Arrange — cycles shuffled, which would silently scramble every rolling window
    fleet = make_fleet({1: 20}).sample(frac=1.0, random_state=0)
    usable = features.usable_sensor_columns(fleet)

    # Act / Assert
    with pytest.raises(ValueError, match="must be sorted"):
        features.build_features(fleet, usable)


def test_every_feature_is_defined():
    # Arrange — a fleet shorter than the widest rolling window
    fleet = make_fleet({1: 3, 2: 3})
    usable = features.usable_sensor_columns(fleet)

    # Act
    built = features.build_features(fleet, usable)

    # Assert — partial windows must produce numbers, not gaps
    assert not built.isna().any().any()


def test_drift_is_zero_on_the_first_cycle():
    # Arrange
    fleet = make_fleet({1: 15})
    usable = features.usable_sensor_columns(fleet)

    # Act
    built = features.build_features(fleet, usable)
    drift_columns = [name for name in built.columns if name.endswith("_drift")]

    # Assert — an engine has not drifted from itself before it has run
    assert (built.iloc[0][drift_columns] == 0.0).all()


# --------------------------------------------------------------------------------------
# The training target
# --------------------------------------------------------------------------------------


def test_cap_target_flattens_long_horizons():
    # Arrange
    fleet = make_fleet({1: config.RUL_CAP + 50})

    # Act
    capped = features.cap_target(fleet[config.RUL_COLUMN])

    # Assert
    assert capped.max() == config.RUL_CAP
    assert capped.min() == 0


def test_cap_target_leaves_the_decision_zone_untouched():
    # Arrange — every engine already fails well inside the cap
    fleet = make_fleet({1: 30})

    # Act
    capped = features.cap_target(fleet[config.RUL_COLUMN])

    # Assert — capping must not distort the range where decisions are made
    assert capped.equals(fleet[config.RUL_COLUMN])


# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------


def test_split_puts_no_engine_on_both_sides():
    # Arrange
    fleet = make_fleet({unit: 10 for unit in range(1, 11)})

    # Act
    fit_frame, validation_frame = features.split_by_engine(fleet)

    # Assert — the whole point: consecutive cycles of one engine are near-identical
    fit_engines = set(fit_frame[config.UNIT_COLUMN])
    validation_engines = set(validation_frame[config.UNIT_COLUMN])
    assert fit_engines.isdisjoint(validation_engines)
    assert len(fit_engines) + len(validation_engines) == 10


def test_split_is_reproducible_for_a_given_seed():
    # Arrange
    fleet = make_fleet({unit: 10 for unit in range(1, 11)})

    # Act
    first = features.split_by_engine(fleet, seed=config.RANDOM_SEED)[1]
    second = features.split_by_engine(fleet, seed=config.RANDOM_SEED)[1]

    # Assert
    assert set(first[config.UNIT_COLUMN]) == set(second[config.UNIT_COLUMN])


def test_split_refuses_a_fraction_that_empties_a_side():
    # Arrange
    fleet = make_fleet({unit: 10 for unit in range(1, 11)})

    # Act / Assert
    with pytest.raises(ValueError, match="at least one on each side"):
        features.split_by_engine(fleet, validation_fraction=0.0)
