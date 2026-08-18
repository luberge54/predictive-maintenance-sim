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


def test_a_constant_sensor_with_floating_point_noise_is_still_excluded():
    """Regression: two real C-MAPSS sensors hold one value but report std ~1e-15.

    Summing identical floats does not always give an exact zero variance. A `std > 0`
    test therefore kept them, and whether it did depended on which engines were in the
    split — the model silently received a different number of columns per run.
    """
    # Arrange — a sensor whose only value is large enough to lose precision when summed
    fleet = make_fleet({1: 40, 2: 40})
    noisy_constant = config.SENSOR_COLUMNS[3]
    fleet[noisy_constant] = 1e16

    # Act
    usable = features.usable_sensor_columns(fleet)

    # Assert
    assert noisy_constant not in usable


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


def test_no_engine_appears_in_two_sets():
    # Arrange
    fleet = make_fleet({unit: 10 for unit in range(1, 11)})

    # Act
    split = features.split_by_engine(fleet)

    # Assert — the whole point: consecutive cycles of one engine are near-identical, so
    # an engine seen twice turns a score into a measure of memorisation
    fit = set(split.fit[config.UNIT_COLUMN])
    tuning = set(split.tuning[config.UNIT_COLUMN])
    evaluation = set(split.evaluation[config.UNIT_COLUMN])
    assert fit.isdisjoint(tuning)
    assert fit.isdisjoint(evaluation)
    assert tuning.isdisjoint(evaluation)


def test_split_keeps_every_engine():
    # Arrange
    fleet = make_fleet({unit: 10 for unit in range(1, 11)})

    # Act
    split = features.split_by_engine(fleet)

    # Assert — no engine may be dropped on the floor between the three sets
    total = len(split.fit) + len(split.tuning) + len(split.evaluation)
    assert total == len(fleet)


def test_evaluation_set_is_reserved_and_never_the_largest():
    # Arrange
    fleet = make_fleet({unit: 10 for unit in range(1, 11)})

    # Act
    split = features.split_by_engine(fleet)

    # Assert — most engines must go to fitting; the reserved sets are held back
    assert split.fit[config.UNIT_COLUMN].nunique() == 6
    assert split.tuning[config.UNIT_COLUMN].nunique() == 2
    assert split.evaluation[config.UNIT_COLUMN].nunique() == 2


def test_split_is_reproducible_for_a_given_seed():
    # Arrange
    fleet = make_fleet({unit: 10 for unit in range(1, 11)})

    # Act
    first = features.split_by_engine(fleet, seed=config.RANDOM_SEED).evaluation
    second = features.split_by_engine(fleet, seed=config.RANDOM_SEED).evaluation

    # Assert
    assert set(first[config.UNIT_COLUMN]) == set(second[config.UNIT_COLUMN])


def test_split_refuses_fractions_that_empty_a_set():
    # Arrange
    fleet = make_fleet({unit: 10 for unit in range(1, 11)})

    # Act / Assert
    with pytest.raises(ValueError, match="each set needs at least one"):
        features.split_by_engine(fleet, tuning_fraction=0.0)
