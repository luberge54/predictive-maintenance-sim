"""Turn raw sensor readings into features a model can learn from.

A single sensor reading is noisy; the *trend* is what carries the degradation signal
(`docs/01-data.md`, section 4). So every usable sensor contributes, per cycle:

- its raw reading
- a rolling mean over each configured window — the smoothed level
- a rolling standard deviation over each window — the instability
- its drift from the engine's very first reading — how far it has moved since new

Two rules this module exists to enforce:

**Windows look backwards only.** A feature at cycle *t* is built from cycles <= *t*. A
centred or forward-looking window would let the model read the future, score beautifully
in validation, and be useless in service.

**The usable-sensor list is decided once, on training data, and passed in.** Detecting it
separately on each split would hand different columns to fit and predict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


def usable_sensor_columns(train: pd.DataFrame) -> list[str]:
    """Sensors that actually move on the training fleet.

    Six of the 21 hold a single value on FD001 and carry no information. They are
    detected rather than hard-coded, so this works unchanged on the other subsets, where
    the dead set differs.

    Deadness is decided by counting distinct values, not by testing the standard
    deviation against zero. Two of the six constant sensors have a standard deviation of
    around 1e-15 rather than an exact zero — floating-point noise from summing identical
    values — so a `std > 0` test lets them through, and whether it does so depends on
    which engines happen to be in the split. Counting distinct values has no such
    tolerance to get wrong.
    """
    distinct_values = train[config.SENSOR_COLUMNS].nunique()
    return distinct_values[distinct_values > 1].index.tolist()


def build_features(
    frame: pd.DataFrame,
    sensor_columns: list[str],
    windows: tuple[int, ...] = config.ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Build the feature matrix for every row of `frame`.

    Args:
        frame: raw rows, as returned by the data loader.
        sensor_columns: the sensors to use — always the list from `usable_sensor_columns`
            computed on the *training* split.
        windows: rolling window widths, in cycles.

    Returns:
        One row of features per input row, index-aligned with `frame`.

    Raises:
        ValueError: if `frame` is not ordered by engine and cycle, which would make the
            rolling windows silently mix engines together.
    """
    require_chronological_order(frame)

    per_engine = frame.groupby(config.UNIT_COLUMN)[sensor_columns]
    parts = [
        # How long this engine has been running is itself a strong predictor.
        frame[[config.CYCLE_COLUMN]],
        frame[sensor_columns],
        _drift_since_new(frame, per_engine, sensor_columns),
    ]
    for window in windows:
        parts.append(_rolling_mean(per_engine, window))
        parts.append(_rolling_deviation(per_engine, window))

    return pd.concat(parts, axis=1)


def cap_target(rul: pd.Series, cap: int = config.RUL_CAP) -> pd.Series:
    """Flatten remaining useful life above `cap`.

    Engines sit healthy for a long time before degrading. "300 cycles left" versus "280"
    is a distinction nobody makes a maintenance decision on, and letting the model chase
    it wastes capacity on the part of the curve that carries no signal. Standard practice
    on this dataset.
    """
    return rul.clip(upper=cap)


@dataclass(frozen=True)
class EngineSplit:
    """Three disjoint sets of engines, each with exactly one job.

    Keeping them separate is what makes the headline saving believable. A threshold
    tuned on the same engines it is then scored against will always look good.
    """

    fit: pd.DataFrame
    tuning: pd.DataFrame
    evaluation: pd.DataFrame


def split_by_engine(
    frame: pd.DataFrame,
    tuning_fraction: float = config.TUNING_SPLIT,
    evaluation_fraction: float = config.EVALUATION_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> EngineSplit:
    """Divide the fleet three ways **by engine**, never by row.

    Consecutive cycles of one engine are near-identical. Splitting by row would put cycle
    99 in one set and cycle 100 in another, and every score would measure memorisation
    rather than prediction.

    Raises:
        ValueError: if the fractions would leave any of the three sets empty.
    """
    engines = np.sort(frame[config.UNIT_COLUMN].unique())
    tuning_size = round(len(engines) * tuning_fraction)
    evaluation_size = round(len(engines) * evaluation_fraction)
    fit_size = len(engines) - tuning_size - evaluation_size

    if min(fit_size, tuning_size, evaluation_size) < 1:
        raise ValueError(
            f"Splitting {len(engines)} engines at tuning={tuning_fraction} and "
            f"evaluation={evaluation_fraction} gives sizes "
            f"{fit_size}/{tuning_size}/{evaluation_size}; each set needs at least one."
        )

    shuffled = np.random.default_rng(seed).permutation(engines)
    return EngineSplit(
        fit=_engines(frame, shuffled[tuning_size + evaluation_size :]),
        tuning=_engines(frame, shuffled[:tuning_size]),
        evaluation=_engines(frame, shuffled[tuning_size : tuning_size + evaluation_size]),
    )


def require_chronological_order(frame: pd.DataFrame) -> None:
    """Fail unless rows are grouped by engine and ordered by cycle.

    Rolling windows and policy replays both walk each engine forwards in time. Out-of-
    order rows silently produce plausible nonsense in either case.

    Raises:
        ValueError: if the rows are not sorted by engine then cycle.
    """
    keys = [config.UNIT_COLUMN, config.CYCLE_COLUMN]
    if not frame[keys].equals(frame[keys].sort_values(keys)):
        raise ValueError(
            f"Rows must be sorted by {keys}, otherwise per-engine sequences are mixed "
            f"together and every result derived from them is meaningless."
        )


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------


def _engines(frame: pd.DataFrame, engine_ids) -> pd.DataFrame:
    """The rows belonging to `engine_ids`, as an independent copy."""
    return frame[frame[config.UNIT_COLUMN].isin(set(engine_ids.tolist()))].copy()


def _rolling_mean(per_engine, window: int) -> pd.DataFrame:
    """Smoothed sensor level over the last `window` cycles of the same engine."""
    smoothed = per_engine.transform(lambda s: s.rolling(window, min_periods=1).mean())
    return smoothed.add_suffix(f"_mean_{window}")


def _rolling_deviation(per_engine, window: int) -> pd.DataFrame:
    """Sensor instability over the last `window` cycles of the same engine.

    The first cycle of an engine has no deviation to speak of, so the undefined value is
    reported as zero rather than left as a gap the model would have to guess at.
    """
    deviation = per_engine.transform(lambda s: s.rolling(window, min_periods=1).std())
    return deviation.fillna(0.0).add_suffix(f"_std_{window}")


def _drift_since_new(
    frame: pd.DataFrame, per_engine, sensor_columns: list[str]
) -> pd.DataFrame:
    """How far each sensor has moved from this engine's own first reading.

    Engines leave the factory with different amounts of initial wear, which NASA
    documents as normal variation. Measuring each engine against itself removes that
    offset, so the model sees degradation rather than manufacturing spread.
    """
    baseline = per_engine.transform("first")
    return (frame[sensor_columns] - baseline).add_suffix("_drift")
