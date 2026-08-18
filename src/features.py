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

import numpy as np
import pandas as pd

from src import config


def usable_sensor_columns(train: pd.DataFrame) -> list[str]:
    """Sensors that actually move on the training fleet.

    Four of the 21 are constant on FD001 and carry no information. They are detected
    rather than hard-coded, so this works unchanged on the other subsets, where the dead
    set differs.
    """
    spread = train[config.SENSOR_COLUMNS].std()
    return spread[spread > 0].index.tolist()


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
    _require_chronological_order(frame)

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


def split_by_engine(
    frame: pd.DataFrame,
    validation_fraction: float = config.VALIDATION_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into training and validation sets **by engine**, never by row.

    Consecutive cycles of one engine are near-identical. Splitting by row would put cycle
    99 in training and cycle 100 in validation, and the validation score would measure
    memorisation rather than prediction.

    Raises:
        ValueError: if the fraction would leave either side without an engine.
    """
    engines = np.sort(frame[config.UNIT_COLUMN].unique())
    validation_size = round(len(engines) * validation_fraction)
    if not 0 < validation_size < len(engines):
        raise ValueError(
            f"validation_fraction={validation_fraction} yields {validation_size} "
            f"validation engines out of {len(engines)}; need at least one on each side."
        )

    shuffled = np.random.default_rng(seed).permutation(engines)
    validation_engines = set(shuffled[:validation_size].tolist())

    is_validation = frame[config.UNIT_COLUMN].isin(validation_engines)
    return frame[~is_validation].copy(), frame[is_validation].copy()


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------


def _require_chronological_order(frame: pd.DataFrame) -> None:
    """Rolling windows are only meaningful on rows grouped by engine, in cycle order."""
    keys = [config.UNIT_COLUMN, config.CYCLE_COLUMN]
    if not frame[keys].equals(frame[keys].sort_values(keys)):
        raise ValueError(
            f"Rows must be sorted by {keys} before building features, otherwise the "
            f"rolling windows mix engines and cycles together."
        )


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
