"""Train and evaluate the remaining-useful-life model.

Run with:  python -m src.model

The model is deliberately ordinary — a gradient-boosted regressor on rolling sensor
features, no tuning beyond the library defaults. It exists to feed the decision layer,
and time spent squeezing another point of RMSE out of it buys nothing for the question
this project actually answers (`docs/00-scoping.md`).

What the artefact carries alongside the estimator matters more than the estimator:

- **the sensor list** it was fitted on, so prediction cannot silently use different
  columns from training
- **the validation RMSE**, measured on the tuning engines, which the decision layer
  turns into the width of the "monitor closely" band
- **the fleet's nominal life**, which sets the cost of a wasted cycle

Everything here is reproducible: same seed, same split, same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src import config, data_loader, features


@dataclass(frozen=True)
class TrainedModel:
    """A fitted estimator plus everything the decision layer needs to use it safely."""

    estimator: HistGradientBoostingRegressor
    sensor_columns: list[str]
    validation_rmse: float
    nominal_life: float

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Predicted remaining useful life, in cycles, for every row of `frame`."""
        matrix = features.build_features(frame, self.sensor_columns)
        return pd.Series(self.estimator.predict(matrix), index=frame.index)


def train(fit_frame: pd.DataFrame, tuning_frame: pd.DataFrame) -> TrainedModel:
    """Fit on `fit_frame` and measure honest error on the held-out tuning engines.

    Both frames come from `features.split_by_engine`, which guarantees no engine appears
    in more than one set. The third set, `evaluation`, is not touched here: it exists so
    the cost saving can be reported on engines that shaped neither the model nor the
    threshold.
    """
    sensor_columns = features.usable_sensor_columns(fit_frame)

    estimator = HistGradientBoostingRegressor(random_state=config.RANDOM_SEED)
    estimator.fit(
        features.build_features(fit_frame, sensor_columns),
        features.cap_target(fit_frame[config.RUL_COLUMN]),
    )

    predictions = estimator.predict(features.build_features(tuning_frame, sensor_columns))
    truth = features.cap_target(tuning_frame[config.RUL_COLUMN])

    return TrainedModel(
        estimator=estimator,
        sensor_columns=sensor_columns,
        validation_rmse=float(root_mean_squared_error(truth, predictions)),
        nominal_life=data_loader.median_engine_life(fit_frame),
    )


def score(model: TrainedModel, frame: pd.DataFrame) -> dict[str, float]:
    """Error metrics for `frame`, overall and inside the decision-critical band.

    A model that is accurate on average but vague near failure is useless here, so the
    two are never reported as one number.
    """
    predictions = model.predict(frame)
    truth = features.cap_target(frame[config.RUL_COLUMN])
    near_failure = truth <= config.DECISION_CRITICAL_RUL

    return {
        "rmse": float(root_mean_squared_error(truth, predictions)),
        "mae": float(mean_absolute_error(truth, predictions)),
        "rmse_near_failure": float(
            root_mean_squared_error(truth[near_failure], predictions[near_failure])
        ),
        "rows": float(len(frame)),
    }


def score_official_test_set(model: TrainedModel) -> dict[str, float]:
    """Error on the held-out NASA test engines, at their cut-off cycle only.

    This is the protocol the C-MAPSS literature reports, so it is the number that can be
    compared with published results. It is one prediction per engine, not per cycle.
    """
    test = data_loader.load_test_data()
    last_cycle = test.groupby(config.UNIT_COLUMN)[config.CYCLE_COLUMN].transform("max")
    at_cutoff = test[test[config.CYCLE_COLUMN] == last_cycle]

    predictions = model.predict(test).loc[at_cutoff.index]
    truth = features.cap_target(at_cutoff[config.RUL_COLUMN])

    return {
        "rmse": float(root_mean_squared_error(truth, predictions)),
        "mae": float(mean_absolute_error(truth, predictions)),
        "engines": float(len(at_cutoff)),
    }


def save(model: TrainedModel, path: Path | None = None) -> Path:
    """Write the trained artefact to disk, creating the directory if needed.

    `path` defaults to `config.MODEL_FILE`, resolved when the function runs rather than
    when the module is imported — a default argument would freeze the path at import time
    and silently ignore any later change to the configuration.

    Stored as a plain dictionary of library types, never as a pickled `TrainedModel`.
    Pickling a custom class records the module it was defined in, and a model trained by
    `python -m src.model` records `__main__` — reloadable only from that same entry
    point, and nowhere else. A dictionary carries no such dependency.
    """
    path = path or config.MODEL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": model.estimator,
            "sensor_columns": list(model.sensor_columns),
            "validation_rmse": model.validation_rmse,
            "nominal_life": model.nominal_life,
        },
        path,
    )
    return path


def load(path: Path | None = None) -> TrainedModel:
    """Read a trained artefact back.

    `path` defaults to `config.MODEL_FILE`, resolved at call time for the same reason
    `save` does.

    Raises:
        FileNotFoundError: if the model has not been trained yet.
        ValueError: if the file was written by an older artefact format.
    """
    path = path or config.MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}\nTrain one with:  python -m src.model"
        )

    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path} is in an old artefact format.\n"
            f"Retrain with:  python -m src.model"
        )
    try:
        return TrainedModel(**payload)
    except TypeError as mismatch:
        raise ValueError(
            f"{path} does not carry the fields this version expects "
            f"({mismatch}).\nRetrain with:  python -m src.model"
        ) from mismatch


def main() -> None:
    """Train, report, and save."""
    split = features.split_by_engine(data_loader.load_training_data())

    print(
        f"Fitting on {split.fit[config.UNIT_COLUMN].nunique()} engines, "
        f"measuring error on {split.tuning[config.UNIT_COLUMN].nunique()}, "
        f"holding {split.evaluation[config.UNIT_COLUMN].nunique()} back for step 3."
    )
    model = train(split.fit, split.tuning)

    _print_scores("Held-out tuning engines (every cycle)", score(model, split.tuning))
    _print_scores("NASA test engines (at cut-off only)", score_official_test_set(model))

    print(f"\n  usable sensors     : {len(model.sensor_columns)}")
    print(f"  nominal engine life: {model.nominal_life:.0f} cycles")
    print(f"  saved to           : {save(model)}")


def _print_scores(title: str, scores: dict[str, float]) -> None:
    print(f"\n{title}")
    for name, value in scores.items():
        print(f"  {name:20s} {value:8.2f}")


if __name__ == "__main__":
    main()
