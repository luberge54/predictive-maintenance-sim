"""Tests for training, scoring, and persisting the RUL model.

These check the contract around the estimator, not its accuracy — how well gradient
boosting fits turbofan data is scikit-learn's business, and asserting a specific RMSE on
synthetic data would only test the fixture. What matters here is that the artefact
carries what the decision layer needs, that predictions line up with the rows they
describe, and that a saved model comes back identical.

`score_official_test_set` is not covered here: it reads the real NASA files, and is
exercised end to end by `python -m src.model`.
"""

import pytest

from conftest import make_fleet
from src import config, features, model

FLEET = {unit: 30 + unit for unit in range(1, 11)}


@pytest.fixture
def trained():
    """A model fitted on a small synthetic fleet."""
    fleet = make_fleet(FLEET, constant_sensors=("sensor_1",))
    fit_frame, validation_frame = features.split_by_engine(fleet)
    return model.train(fit_frame, validation_frame), validation_frame


def test_trained_model_records_the_sensors_it_used(trained):
    # Arrange
    fitted, _ = trained

    # Act / Assert — prediction must never guess which columns training used
    assert "sensor_1" not in fitted.sensor_columns
    assert len(fitted.sensor_columns) == len(config.SENSOR_COLUMNS) - 1


def test_trained_model_records_what_the_decision_layer_needs(trained):
    # Arrange
    fitted, _ = trained

    # Act / Assert — RMSE sets the width of the "monitor closely" band, nominal life
    # sets the cost of a wasted cycle. A model without them cannot drive a decision.
    assert fitted.validation_rmse > 0
    assert fitted.nominal_life > 0


def test_predict_returns_one_value_per_row(trained):
    # Arrange
    fitted, validation_frame = trained

    # Act
    predictions = fitted.predict(validation_frame)

    # Assert — index alignment is what lets a prediction be traced back to its engine
    assert len(predictions) == len(validation_frame)
    assert predictions.index.equals(validation_frame.index)


def test_score_reports_the_critical_band_separately(trained):
    # Arrange
    fitted, validation_frame = trained

    # Act
    scores = model.score(fitted, validation_frame)

    # Assert — a model accurate on average but vague near failure is useless here
    assert "rmse" in scores
    assert "rmse_near_failure" in scores
    assert scores["rows"] == len(validation_frame)


def test_saved_model_round_trips(tmp_path, trained):
    # Arrange
    fitted, validation_frame = trained
    path = tmp_path / "model.joblib"

    # Act
    model.save(fitted, path)
    reloaded = model.load(path)

    # Assert
    assert reloaded.sensor_columns == fitted.sensor_columns
    assert reloaded.validation_rmse == fitted.validation_rmse
    assert reloaded.predict(validation_frame).equals(fitted.predict(validation_frame))


def test_saving_creates_the_directory_if_missing(tmp_path, trained):
    # Arrange
    fitted, _ = trained
    path = tmp_path / "does" / "not" / "exist" / "model.joblib"

    # Act
    saved_to = model.save(fitted, path)

    # Assert
    assert saved_to.exists()


def test_loading_a_missing_model_says_how_to_train_one(tmp_path):
    # Arrange
    path = tmp_path / "nothing_here.joblib"

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="python -m src.model"):
        model.load(path)
