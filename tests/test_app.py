"""Smoke tests for the Streamlit dashboard.

`AppTest` runs the real page in-process and surfaces any exception it raises, which is
the only way to catch a dashboard that imports cleanly and then dies on render.

These need the dataset and a trained model, neither of which is committed, so they skip
rather than fail on a fresh clone. Everything they cover is a claim the page makes to a
reader: that it renders, that the sliders actually change the answer, and that a missing
model produces instructions instead of a traceback.
"""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config

APP = str(config.PROJECT_ROOT / "app.py")
RENDER_TIMEOUT_SECONDS = 300

needs_trained_model = pytest.mark.skipif(
    not config.MODEL_FILE.exists() or not config.TRAIN_FILE.exists(),
    reason="needs data/raw/ and a model from `python -m src.model`",
)


def saving_shown(app) -> str:
    """The headline metric, read back off the rendered page."""
    return next(m.value for m in app.metric if "calendar baseline" in m.label)


@pytest.fixture
def dashboard():
    return AppTest.from_file(APP, default_timeout=RENDER_TIMEOUT_SECONDS).run()


@needs_trained_model
def test_the_page_renders_without_raising(dashboard):
    # Assert
    assert not dashboard.exception
    assert dashboard.title[0].value == "Predictive Maintenance Simulator"


@needs_trained_model
def test_the_page_shows_every_section(dashboard):
    # Act
    headings = [section.value for section in dashboard.subheader]

    # Assert — a section that silently stops rendering would otherwise go unnoticed
    assert len(headings) == 5


@needs_trained_model
def test_raising_the_cost_of_failure_raises_the_saving(dashboard):
    # Act
    dashboard.slider[0].set_value(2.0).run()
    cheap_failures = saving_shown(dashboard)
    dashboard.slider[0].set_value(20.0).run()
    expensive_failures = saving_shown(dashboard)

    # Assert — the more a failure costs, the more avoiding one is worth
    assert _percentage(expensive_failures) > _percentage(cheap_failures)


@needs_trained_model
def test_demanding_more_notice_moves_the_intervention_threshold(dashboard):
    # Act
    dashboard.slider[1].set_value(10).run()
    with_little_notice = dashboard.metric[0].value
    dashboard.slider[1].set_value(45).run()
    with_more_notice = dashboard.metric[0].value

    # Assert — this is the slider that actually moves the threshold on this dataset
    assert with_little_notice == "10 cycles"
    assert with_more_notice == "45 cycles"


@needs_trained_model
def test_the_fleet_table_is_ordered_most_urgent_first(dashboard):
    # Act — the second table on the page is the live fleet
    fleet = dashboard.dataframe[1].value

    # Assert
    assert fleet["Predicted RUL"].is_monotonic_increasing


def test_a_missing_model_produces_instructions_not_a_traceback(tmp_path, monkeypatch):
    # Arrange — point the app at a model file that does not exist. The caches have to go
    # too: st.cache_resource would otherwise hand back the model a previous test loaded.
    st.cache_data.clear()
    st.cache_resource.clear()
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "absent.joblib")

    # Act
    app = AppTest.from_file(APP, default_timeout=RENDER_TIMEOUT_SECONDS).run()

    # Assert — a reader who has not trained yet must be told what to run
    assert not app.exception
    assert any("needs the dataset and a trained model" in e.value for e in app.error)


def _percentage(rendered: str) -> float:
    """Read a value back off the page, e.g. '26.7%' -> 26.7."""
    return float(rendered.rstrip("%"))
