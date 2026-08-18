"""Tests for the C-MAPSS loader.

The loader is the only place in the project that touches the raw file format, so what is
worth testing is: does it read the documented shape, does it compute RUL correctly for
both splits, and does it refuse a file it does not understand instead of quietly
returning nonsense.

No real dataset is needed — each test writes a small file in the documented format.
"""

from pathlib import Path

import pandas as pd
import pytest

from src import config, data_loader

# A C-MAPSS row is: unit, cycle, 3 operational settings, then the sensors.
COLUMNS_BEFORE_SENSORS = 5


def write_cmapss_file(path: Path, cycles_per_engine: dict[int, int]) -> None:
    """Write a file in the raw C-MAPSS format, trailing whitespace included."""
    sensor_count = len(config.RAW_COLUMNS) - COLUMNS_BEFORE_SENSORS
    lines = []
    for unit, cycle_count in cycles_per_engine.items():
        for cycle in range(1, cycle_count + 1):
            settings = [f"{0.001 * cycle:.4f}"] * 3
            sensors = [f"{100.0 - cycle:.2f}"] * sensor_count
            lines.append("  ".join([str(unit), str(cycle), *settings, *sensors]) + "  ")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rul_file(path: Path, rul_per_engine: list[int]) -> None:
    """Write a RUL_*.txt file: one value per engine, in engine-number order."""
    body = "\n".join(str(value) for value in rul_per_engine)
    path.write_text(body + "\n", encoding="utf-8")


@pytest.fixture
def raw_files(tmp_path, monkeypatch):
    """Point the loader at a throwaway directory instead of the real data/raw/."""
    paths = {
        "train": tmp_path / "train_TEST.txt",
        "test": tmp_path / "test_TEST.txt",
        "rul": tmp_path / "RUL_TEST.txt",
    }
    monkeypatch.setattr(config, "TRAIN_FILE", paths["train"])
    monkeypatch.setattr(config, "TEST_FILE", paths["test"])
    monkeypatch.setattr(config, "TEST_RUL_FILE", paths["rul"])
    return paths


# --------------------------------------------------------------------------------------
# Reading the format
# --------------------------------------------------------------------------------------


def test_training_data_has_the_documented_columns(raw_files):
    # Arrange
    write_cmapss_file(raw_files["train"], {1: 3, 2: 2})

    # Act
    frame = data_loader.load_training_data()

    # Assert
    assert list(frame.columns) == [*config.RAW_COLUMNS, config.RUL_COLUMN]


def test_trailing_whitespace_does_not_create_phantom_columns(raw_files):
    # Arrange — write_cmapss_file deliberately ends every line with two spaces
    write_cmapss_file(raw_files["train"], {1: 4})

    # Act
    frame = data_loader.load_training_data()

    # Assert
    assert len(frame) == 4
    assert not frame.isna().any().any()


# --------------------------------------------------------------------------------------
# Remaining useful life
# --------------------------------------------------------------------------------------


def test_training_rul_counts_down_to_zero_on_the_last_cycle(raw_files):
    # Arrange — engine 1 runs to failure at cycle 5
    write_cmapss_file(raw_files["train"], {1: 5})

    # Act
    frame = data_loader.load_training_data()

    # Assert
    assert list(frame[config.RUL_COLUMN]) == [4, 3, 2, 1, 0]


def test_each_training_engine_gets_its_own_countdown(raw_files):
    # Arrange — two engines with different lifetimes
    write_cmapss_file(raw_files["train"], {1: 3, 2: 6})

    # Act
    frame = data_loader.load_training_data()
    longest_countdown = frame.groupby(config.UNIT_COLUMN)[config.RUL_COLUMN].max()

    # Assert — the countdown restarts per engine, it is not fleet-wide
    assert longest_countdown[1] == 2
    assert longest_countdown[2] == 5


def test_test_engine_rul_includes_the_life_left_after_the_cut_off(raw_files):
    # Arrange — engine recorded for 5 cycles, still had 10 cycles left when cut off
    write_cmapss_file(raw_files["test"], {1: 5})
    write_rul_file(raw_files["rul"], [10])

    # Act
    frame = data_loader.load_test_data()

    # Assert — never reaches zero: the engine had not failed yet
    assert list(frame[config.RUL_COLUMN]) == [14, 13, 12, 11, 10]


def test_test_rul_file_is_matched_to_engines_by_line_order(raw_files):
    # Arrange — the RUL file carries no engine ids: line 1 describes engine 1
    write_cmapss_file(raw_files["test"], {1: 2, 2: 2})
    write_rul_file(raw_files["rul"], [7, 30])

    # Act
    frame = data_loader.load_test_data()
    rul_at_last_record = frame.groupby(config.UNIT_COLUMN)[config.RUL_COLUMN].min()

    # Assert
    assert rul_at_last_record[1] == 7
    assert rul_at_last_record[2] == 30


# --------------------------------------------------------------------------------------
# Refusing what it does not understand
# --------------------------------------------------------------------------------------


def test_missing_file_error_says_where_to_download_the_data(raw_files):
    # Arrange — nothing is written, so the file does not exist

    # Act / Assert
    with pytest.raises(FileNotFoundError) as error:
        data_loader.load_training_data()

    assert config.DATASET_DOWNLOAD_URL in str(error.value)
    assert str(config.RAW_DATA_DIR) in str(error.value)


def test_file_with_too_few_columns_is_rejected(raw_files):
    # Arrange — a plausible-looking file that is not C-MAPSS
    raw_files["train"].write_text("1 1 2 3\n1 2 3 4\n", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError, match="does not look like a C-MAPSS file"):
        data_loader.load_training_data()


def test_extra_columns_holding_real_data_are_refused():
    # Arrange — one column too many, full of numbers rather than padding
    frame = pd.DataFrame([[1.0] * (len(config.RAW_COLUMNS) + 1)] * 3)

    # Act / Assert — refusing beats guessing which column to drop
    with pytest.raises(ValueError, match="contain data, not padding"):
        data_loader._drop_phantom_columns(frame, Path("fake.txt"))


def test_empty_padding_columns_are_trimmed():
    # Arrange — the documented columns followed by two all-empty ones
    row = [1.0] * len(config.RAW_COLUMNS) + [None, None]
    frame = pd.DataFrame([row] * 3)

    # Act
    trimmed = data_loader._drop_phantom_columns(frame, Path("fake.txt"))

    # Assert
    assert trimmed.shape[1] == len(config.RAW_COLUMNS)


def test_test_engine_without_a_rul_entry_is_refused(raw_files):
    # Arrange — two engines in the data, only one RUL value
    write_cmapss_file(raw_files["test"], {1: 2, 2: 2})
    write_rul_file(raw_files["rul"], [7])

    # Act / Assert
    with pytest.raises(ValueError, match="no entry in"):
        data_loader.load_test_data()


# --------------------------------------------------------------------------------------
# Feeding the cost model
# --------------------------------------------------------------------------------------


def test_median_engine_life_is_read_from_the_fleet(raw_files):
    # Arrange — lifetimes of 10, 20 and 30 cycles, so the median is 20
    write_cmapss_file(raw_files["train"], {1: 10, 2: 20, 3: 30})

    # Act
    frame = data_loader.load_training_data()

    # Assert
    assert data_loader.median_engine_life(frame) == 20.0


def test_median_engine_life_feeds_a_usable_wasted_cycle_cost(raw_files):
    # Arrange
    write_cmapss_file(raw_files["train"], {1: 100, 2: 200, 3: 300})
    frame = data_loader.load_training_data()

    # Act
    nominal_life = data_loader.median_engine_life(frame)
    wasted_cycle_cost = config.cost_per_wasted_cycle(nominal_life)

    # Assert — scrapping a full nominal life costs exactly one preventive overhaul
    assert wasted_cycle_cost * nominal_life == pytest.approx(config.COST_PREVENTIVE)
