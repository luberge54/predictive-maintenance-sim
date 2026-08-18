r"""Load the NASA C-MAPSS turbofan degradation files into validated DataFrames.

The raw files are headerless and space-separated, with irregular spacing and trailing
whitespace on every line. `sep=r"\s+"` absorbs all of it and yields exactly the 26
documented columns — verified against pandas 3.0, not assumed. The column-count check
below is a guard against being handed a file that is not C-MAPSS at all. Every piece of
knowledge about the raw format lives in this module, so the rest of the project only
ever sees a clean, labelled frame with a `rul` column it can trust.

Remaining useful life (RUL) is derived differently for the two splits:

- Training engines are recorded until they fail, so RUL is simply how many cycles are
  left before the last recorded one.
- Test engines are cut off *before* failing. The true RUL at that cut-off point ships in
  a separate file and has to be added back, otherwise every test engine looks like it
  fails the moment the recording stops.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config


def load_training_data() -> pd.DataFrame:
    """Training engines, run to failure, with a `rul` column.

    Raises:
        FileNotFoundError: if the raw file has not been downloaded.
        ValueError: if the file does not have the documented C-MAPSS shape.
    """
    frame = _read_cmapss_file(config.TRAIN_FILE)
    cycles_left = _cycles_until_last_record(frame)
    return frame.assign(**{config.RUL_COLUMN: cycles_left})


def load_test_data() -> pd.DataFrame:
    """Test engines, cut off before failure, with a corrected `rul` column.

    The per-engine RUL remaining at the cut-off point is read from the companion
    `RUL_*.txt` file and added to every row of that engine.

    Raises:
        FileNotFoundError: if either raw file has not been downloaded.
        ValueError: if the two files disagree on which engines exist.
    """
    frame = _read_cmapss_file(config.TEST_FILE)
    rul_at_cutoff = _read_rul_at_cutoff(config.TEST_RUL_FILE)
    _check_every_engine_has_a_rul(frame, rul_at_cutoff)

    cycles_left = _cycles_until_last_record(frame)
    remaining_after_cutoff = frame[config.UNIT_COLUMN].map(rul_at_cutoff)
    return frame.assign(**{config.RUL_COLUMN: cycles_left + remaining_after_cutoff})


def median_engine_life(frame: pd.DataFrame) -> float:
    """Median number of cycles an engine survives — the project's `NOMINAL_LIFE`.

    Feeds `config.cost_per_wasted_cycle`. Read from the data rather than typed in by
    hand, so the cost model stays anchored to the fleet it is describing.
    """
    return float(frame.groupby(config.UNIT_COLUMN)[config.CYCLE_COLUMN].max().median())


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------


def _read_cmapss_file(path: Path) -> pd.DataFrame:
    """Read one headerless C-MAPSS file and label its columns."""
    _require_file(path)
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    frame = _drop_phantom_columns(frame, path)
    frame.columns = config.RAW_COLUMNS
    return frame


def _read_rul_at_cutoff(path: Path) -> pd.Series:
    """Read `RUL_*.txt`: one value per engine, in engine-number order.

    The file carries no engine ids — line 1 is engine 1. That implicit contract is the
    whole reason this is a named function and not an inline call.
    """
    _require_file(path)
    values = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0]
    values.index = pd.RangeIndex(start=1, stop=len(values) + 1, name=config.UNIT_COLUMN)
    return values


def _require_file(path: Path) -> None:
    """Fail with instructions rather than a bare traceback."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw data file: {path}\n"
            f"Download the C-MAPSS archive and unzip the .txt files into "
            f"{config.RAW_DATA_DIR}\n"
            f"Source: {config.DATASET_DOWNLOAD_URL}"
        )


def _drop_phantom_columns(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    r"""Verify the column count, trimming empty padding columns if a parser produced any.

    In practice `sep=r"\s+"` already returns exactly 26 columns. This is the guard for
    the day someone feeds in the wrong file, or a future pandas changes its whitespace
    handling: silently trimming real data would corrupt every number downstream while
    looking like it worked.
    """
    expected = len(config.RAW_COLUMNS)
    actual = frame.shape[1]

    if actual == expected:
        return frame
    if actual < expected:
        raise ValueError(
            f"{path.name} has {actual} columns, expected {expected}. "
            f"This does not look like a C-MAPSS file."
        )

    trailing = frame.iloc[:, expected:]
    if not trailing.isna().all().all():
        raise ValueError(
            f"{path.name} has {actual} columns and the {actual - expected} extra ones "
            f"contain data, not padding. Refusing to guess which to keep."
        )
    return frame.iloc[:, :expected]


def _cycles_until_last_record(frame: pd.DataFrame) -> pd.Series:
    """How many cycles remain before this engine's last recorded cycle."""
    last_cycle = frame.groupby(config.UNIT_COLUMN)[config.CYCLE_COLUMN].transform("max")
    return last_cycle - frame[config.CYCLE_COLUMN]


def _check_every_engine_has_a_rul(frame: pd.DataFrame, rul_at_cutoff: pd.Series) -> None:
    """The two test files must describe the same set of engines."""
    engines_in_data = set(frame[config.UNIT_COLUMN].unique())
    engines_with_rul = set(rul_at_cutoff.index)

    missing = engines_in_data - engines_with_rul
    if missing:
        raise ValueError(
            f"{len(missing)} test engine(s) have no entry in "
            f"{config.TEST_RUL_FILE.name}: {sorted(missing)[:5]}..."
        )
