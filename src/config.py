"""Central configuration for the Predictive Maintenance Simulator.

Every tunable number in the project lives here. Nothing is typed twice, and nothing
important is buried inside a function.

The cost model implemented by these constants is specified in `docs/00-scoping.md`.
If you change a value here, update that document too — it is the contract.
"""

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# C-MAPSS ships four subsets. FD001 is the simplest: one fault mode, one operating
# condition. Start here; the pipeline is written so the subset can be swapped.
DATASET_ID = "FD001"

TRAIN_FILE = RAW_DATA_DIR / f"train_{DATASET_ID}.txt"
TEST_FILE = RAW_DATA_DIR / f"test_{DATASET_ID}.txt"
TEST_RUL_FILE = RAW_DATA_DIR / f"RUL_{DATASET_ID}.txt"

# --------------------------------------------------------------------------------------
# Dataset schema
# --------------------------------------------------------------------------------------

# The raw files are space-separated with no header. This is the documented column order.
UNIT_COLUMN = "unit_number"
CYCLE_COLUMN = "time_in_cycles"
RUL_COLUMN = "rul"

OPERATIONAL_SETTING_COLUMNS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

RAW_COLUMNS = [UNIT_COLUMN, CYCLE_COLUMN] + OPERATIONAL_SETTING_COLUMNS + SENSOR_COLUMNS

# --------------------------------------------------------------------------------------
# Cost model  (docs/00-scoping.md, section 1)
# --------------------------------------------------------------------------------------

# Everything is expressed relative to ONE planned preventive overhaul. The absolute
# currency never matters — only the ratios do.
COST_PREVENTIVE = 1.0

# How much more an unplanned in-service failure costs than a planned overhaul.
# This is THE parameter of the project. It is a slider in the dashboard, never a fixed
# truth: it belongs to whoever operates the fleet.
DEFAULT_COST_RATIO = 5.0
COST_RATIO_MIN = 1.0
COST_RATIO_MAX = 20.0
COST_RATIO_STEP = 0.5

# --------------------------------------------------------------------------------------
# Decision policy  (docs/00-scoping.md, section 2)
# --------------------------------------------------------------------------------------

# Candidate intervention thresholds, in cycles of predicted remaining useful life.
# The optimal one is found by replaying the fleet at each value and keeping the cheapest.
THRESHOLD_GRID_MIN = 1
THRESHOLD_GRID_MAX = 150

# Candidate intervals for the calendar-based baseline, in cycles. It gets the same
# optimisation pass as our own policy — beating a badly tuned baseline proves nothing.
CALENDAR_INTERVAL_GRID_MIN = 10
CALENDAR_INTERVAL_GRID_MAX = 300

# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

# Fixed so that two runs of the pipeline produce the same numbers.
RANDOM_SEED = 42

# Fraction of engines held out to tune the decision thresholds. Split by engine, never
# by row: rows from the same engine are not independent.
VALIDATION_SPLIT = 0.2

# Rolling-window widths (in cycles) used to build the sensor features.
ROLLING_WINDOWS = (5, 20)

# C-MAPSS engines run healthy for a long time before degrading. Capping the training
# target stops the model from wasting capacity on "this engine has 300 cycles left",
# which is a distinction nobody makes a decision on. Standard practice on this dataset.
RUL_CAP = 130


def cost_of_failure(cost_ratio: float = DEFAULT_COST_RATIO) -> float:
    """Cost of one unplanned failure, for a given cost ratio.

    Raises:
        ValueError: if the ratio is outside the range the dashboard allows.
    """
    if not COST_RATIO_MIN <= cost_ratio <= COST_RATIO_MAX:
        raise ValueError(
            f"cost_ratio must be between {COST_RATIO_MIN} and {COST_RATIO_MAX}, "
            f"got {cost_ratio}"
        )
    return COST_PREVENTIVE * cost_ratio


def cost_per_wasted_cycle(nominal_life: float) -> float:
    """Cost of throwing away one cycle of still-usable engine life.

    Derived, never chosen: scrapping a full nominal engine life costs exactly one extra
    preventive overhaul. Without this term the cheapest policy would be "replace every
    engine on cycle 1", which never fails and is obviously absurd.

    Args:
        nominal_life: median observed engine lifetime, read from the training data.

    Raises:
        ValueError: if nominal_life is not strictly positive.
    """
    if nominal_life <= 0:
        raise ValueError(f"nominal_life must be > 0, got {nominal_life}")
    return COST_PREVENTIVE / nominal_life
