"""Shared test helpers.

Building small synthetic fleets in memory keeps every test independent of the real
dataset, which is not committed and may not be downloaded.
"""

from __future__ import annotations

import pandas as pd

from src import config


def make_fleet(
    cycles_per_engine: dict[int, int],
    constant_sensors: tuple[str, ...] = (),
    sensor_value: float | None = None,
) -> pd.DataFrame:
    """Build a loader-shaped frame: raw columns plus `rul`, sorted by engine and cycle.

    Args:
        cycles_per_engine: how many cycles each engine ran before failing.
        constant_sensors: sensors pinned to a fixed value, to stand in for the dead
            sensors the real dataset carries.
        sensor_value: if given, every live sensor holds this value on every row. Useful
            for checking that one engine's readings never bleed into another's.
    """
    rows = []
    for unit, lifetime in cycles_per_engine.items():
        for cycle in range(1, lifetime + 1):
            row = {config.UNIT_COLUMN: unit, config.CYCLE_COLUMN: cycle}
            row.update({name: 0.0 for name in config.OPERATIONAL_SETTING_COLUMNS})
            for offset, name in enumerate(config.SENSOR_COLUMNS):
                if name in constant_sensors:
                    row[name] = 42.0
                elif sensor_value is not None:
                    row[name] = sensor_value
                else:
                    # Drifts with age, so the sensors carry a learnable signal.
                    row[name] = float(cycle + offset)
            row[config.RUL_COLUMN] = lifetime - cycle
            rows.append(row)

    return pd.DataFrame(rows, columns=[*config.RAW_COLUMNS, config.RUL_COLUMN])
