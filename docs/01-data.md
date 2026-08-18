# Step 1 — The data

NASA C-MAPSS turbofan engine degradation simulation, subset **FD001**.

Reproduce every number on this page with:

```powershell
.\.venv\Scripts\python.exe -m src.explore
```

---

## 1. What the subset is

FD001 is the simplest of the four C-MAPSS subsets: **one operating condition, one fault
mode (high-pressure compressor degradation)**. That is deliberate. The point of this
project is the decision layer, not squeezing accuracy out of a six-condition dataset — a
harder subset would cost modelling time and buy nothing for the argument being made.

| | Rows | Engines |
|---|---|---|
| Training | 20,631 | 100 |
| Test | 13,096 | 100 |

Each engine starts with a different, unknown amount of initial wear. That is treated as
normal manufacturing variation, not a fault.

---

## 2. Engine lifetimes

| Statistic | Cycles |
|---|---|
| Shortest life | 128 |
| 25th percentile | 177 |
| **Median (`NOMINAL_LIFE`)** | **199** |
| Mean | 206 |
| 75th percentile | 229 |
| Longest life | 362 |

The spread is the whole reason condition-based maintenance can beat a calendar. The
longest-lived engine survives **2.8x** longer than the shortest. Any fixed interval is
either scrapping good engines or losing bad ones — usually both at once.

**Consequence for the cost model:** `NOMINAL_LIFE = 199`, so one wasted cycle costs
`1 / 199 = 0.005025` preventive overhauls. This is read from the data at runtime, never
typed into the code.

---

## 3. RUL sanity checks

The two splits encode remaining useful life differently, and getting this wrong is the
classic error on this dataset. Both checks pass:

| Check | Expected | Actual |
|---|---|---|
| Training RUL reaches zero | 0 — engines are recorded until they fail | **0** |
| Test RUL never reaches zero | > 0 — recording stops before failure | **min 7, max 145** |

If the test split had been handled with the training formula, every test engine would
appear to fail exactly when its recording stops, and the model would be trained against a
target that does not exist. The `RUL_FD001.txt` file exists precisely to prevent that.

---

## 4. Sensors

**4 of the 21 sensors are constant across the entire fleet** and carry zero information:

`sensor_1`, `sensor_10`, `sensor_18`, `sensor_19`

They are dropped by detecting zero variance at runtime rather than by hard-coding this
list — the same code then works unchanged on FD002–FD004, where the dead set differs.

The remaining **17 sensors all correlate with RUL**; none is noise. The strongest:

| Sensor | Correlation with RUL | Reads as |
|---|---|---|
| `sensor_11` | −0.696 | rises as the engine wears |
| `sensor_4` | −0.679 | rises as the engine wears |
| `sensor_12` | +0.672 | falls as the engine wears |
| `sensor_7` | +0.657 | falls as the engine wears |
| `sensor_15` | −0.643 | rises as the engine wears |
| `sensor_21` | +0.636 | falls as the engine wears |

A correlation around 0.7 on raw readings is a good sign for step 2: the degradation
signal is real and strong enough that a simple model will find it. It also sets
expectations — no single sensor is a clean countdown, which is why rolling-window
features are worth building.

---

## 5. Operating conditions

`op_setting_3` is **constant at 100.0** across all 20,631 rows, confirming the single
flight condition NASA documents for FD001. `op_setting_1` and `op_setting_2` vary only
within ±0.009 and ±0.0006 — simulated sensor noise, not real regime changes.

**Consequence:** no per-condition normalisation is needed here. On FD002 or FD004 (six
conditions) it would be mandatory, and sensor readings would have to be standardised
within each condition before they meant anything.

---

## 6. What this settles for step 2

- Drop zero-variance sensors, detected rather than listed.
- Build rolling-window features over the 17 surviving sensors — single readings are noisy,
  trends are not.
- Split **by engine**, never by row: rows from one engine are not independent, and mixing
  them across the split leaks the answer.
- Cap the training target at `RUL_CAP = 130`. Engines sit healthy for a long time before
  degrading, and "this engine has 300 cycles left" versus "280" is a distinction nobody
  makes a decision on. Capping stops the model spending capacity on it.
