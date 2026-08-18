# Step 2 — Features and the RUL model

Reproduce every number on this page with:

```powershell
.\.venv\Scripts\python.exe -m src.model
```

The model is deliberately ordinary. It exists to feed the decision layer, and a point of
RMSE bought here buys nothing for the question the project actually answers.

---

## 1. Features

91 columns, built from the 15 sensors that move (`docs/01-data.md`, section 4). Per
sensor, per cycle:

| Feature | What it captures |
|---|---|
| raw reading | the current value |
| rolling mean over 5 and 20 cycles | the smoothed level — single readings are noisy |
| rolling std over 5 and 20 cycles | instability, which rises as a part fails |
| drift from the engine's first reading | how far it has moved since new |

Plus `time_in_cycles` itself: how long an engine has already run is a strong predictor.

**Drift is measured per engine, against that engine's own first reading.** NASA documents
that engines leave the factory with different amounts of initial wear, treated as normal
variation. Measuring each engine against itself removes that offset, so the model sees
degradation rather than manufacturing spread.

Operational settings are not used. On FD001 `op_setting_3` is constant and the other two
vary within ±0.009 — simulated noise, not regime changes. On the six-condition subsets
they would be essential.

---

## 2. Two ways this could have silently cheated

Both produce a model that scores beautifully in validation and is useless in service.
Both have a dedicated test.

**A rolling window that reads the future.** A centred or forward-looking window at cycle
*t* would include cycles the engine has not lived yet. The test truncates an engine's
history and asserts the early features are unchanged — if the window looked forward, they
would not be.

**A split that puts one engine on both sides.** Cycle 99 and cycle 100 of the same engine
are near-identical. Splitting by row would put one in training and one in validation, and
the score would measure memorisation. The split is by engine, seeded, with no overlap —
and it has **three** parts rather than two:

| Set | Engines | Job | Read |
|---|---|---|---|
| `fit` | 60 | fits the model | — |
| `tuning` | 20 | measures model error, picks the intervention threshold | as often as needed |
| `evaluation` | 20 | reports the cost saving | **once** |

Two sets would have been enough for the model alone. The third exists because step 3
tunes a threshold, and a threshold scored on the engines that chose it always looks good.
Fitting on 60 engines instead of 80 costs about 0.3 RMSE. That is the price of a headline
number that means something.

A third guard: the list of usable sensors is decided once on the training split and
carried inside the model artefact, so prediction can never quietly use different columns
from training.

---

## 3. Choosing the model

Three candidates, same features, same seed. Measured during selection, on the 80/20
split that existed before the third set was carved out — so the absolute numbers are
slightly better than the final model's, but the comparison between rows is fair.

| Model | RMSE | MAE | **RMSE within 50 cycles of failure** | Fit time |
|---|---|---|---|---|
| Baseline — always predicts the mean | 43.64 | 38.35 | 66.06 | 0.1 s |
| Random Forest | 13.38 | 8.99 | 9.57 | 9.6 s |
| **HistGradientBoosting** | **13.10** | **8.70** | **7.85** | 3.0 s |

The fourth column is the one that decided it. Being 8 cycles wrong about an engine that
has 300 cycles left changes no decision. Being 8 cycles wrong about an engine that has 20
left changes everything. The two tree models are within 0.3 RMSE overall, but gradient
boosting is **18% more accurate in the band where the call is actually made** — and three
times faster to fit.

The baseline is in the table on purpose. Without it, "RMSE 13" is a number with no scale.
Against it, the model is **3.3x better** than knowing nothing.

No hyperparameter tuning was done. Library defaults, one fixed seed.

---

## 4. Results

Final model, fitted on 60 engines:

| | RMSE | MAE | Scope |
|---|---|---|---|
| Held-out tuning engines, every cycle | 13.39 | 8.75 | 3,852 rows / 20 engines |
| Within 50 cycles of failure | **9.67** | — | the decision band |
| NASA test engines, at cut-off | 13.22 | 9.95 | 100 engines |

**The held-out split and the official NASA test set agree to within 0.17 RMSE.** That is
the result worth trusting: the model was never tuned against the test set, and it performs
on unseen engines exactly as the held-out split predicted. No overfitting.

The NASA row uses the protocol the C-MAPSS literature reports — one prediction per
engine, at the cycle where recording stopped — so it is the number comparable with
published work. Classical machine-learning results on FD001 typically land in the 13–16
range, deep learning around 12–13.

---

## 5. What this settles for step 3

- **The prediction is good enough.** Typical error near failure is under 10 cycles
  against a median engine life of about 200. The remaining uncertainty is small enough
  for a cost argument to be meaningful, and large enough that ignoring it would be
  dishonest.
- **`validation_rmse = 13.39` is now a number the decision layer consumes**, not a
  reporting statistic. It sets the width of the "monitor closely" band: the zone where
  the model says an engine is safe but its own typical error is wide enough that it could
  be wrong.
- **`nominal_life = 202` fixes the cost of a wasted cycle** at 0.00495 preventive
  overhauls. It is the median of the 60 fitted engines, not of all 100.

Both travel inside the saved artefact, so the decision layer cannot use a threshold
derived from one model against the predictions of another.

---

## 6. Limitations

- **Predictions are capped at 130 cycles** by the training target. The model cannot
  distinguish "220 cycles left" from "300", by design — no decision depends on it.
- **Typical error near failure is 9.67 cycles**, against a median engine life of about
  200. Small enough for a cost argument to mean something; large enough that pretending
  a prediction is a date would be dishonest. Step 3 carries it into the recommendation
  rather than dropping it.
- **The rolling windows need history.** An engine on its first cycles has partial
  windows. Real early-life predictions are correspondingly weaker.
- **One seed, one split.** The reported RMSE is a single draw, not a cross-validated
  mean. Repeated splits would give a range; the honest reading of 13.39 is "about 13".
- **FD001 only.** One operating condition, one fault mode. Performance on the
  six-condition subsets would be worse and would need per-condition normalisation.
