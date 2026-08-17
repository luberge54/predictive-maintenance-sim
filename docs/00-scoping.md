# Step 0 — Scoping

The business trade-off is defined here, before any data is loaded. The model serves this
document, not the other way around.

---

## 1. Cost model

The two ways a maintenance decision can be wrong:

| Error | What happens | Cost |
|-------|--------------|------|
| False alarm — intervene too early | Machine pulled from service while still healthy; remaining useful life wasted | `C_early` = TBD |
| Missed failure — intervene too late | Unplanned breakdown: repair, downtime, possible collateral damage | `C_late` = TBD |

**Cost ratio** `C_late / C_early` = TBD

> Do not hard-code a single ratio. Expose it as a slider in the dashboard and show how the
> optimal intervention threshold moves across the range. The honest answer to "where does
> your ratio come from?" is "it's yours to set — here is the curve".

---

## 2. Decision policy

How a predicted remaining useful life (RUL) becomes an instruction a plant manager can act on.

| Recommendation | Condition | Meaning |
|----------------|-----------|---------|
| Intervene now | TBD | |
| Monitor closely | TBD | |
| Healthy for N more cycles | TBD | |

Thresholds are derived from the cost ratio, not chosen by hand.

---

## 3. Baselines to beat

The model is only worth something relative to what a plant does today.

| Baseline | Policy | Total cost |
|----------|--------|------------|
| Run to failure | Repair only after breakdown | TBD |
| Calendar-based | Repair every N cycles regardless of condition | TBD |
| **This project** | Condition-based, cost-optimised threshold | TBD |

---

## 4. Success metric

**Headline number:** percentage of total maintenance cost saved versus the calendar-based
baseline, at the chosen cost ratio.

Model accuracy metrics (RMSE on RUL) are reported as supporting detail only. They are never
the headline.

---

## 5. Known limitations

- TBD
