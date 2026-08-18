# Predictive Maintenance Simulator

Turns a machine-failure prediction into an **explicit maintenance decision**, driven by a
configurable cost trade-off between intervening too early and intervening too late.

The model is deliberately simple. The value is in the decision layer on top of it.

## Why this exists

Portfolio project for a product / strategy role in deep tech. Most portfolio ML projects stop
at "here is my accuracy". This one answers the question a business actually asks:
**given what the model predicts, what should we do, and what does that save us?**

## Headline metric

Not RMSE. Total maintenance cost of the policy, compared against two baselines:

- **Run to failure** — repair only when the machine breaks
- **Calendar-based** — repair every N cycles regardless of condition (what most plants do today)

## Pipeline

| Step | What it does | Output |
|------|--------------|--------|
| 0 | Scoping: cost model, decision policy, baselines, success metric | `docs/00-scoping.md` |
| 1 | Dataset: NASA Turbofan Engine Degradation (C-MAPSS) | `data/raw/` |
| 2 | Feature engineering + RUL model (scikit-learn) | `models/` |
| 3 | Decision layer: prediction + cost ratio -> recommendation | `src/` |
| 4 | Streamlit dashboard with a cost-ratio slider | `src/app.py` |
| 5 | Narrative: the trade-off chosen and why | `docs/05-narrative.md` |

## Setup

```powershell
cd C:\Users\lukab\Developper\predictive-maintenance-sim
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No API key and no network access required — the project runs entirely offline.

## Status

- **Step 0 done** — cost model and decision policy defined in `docs/00-scoping.md`,
  mirrored as constants in `src/config.py`.
- **Step 1 done** — C-MAPSS FD001 loaded and validated (`src/data_loader.py`, 13 tests);
  findings written up in `docs/01-data.md` and reproducible with `python -m src.explore`.
- **Step 2 done** — 91 rolling-window features and a gradient-boosted RUL model
  (`src/features.py`, `src/model.py`). Held-out RMSE 13.39, NASA test-set RMSE 13.22.
  Write-up in `docs/02-model.md`; retrain with `python -m src.model`.
- **Step 3 done** — the decision layer (`src/decision.py`). Thresholds derived by
  replaying the fleet at every candidate and pricing each one. **26.7% cheaper than a
  cost-optimised calendar policy, 78.3% cheaper than running to failure**, measured on
  engines that shaped neither the model nor the threshold. Write-up in
  `docs/03-decision.md`; run with `python -m src.decision`.
- **Next** — the Streamlit dashboard.

61 tests, no network access, no API key.

## The result that took the longest to get right

Priced on cost alone, the optimal policy is "intervene five cycles before failure" —
arithmetically correct, operationally impossible. The cost model prices wasted life and
unplanned failure but puts no value on *notice*, and no shop sources a part in five
flights.

Adding the missing operator input — how much warning the shop actually needs — did not
cost anything. It made the policy **cheaper** on unseen engines, because the knife-edge
optimum had been overfitted to the engines that chose it. `docs/03-decision.md`, section 3.

The raw `.txt` files are not committed. Download them into `data/raw/`; any entry point
that needs them fails with the download URL rather than a bare traceback.
