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

Step 0 done — cost model and decision policy defined in `docs/00-scoping.md`, mirrored as
constants in `src/config.py`. Next: load the C-MAPSS data.
