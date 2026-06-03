# MrScraper Take-Home Submission Package

This folder is the curated deliverable version of the project. It excludes raw
CSV/parquet data because the files are downloaded from the Google Drive links in
`src/config.py` via `make download`.

## Submission Checklist

| Requirement | Included here |
|---|---|
| README explaining approach, run steps, and findings | `README.md` |
| Preprocessing, feature engineering, and model code | `src/` |
| Results summary comparing both approaches | `reports/results_summary.md`, `reports/*.json`, `notebooks/results.ipynb` |
| Trained model artifacts or reproduction instructions | `models/` plus `README.md` reproduction commands |
| Requirements/environment file | `pyproject.toml`, `uv.lock`, `requirements.txt`, `requirements-lock.txt` |
| Final inference entrypoint | `src/infer.py`, `make infer` |
| Assignment PDF | `test.pdf` |

## Default Reproduction

```bash
uv sync --locked
make download
make prepare
make backtest
make tier1
make calib
make tier2
make results
make infer
```

## Selected Production Model

The selected final model is Tier 2 hierarchical LOCF with per-category anchor
calibration:

```text
modelId -> itemId -> shopId -> cat_id -> global median
```

The same-day anchors estimate global and category-level multiplicative
correction factors. The category factors are shrunk toward the global factor to
avoid overfitting thin categories.

## Leakage Control

Validation simulates the outage day directly: each held-out day uses strict
pre-day history, 100 same-day anchors, and hidden target prices for all
remaining rows. Tier 1 CatBoost training features are also built day by day
from prior history only, while the selected Tier 2 production path fits entity
statistics from training history and calibrates each prediction day using only
that day's anchors.
