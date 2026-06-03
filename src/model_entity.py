"""Tier 2 — Shop / Product-level model.

A hierarchical per-entity predictor that captures entity-specific pricing.
The base prediction walks a fallback chain to handle cold-start gracefully:

    modelId last price
      -> itemId median price
        -> shopId median price
          -> cat_id median price
            -> global median

On top of the base, an optional anchor calibration (global or per-category,
reused from src.calibration) corrects for day-level drift at the entity group
level. This is the "conditioned model" the brief describes for Tier 2.

We also report a robust variant that uses a trimmed per-modelId estimate
(median of the most recent K observations) to damp flash-sale outliers.

Usage:
    python -m src.model_entity
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config as C
from src import data_io, metrics
from src.features import EntityStats
from src.validation import make_backtest, run_backtest, locf_predictor, _print
from src.calibration import estimate_factors, apply_calibration


def hierarchical_predictor(history, targets, anchors, es: EntityStats) -> np.ndarray:
    """Pure hierarchical fallback (no calibration)."""
    feats = es.transform(targets)
    return feats["last_price"].to_numpy()


def make_calibrated_entity_predictor(strategy: str = "category"):
    """Hierarchical base + anchor calibration at the group level."""
    def _pred(history, targets, anchors, es: EntityStats) -> np.ndarray:
        feats = es.transform(targets)
        base = feats["last_price"].to_numpy()
        fac = estimate_factors(anchors, es, strategy)
        return apply_calibration(base, targets, es, strategy, fac)
    return _pred


def robust_recent_predictor(history, targets, anchors, es: EntityStats,
                            k: int = 5) -> np.ndarray:
    """Median of the most recent K prices per modelId (outlier-damped LOCF)."""
    h = history.dropna(subset=[C.TARGET]).sort_values(C.TIME_COL)
    recent = (
        h.groupby("modelId").tail(k)
        .groupby("modelId")[C.TARGET].median()
        .rename("recent_med")
    )
    merged = targets[["modelId"]].merge(recent, on="modelId", how="left")
    feats = es.transform(targets)
    base = feats["last_price"].to_numpy()
    out = merged["recent_med"].to_numpy()
    return np.where(np.isfinite(out), out, base)


if __name__ == "__main__":
    df = data_io.load_train()
    folds = make_backtest(df, n_days=5)

    print("=== Tier 2 Shop/Product-level — backtest ===")
    res = {}
    res["locf"] = run_backtest(locf_predictor, folds, "LOCF (modelId last)")
    res["hier"] = run_backtest(hierarchical_predictor, folds, "Hierarchical fallback")
    res["robust_recent"] = run_backtest(robust_recent_predictor, folds, "Robust recent-K median")
    res["cal_global"] = run_backtest(
        make_calibrated_entity_predictor("global"), folds, "Hier + global anchor cal")
    res["cal_category"] = run_backtest(
        make_calibrated_entity_predictor("category"), folds, "Hier + per-category anchor cal")

    for k in res:
        _print(res[k])

    (C.REPORTS_DIR / "tier2_backtest.json").write_text(json.dumps(res, indent=2))
    print("\nsaved reports/tier2_backtest.json")
