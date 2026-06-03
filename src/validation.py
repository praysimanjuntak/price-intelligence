"""Outage-day backtest harness.

Mirrors the real test conditions: hold out the last K *train* days entirely
as pseudo "outage days". For each held-out day:
  - reveal a random ANCHORS_PER_DAY (=100) rows as the calibration anchor set,
  - hide the price of all remaining rows,
  - fit features on the strict history (rows strictly before that day),
  - predict the hidden prices and score them.

This gives label-backed error estimates (the real test blank rows have no
labels). A predictor is any callable:

    predict(history_df, target_df, anchors_df, entity_stats) -> np.ndarray

where target_df has the price column blanked.

Usage:
    python -m src.validation            # run LOCF baselines on backtest
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src import config as C
from src import data_io, metrics
from src.features import EntityStats

Predictor = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, EntityStats], np.ndarray]


@dataclass
class BacktestDay:
    day: object
    history: pd.DataFrame   # rows strictly before `day`
    anchors: pd.DataFrame   # 100 revealed rows on `day` (price known)
    targets: pd.DataFrame   # remaining rows on `day` (price hidden from model)
    y_true: np.ndarray      # true prices for targets


def make_backtest(
    df: pd.DataFrame,
    n_days: int = 5,
    anchors_per_day: int = C.ANCHORS_PER_DAY,
    seed: int = C.SEED,
) -> list[BacktestDay]:
    rng = np.random.default_rng(seed)
    df = df.dropna(subset=[C.TARGET]).copy()
    df["_day"] = df[C.TIME_COL].dt.date
    all_days = sorted(df["_day"].unique())
    held = all_days[-n_days:]
    folds: list[BacktestDay] = []
    for day in held:
        day_rows = df[df["_day"] == day]
        if len(day_rows) <= anchors_per_day:
            continue
        history = df[df["_day"] < day].drop(columns="_day")
        n_anchor = min(anchors_per_day, len(day_rows) - 1)
        anchor_idx = rng.choice(day_rows.index, size=n_anchor, replace=False)
        anchors = day_rows.loc[anchor_idx].drop(columns="_day")
        targets_full = day_rows.drop(index=anchor_idx).drop(columns="_day")
        y_true = targets_full[C.TARGET].to_numpy()
        # Blank the price + the columns that are blank in the real test set.
        targets = targets_full.copy()
        hide_cols = [c for c in targets.columns if c not in
                     ["modelId", "itemId", "shopId", C.TIME_COL]]
        targets[hide_cols] = np.nan
        folds.append(BacktestDay(day, history, anchors, targets, y_true))
    return folds


def run_backtest(predictor: Predictor, folds: list[BacktestDay], name: str) -> dict:
    per_day = []
    all_y, all_p = [], []
    for f in folds:
        es = EntityStats().fit(f.history)
        yhat = predictor(f.history, f.targets, f.anchors, es)
        m = metrics.evaluate(f.y_true, yhat)
        per_day.append({"day": str(f.day), **m.as_dict()})
        all_y.append(f.y_true)
        all_p.append(np.asarray(yhat, float))
    overall = metrics.evaluate(np.concatenate(all_y), np.concatenate(all_p))
    return {"name": name, "overall": overall.as_dict(), "per_day": per_day}


# --------------------------------------------------------------------------- #
# Baseline predictors
# --------------------------------------------------------------------------- #
def locf_predictor(history, targets, anchors, es: EntityStats) -> np.ndarray:
    """Last-observation-carried-forward via the entity fallback chain."""
    feats = es.transform(targets)
    return feats["last_price"].to_numpy()


def median_predictor(history, targets, anchors, es: EntityStats) -> np.ndarray:
    feats = es.transform(targets)
    return feats["median_price"].to_numpy()


def global_mean_predictor(history, targets, anchors, es: EntityStats) -> np.ndarray:
    return np.full(len(targets), float(history[C.TARGET].median()))


def _print(res: dict) -> None:
    o = res["overall"]
    print(f"\n[{res['name']}] OVERALL  "
          f"MAE={o['mae']:,.0f}  RMSE={o['rmse']:,.0f}  "
          f"MAPE={o['mape']:.3f}%  sMAPE={o['smape']:.3f}%  "
          f"MedAPE={o['medape']:.3f}%  R2={o['r2']:.5f}  n={o['n']:,}")
    for d in res["per_day"]:
        print(f"    {d['day']}: MAPE={d['mape']:.3f}%  MedAPE={d['medape']:.3f}%  "
              f"MAE={d['mae']:,.0f}  n={d['n']:,}")


if __name__ == "__main__":
    df = data_io.load_train()
    folds = make_backtest(df, n_days=5)
    print(f"Backtest folds: {[str(f.day) for f in folds]}")
    print(f"Targets per fold: {[len(f.targets) for f in folds]}")
    for name, fn in [
        ("global_median", global_mean_predictor),
        ("entity_median", median_predictor),
        ("LOCF (last price)", locf_predictor),
    ]:
        _print(run_backtest(fn, folds, name))
