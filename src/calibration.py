"""Phase 4 — Anchor-set calibration.

The blank test rows give us no same-day features; the ONLY same-day signal is
the 100 anchor prices. Calibration uses the ratio between each anchor's true
price and its *base prediction* to estimate a day-level (and optionally
group-level) multiplicative correction, then applies it to all predictions.

Calibration is base-predictor-agnostic: any callable that maps
(EntityStats, DataFrame) -> ndarray can serve as the base. The default is
LOCF, but you can plug in a trained model's output (e.g. the Tier 1 CatBoost)
to calibrate that model's predictions too.

Strategies
----------
- none      : no correction (pure base prediction)
- global    : single multiplicative factor = median(anchor_true / anchor_pred)
- category  : per cat_id factor, shrunk toward the global factor
- shoptier  : per shop-rating tier factor, shrunk toward global

Shrinkage guards thin groups (few anchors). Because the 3 shared days have ~0
real drift, we also provide a synthetic-shift stress test that injects a known
platform-wide (and category-specific) price shift on the held-out day and
checks that calibration recovers it.

Usage:
    python -m src.calibration            # backtest + synthetic-shift stress test
"""
from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd

from src import config as C
from src import data_io, metrics
from src.features import EntityStats

BasePredictor = Callable[[EntityStats, pd.DataFrame], np.ndarray]


def locf_base(es: EntityStats, frame: pd.DataFrame) -> np.ndarray:
    """Default base predictor: hierarchical last-price via EntityStats."""
    return es.transform(frame)["last_price"].to_numpy()


def _safe_ratio(true_vals, pred_vals):
    pred = np.clip(pred_vals, 1.0, None)
    return np.asarray(true_vals, float) / pred


def estimate_factors(
    anchors: pd.DataFrame,
    es: EntityStats,
    strategy: str,
    base_predictor: BasePredictor = locf_base,
    shrink: float = 10.0,
) -> dict:
    """Return {'global': g, 'by_cat': {...}, 'by_tier': {...}} from anchors.

    Parameters
    ----------
    anchors : DataFrame with true prices in the target column.
    es : EntityStats fitted on strict pre-day history.
    strategy : "none" | "global" | "category" | "shoptier"
    base_predictor : callable (es, frame) -> ndarray of base predictions.
    shrink : shrinkage weight for group-level estimates (higher = more toward global).
    """
    base = base_predictor(es, anchors)
    true = anchors[C.TARGET].to_numpy(float)
    log_ratio = np.log(_safe_ratio(true, base))
    g = float(np.median(log_ratio))

    info = {"global": g, "by_cat": {}, "by_tier": {}}
    a = anchors.copy()
    a["_lr"] = log_ratio

    if strategy == "category":
        cats = es.transform(anchors)
        a["_cat"] = cats["cat_id"].astype("string").fillna("NA").to_numpy() \
            if "cat_id" in cats.columns else "NA"
        for cat, sub in a.groupby("_cat"):
            n = len(sub)
            local = float(np.median(sub["_lr"]))
            w = n / (n + shrink)
            info["by_cat"][str(cat)] = w * local + (1 - w) * g

    if strategy == "shoptier":
        if "shop_rating" in anchors:
            tier = pd.qcut(anchors["shop_rating"].astype(float),
                           q=4, duplicates="drop")
            a["_tier"] = tier.astype("string").fillna("NA").to_numpy()
            for tv, sub in a.groupby("_tier"):
                n = len(sub)
                local = float(np.median(sub["_lr"]))
                w = n / (n + shrink)
                info["by_tier"][str(tv)] = w * local + (1 - w) * g

    return info


def apply_calibration(
    base_pred: np.ndarray,
    frame: pd.DataFrame,
    es: EntityStats,
    strategy: str,
    factors: dict,
) -> np.ndarray:
    """Apply estimated factors to an array of base predictions.

    Parameters
    ----------
    base_pred : ndarray from the base predictor (may be any model's output).
    frame : DataFrame being predicted (needed for cat_id lookup in category mode).
    es : EntityStats (needed for cat_id recovery).
    strategy : "none" | "global" | "category" | "shoptier"
    factors : dict from estimate_factors().

    Returns
    -------
    calibrated_predictions : ndarray, same length as base_pred.
    """
    if strategy == "none":
        return base_pred
    log_adj = np.full(len(frame), factors["global"])
    feats = es.transform(frame)
    if strategy == "category" and factors["by_cat"]:
        cat = feats["cat_id"].astype("string").fillna("NA").to_numpy() \
            if "cat_id" in feats.columns else np.array(["NA"] * len(frame))
        gmap = factors["by_cat"]
        log_adj = np.array([gmap.get(str(c), factors["global"]) for c in cat])
    if strategy == "shoptier" and factors["by_tier"]:
        log_adj = np.full(len(frame), factors["global"])
    return base_pred * np.exp(log_adj)


# --------------------------------------------------------------------------- #
# Backtest across strategies (on real shared days: expect ~no shift)
# --------------------------------------------------------------------------- #
def run_strategies(folds, strategies=("none", "global", "category"),
                   base_predictor: BasePredictor = locf_base) -> dict:
    results = {s: {"per_day": [], "_y": [], "_p": []} for s in strategies}
    for f in folds:
        es = EntityStats().fit(f.history)
        base = base_predictor(es, f.targets)
        for s in strategies:
            if s == "none":
                pred = base
            else:
                fac = estimate_factors(f.anchors, es, s, base_predictor)
                pred = apply_calibration(base, f.targets, es, s, fac)
            m = metrics.evaluate(f.y_true, pred)
            results[s]["per_day"].append({"day": str(f.day), **m.as_dict()})
            results[s]["_y"].append(f.y_true)
            results[s]["_p"].append(pred)
    out = {}
    for s in strategies:
        y = np.concatenate(results[s]["_y"]); p = np.concatenate(results[s]["_p"])
        out[s] = {"overall": metrics.evaluate(y, p).as_dict(),
                  "per_day": results[s]["per_day"]}
    return out


# --------------------------------------------------------------------------- #
# Synthetic-shift stress test: inject a KNOWN shift, verify recovery
# --------------------------------------------------------------------------- #
def synthetic_shift_test(df: pd.DataFrame, shift: float = 0.15,
                         cat_extra: float = 0.10, seed: int = C.SEED) -> dict:
    """Hold out the last day, multiply ALL its prices by (1+shift) plus an extra
    per-category bump, then check calibration recovers it from the anchors."""
    rng = np.random.default_rng(seed)
    df = df.dropna(subset=[C.TARGET]).copy()
    df["_day"] = df[C.TIME_COL].dt.date
    day = sorted(df["_day"].unique())[-1]
    history = df[df["_day"] < day].drop(columns="_day")
    day_rows = df[df["_day"] == day].drop(columns="_day").copy()

    cats = day_rows["cat_id"].dropna().unique()
    bump_cats = set(rng.choice(cats, size=max(1, len(cats) // 2),
                  replace=False)) if len(cats) else set()
    factor = np.where(day_rows["cat_id"].isin(bump_cats), 1 + shift + cat_extra, 1 + shift)
    day_rows[C.TARGET] = day_rows[C.TARGET].to_numpy() * factor

    anchor_idx = rng.choice(day_rows.index,
                            size=min(C.ANCHORS_PER_DAY, len(day_rows) - 1),
                            replace=False)
    anchors = day_rows.loc[anchor_idx]
    targets_full = day_rows.drop(index=anchor_idx)
    y_true = targets_full[C.TARGET].to_numpy()
    targets = targets_full.copy()
    hide = [c for c in targets.columns if c not in ["modelId", "itemId", "shopId", C.TIME_COL]]
    targets[hide] = np.nan

    es = EntityStats().fit(history)
    base = locf_base(es, targets)
    out = {"injected_global_shift": shift, "injected_cat_extra": cat_extra,
           "results": {}}
    for s in ["none", "global", "category"]:
        if s == "none":
            pred = base
        else:
            fac = estimate_factors(anchors, es, s, locf_base)
            pred = apply_calibration(base, targets, es, s, fac)
            if s == "global":
                out["recovered_global_factor"] = float(np.exp(fac["global"]) - 1)
        out["results"][s] = metrics.evaluate(y_true, pred).as_dict()
    return out


def _show(tag, res):
    for s, r in res.items():
        o = r["overall"] if "overall" in r else r
        print(f"  [{tag}/{s:<9}] MAPE={o['mape']:.3f}%  MedAPE={o['medape']:.3f}%  "
              f"MAE={o['mae']:,.0f}  n={o['n']:,}")


if __name__ == "__main__":
    df = data_io.load_train()
    folds = None
    from src.validation import make_backtest
    folds = make_backtest(df, n_days=5)

    print("=== Calibration on real shared days (expect ~no shift) ===")
    real = run_strategies(folds)
    _show("real", real)

    print("\n=== Synthetic shift stress test (inject +15% global, +10% half cats) ===")
    syn = synthetic_shift_test(df, shift=0.15, cat_extra=0.10)
    print(f"  recovered global factor: {syn.get('recovered_global_factor'):.4f} "
          f"(injected 0.1500)")
    _show("synthetic", syn["results"])

    (C.REPORTS_DIR / "calibration.json").write_text(
        json.dumps({"real_days": real, "synthetic_shift": syn}, indent=2)
    )
    print("\nsaved reports/calibration.json")
