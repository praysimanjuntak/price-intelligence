"""Tier 1 — Global Marketplace Model.

A single CatBoost model trained on the entire history, defaulting to **CPU**
for submission reproducibility. Pass `--gpu` for GPU training.

Rather than predict the raw price (which is dominated by entity identity), we
predict the *log residual* against the LOCF baseline:

    target = log1p(price) - log1p(last_known_price)

This focuses all model capacity on the ~17% of entities whose price actually
moves, while inheriting LOCF's near-perfect accuracy elsewhere. The final
prediction is:

    price_hat = expm1( log1p(last_price) + residual_hat )

The model consumes the full engineered feature set (entity price stats,
recency, volatility, temporal, categorical IDs), so it can learn shop/category
drift patterns the plain LOCF cannot.

Training features are built with a strict pre-day cutoff: for rows on day D,
entity statistics are fitted only on rows before D. This mirrors the outage
backtest and prevents within-fold look-ahead from later same-entity prices.

Usage:
    python -m src.model_global             # CPU backtest + train + SHAP
    python -m src.model_global --gpu       # GPU backtest
    python -m src.model_global --cpu       # explicit CPU (same as default)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src import config as C
from src import data_io, metrics
from src.features import EntityStats, feature_columns, CATEGORICAL_FEATURES
from src.validation import make_backtest, _print
from src.calibration import estimate_factors, apply_calibration


CB_PARAMS = dict(
    iterations=1200,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=C.SEED,
    od_type="Iter",
    od_wait=80,
    verbose=False,
)


def _prep_xy(history, frame, es: EntityStats, with_target: bool):
    feats = es.transform(frame)
    cols = feature_columns(feats)
    X = feats[cols].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("NA")
    base = np.log1p(feats["last_price"].clip(lower=0).to_numpy())
    if with_target:
        y_price = frame[C.TARGET].to_numpy(dtype=float)
        resid = np.log1p(np.clip(y_price, 0, None)) - base
        return X, cols, base, resid
    return X, cols, base, None


def _prep_training_xy(history: pd.DataFrame):
    """Build CatBoost training rows with strict pre-day entity statistics.

    For every training day D, features for rows on D are generated from
    EntityStats fitted on rows with capturedAt date strictly before D. This is
    slower than fitting one EntityStats object over the full fold, but it
    matches the outage setting and avoids look-ahead leakage.
    """
    h = history.dropna(subset=[C.TARGET]).copy()
    h["_day"] = h[C.TIME_COL].dt.date
    pieces = []

    for day in sorted(h["_day"].unique()):
        hist = h[h["_day"] < day].drop(columns="_day")
        if hist.empty:
            continue
        frame = h[h["_day"] == day].drop(columns="_day")
        es_day = EntityStats().fit(hist)
        feats = es_day.transform(frame)
        pieces.append((frame, feats))

    if not pieces:
        raise ValueError("Need at least two dated history slices to train Tier 1 safely")

    frames = [p[0] for p in pieces]
    feats_all = pd.concat([p[1] for p in pieces], axis=0).sort_index()
    frame_all = pd.concat(frames, axis=0).sort_index()
    cols = feature_columns(feats_all)
    X = feats_all[cols].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("NA")
    base = np.log1p(feats_all["last_price"].clip(lower=0).to_numpy())
    y_price = frame_all[C.TARGET].to_numpy(dtype=float)
    resid = np.log1p(np.clip(y_price, 0, None)) - base
    return X, cols, base, resid


def _cat_idx(cols):
    return [i for i, c in enumerate(cols) if c in CATEGORICAL_FEATURES]


def train_global(history: pd.DataFrame, use_gpu: bool = False) -> dict:
    es = EntityStats().fit(history)
    X, cols, base, resid = _prep_training_xy(history)
    cat_idx = _cat_idx(cols)
    params = dict(CB_PARAMS)
    if use_gpu:
        params.update(task_type="GPU", devices="0")
    model = CatBoostRegressor(**params)
    pool = Pool(X, label=resid, cat_features=cat_idx)
    model.fit(pool)
    return {"model": model, "es": es, "cols": cols, "cat_idx": cat_idx}


def run_tier1_backtests(folds, use_gpu: bool) -> dict:
    """Evaluate all Tier 1 variants while training CatBoost once per fold."""
    specs = {
        "locf": "LOCF baseline",
        "global_catboost": "Tier1 Global CatBoost",
        "gated": "Tier1 Gated (LOCF+model)",
        "cb_global_cal": "Tier1 CatBoost + global anchor cal",
        "cb_category_cal": "Tier1 CatBoost + per-category anchor cal",
    }
    acc = {k: {"name": v, "per_day": [], "_y": [], "_p": []} for k, v in specs.items()}

    for f in folds:
        es = EntityStats().fit(f.history)
        art = train_global(f.history, use_gpu=use_gpu)
        feats = es.transform(f.targets)
        locf = feats["last_price"].to_numpy()
        raw = predict_global(art, f.history, f.targets, f.anchors, es)

        cv = feats["price_cv"].fillna(0.0).to_numpy()
        n_obs = feats["model_n_obs"].fillna(0).to_numpy()
        use_locf = (cv < 0.01) & (n_obs >= 2)
        gated = np.where(use_locf, locf, raw)

        preds = {
            "locf": locf,
            "global_catboost": raw,
            "gated": gated,
            "cb_global_cal": raw,
            "cb_category_cal": raw,
        }
        if len(f.anchors) > 0:
            base_predictor = lambda e, frame, a=art, h=f.history: predict_global(a, h, frame, None, e)
            for key, strategy in [("cb_global_cal", "global"), ("cb_category_cal", "category")]:
                fac = estimate_factors(f.anchors, es, strategy, base_predictor=base_predictor)
                preds[key] = apply_calibration(raw, f.targets, es, strategy, fac)

        for key, pred in preds.items():
            m = metrics.evaluate(f.y_true, pred)
            acc[key]["per_day"].append({"day": str(f.day), **m.as_dict()})
            acc[key]["_y"].append(f.y_true)
            acc[key]["_p"].append(np.asarray(pred, float))

    out = {}
    for key, res in acc.items():
        y = np.concatenate(res.pop("_y"))
        p = np.concatenate(res.pop("_p"))
        res["overall"] = metrics.evaluate(y, p).as_dict()
        out[key] = res
    return out


def predict_global(art: dict, history, targets, anchors, es: EntityStats) -> np.ndarray:
    X, cols, base, _ = _prep_xy(history, targets, es, with_target=False)
    X = X[art["cols"]]
    pool = Pool(X, cat_features=art["cat_idx"])
    resid_hat = art["model"].predict(pool)
    return np.expm1(base + resid_hat)


def make_backtest_predictor(use_gpu: bool):
    def _pred(history, targets, anchors, es: EntityStats) -> np.ndarray:
        art = train_global(history, use_gpu=use_gpu)
        return predict_global(art, history, targets, anchors, es)
    return _pred


def make_gated_predictor(use_gpu: bool, cv_threshold: float = 0.01):
    def _pred(history, targets, anchors, es: EntityStats) -> np.ndarray:
        art = train_global(history, use_gpu=use_gpu)
        model_pred = predict_global(art, history, targets, anchors, es)
        feats = es.transform(targets)
        last = feats["last_price"].to_numpy()
        cv = feats["price_cv"].fillna(0.0).to_numpy()
        n_obs = feats["model_n_obs"].fillna(0).to_numpy()
        use_locf = (cv < cv_threshold) & (n_obs >= 2)
        return np.where(use_locf, last, model_pred)
    return _pred


def make_calibrated_tier1_predictor(use_gpu: bool, strategy: str):
    """Tier 1 (global CatBoost base) + anchor calibration.

    Trains the global model on fold history, then applies per-category or
    global anchor calibration to its predictions. This is how Tier 1 uses the
    100 anchors — scored explicitly in the brief under anchor utilisation."""
    def _pred(history, targets, anchors, es: EntityStats) -> np.ndarray:
        art = train_global(history, use_gpu=use_gpu)
        base = predict_global(art, history, targets, anchors, es)
        if strategy == "none" or len(anchors) == 0:
            return base
        fac = estimate_factors(anchors, es, strategy,
                               base_predictor=lambda e, f: predict_global(art, history, f, anchors, e))
        return apply_calibration(base, targets, es, strategy, fac)
    return _pred


def shap_report(art: dict, history, n_sample: int = 5000) -> None:
    es = art["es"]
    sample = history.sample(min(n_sample, len(history)), random_state=C.SEED)
    X, cols, base, _ = _prep_xy(history, sample, es, with_target=False)
    X = X[art["cols"]]
    pool = Pool(X, cat_features=art["cat_idx"])
    sv = art["model"].get_feature_importance(pool, type="ShapValues")
    imp = np.abs(sv[:, :-1]).mean(axis=0)
    order = np.argsort(imp)[::-1]
    rows = [{"feature": art["cols"][i], "mean_abs_shap": float(imp[i])} for i in order]
    out = C.REPORTS_DIR / "shap_global.json"
    out.write_text(json.dumps(rows, indent=2))
    print("\n[SHAP] top 15 features (global model):")
    for r in rows[:15]:
        print(f"   {r['feature']:<28} {r['mean_abs_shap']:.5f}")


if __name__ == "__main__":
    use_gpu = "--gpu" in sys.argv

    df = data_io.load_train()
    folds = make_backtest(df, n_days=5)

    print(f"=== Tier 1 Global CatBoost — backtest ({'GPU' if use_gpu else 'CPU'}) ===")
    tier1_results = run_tier1_backtests(folds, use_gpu=use_gpu)

    for key in ["locf", "global_catboost", "gated", "cb_global_cal", "cb_category_cal"]:
        _print(tier1_results[key])
    (C.REPORTS_DIR / "tier1_backtest.json").write_text(
        json.dumps(tier1_results, indent=2))

    print(f"\n=== Training full global model on all history ({'GPU' if use_gpu else 'CPU'}) ===")
    art = train_global(df, use_gpu=use_gpu)
    art["model"].save_model(str(C.MODELS_DIR / "global_catboost.cbm"))
    print("saved models/global_catboost.cbm")
    try:
        shap_report(art, df)
    except Exception as e:
        print(f"[SHAP] skipped ({e})")
