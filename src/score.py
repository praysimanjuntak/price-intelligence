"""Score a completed validation CSV against a ground-truth CSV.

This is intended for the situation where you receive:

    data/val.csv         # your completed/predicted file
    data/val_truth.csv   # same rows with true prices

The scorer first checks that the files are comparable:
  - same row count
  - same column order
  - all non-price columns are identical after dtype coercion
  - price columns are numeric and finite

Then it reports MAE, RMSE, MAPE, sMAPE, MedAPE, and R2 on the price column.

Usage:
    python -m src.score --pred data/val.csv --truth data/val_truth.csv
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src import config as C
from src import data_io, metrics


def _read(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, low_memory=False)
    coerced = data_io._coerce(raw)
    return raw, coerced


def _assert_same_schema(pred_raw: pd.DataFrame, truth_raw: pd.DataFrame) -> None:
    if len(pred_raw) != len(truth_raw):
        raise AssertionError(
            f"row count mismatch: pred={len(pred_raw):,} truth={len(truth_raw):,}"
        )
    if list(pred_raw.columns) != list(truth_raw.columns):
        pred_cols = list(pred_raw.columns)
        truth_cols = list(truth_raw.columns)
        raise AssertionError(
            "column order/name mismatch:\n"
            f"pred columns : {pred_cols}\n"
            f"truth columns: {truth_cols}"
        )
    if C.TARGET not in pred_raw.columns:
        raise AssertionError(f"missing required target column: {C.TARGET}")


def _assert_non_price_identical(pred: pd.DataFrame, truth: pd.DataFrame) -> None:
    check_cols = [c for c in pred.columns if c != C.TARGET]
    mismatches: list[str] = []

    for col in check_cols:
        left = pred[col]
        right = truth[col]
        same = left.eq(right) | (left.isna() & right.isna())
        if not bool(same.all()):
            bad_idx = same[~same].index[:5].tolist()
            mismatches.append(f"{col} at rows {bad_idx}")

    if mismatches:
        preview = "\n".join(f"  - {m}" for m in mismatches[:20])
        extra = "" if len(mismatches) <= 20 else f"\n  ... {len(mismatches) - 20} more columns"
        raise AssertionError(
            "non-price columns are not identical between prediction and truth:\n"
            f"{preview}{extra}"
        )


def score(pred_path: str, truth_path: str) -> metrics.Metrics:
    pred_raw, pred = _read(pred_path)
    truth_raw, truth = _read(truth_path)

    _assert_same_schema(pred_raw, truth_raw)
    _assert_non_price_identical(pred, truth)

    yhat = pd.to_numeric(pred[C.TARGET], errors="coerce").to_numpy(float)
    y = pd.to_numeric(truth[C.TARGET], errors="coerce").to_numpy(float)

    pred_bad = int((~np.isfinite(yhat)).sum())
    truth_bad = int((~np.isfinite(y)).sum())
    if pred_bad or truth_bad:
        raise AssertionError(
            f"non-finite price values: pred={pred_bad:,} truth={truth_bad:,}"
        )

    return metrics.evaluate(y, yhat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(C.DATA_DIR / "val.csv"),
                    help="Completed/predicted validation CSV")
    ap.add_argument("--truth", default=str(C.DATA_DIR / "val_truth.csv"),
                    help="Ground-truth validation CSV")
    args = ap.parse_args()

    try:
        m = score(args.pred, args.truth)
    except Exception as e:
        print(f"[score] FAILED: {e}", file=sys.stderr)
        raise

    print("[score] checks passed: same rows, same schema, non-price columns identical")
    print(f"[score] pred={args.pred}")
    print(f"[score] truth={args.truth}")
    print(f"n={m.n:,}")
    print(f"MAE={m.mae:,.0f}")
    print(f"RMSE={m.rmse:,.0f}")
    print(f"MAPE={m.mape:.4f}%")
    print(f"sMAPE={m.smape:.4f}%")
    print(f"MedAPE={m.medape:.4f}%")
    print(f"R2={m.r2:.6f}")


if __name__ == "__main__":
    main()
