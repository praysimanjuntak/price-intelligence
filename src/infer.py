"""Inference — fill blank prices in the test file and write a completed CSV.

Production model (selected by backtest): hierarchical LOCF base
(modelId -> itemId -> shopId -> cat_id -> global) with per-category anchor
calibration applied *per prediction day*.

HISTORY POLICY (strict temporal isolation)
------------------------------------------
- Entity statistics are fitted ONCE on the full training set (dates
  2025-01-01 .. 2025-03-22). Test data is NEVER merged into the history.
- Each prediction day is processed independently: the 100 anchors for day D
  calibrate D's predictions only; no anchor information leaks across days.
- Prior test-day *predictions* (filled prices) are NEVER fed back as history
  for later test days. This prevents compounding errors and keeps the
  inference pipeline stateless per day — you can process days in any order.

Rationale: in a real outage, the 100 manually-collected prices are the only
same-day information available. Letting predictions from earlier (already
predicted) days influence later days would construct a fake signal that
doesn't exist in production.

Output guarantees
------------------
- Anchors keep their original (true) price (never overwritten).
- Column order matches the input CSV exactly.
- All blank-price rows receive a predicted value.
- `price` is written as integer IDR (platform's smallest currency unit).

Usage:
    python -m src.infer                          # 3-day file -> data/test_completed.csv
    python -m src.infer --test path/to/test.csv --out path/to/out.csv --strategy category
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config as C
from src import data_io
from src.features import EntityStats
from src.calibration import estimate_factors, apply_calibration, locf_base

CALIB_STRATEGY = "category"


def fill_prices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    strategy: str = CALIB_STRATEGY,
) -> pd.DataFrame:
    """Fill blank prices in `test`.

    History is the full training set. EntityStats is fitted once upfront; no
    test data (anchor rows or predictions) ever enters the history.

    Per day: use that day's 100 anchors to estimate calibration factors, then
    calibrate the base LOCF predictions for the blank rows.
    """
    es = EntityStats().fit(train)
    out = test.copy()
    out["_day"] = out[C.TIME_COL].dt.date

    filled_total = 0
    for day, idx in out.groupby("_day").groups.items():
        sub = out.loc[idx]
        anchors = sub[sub[C.TARGET].notna()]
        blanks = sub[sub[C.TARGET].isna()]
        if blanks.empty:
            continue
        base = locf_base(es, blanks)
        if len(anchors) > 0:
            fac = estimate_factors(anchors, es, strategy, base_predictor=locf_base)
            pred = apply_calibration(base, blanks, es, strategy, fac)
            gfac = float(np.exp(fac["global"]) - 1)
        else:
            pred = base
            gfac = float("nan")
        out.loc[blanks.index, C.TARGET] = pred
        filled_total += len(blanks)
        print(f"  {day}: filled {len(blanks):,} blanks  "
              f"anchors={len(anchors)}  global_shift={gfac:+.4f}")

    out = out.drop(columns="_day")
    assert out[C.TARGET].notna().all(), "some prices remain blank!"
    print(f"[infer] filled {filled_total:,} rows; total non-null price = "
          f"{int(out[C.TARGET].notna().sum()):,}/{len(out):,}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(C.TEST_CSV),
                    help="Path to the test CSV file")
    ap.add_argument("--out", default=str(C.COMPLETED_CSV),
                    help="Path to write the completed CSV")
    ap.add_argument("--strategy", default=CALIB_STRATEGY,
                    choices=["none", "global", "category"],
                    help="Anchor calibration strategy")
    args = ap.parse_args()

    train = data_io.load_train()
    raw = pd.read_csv(args.test, low_memory=False)
    test = data_io._coerce(raw)

    print(f"[infer] train={train.shape}  test={test.shape}  strategy={args.strategy}")
    completed = fill_prices(train, test, args.strategy)

    # Preserve original column order; write price as integer IDR.
    completed = completed[[c for c in raw.columns if c in completed.columns]]
    completed[C.TARGET] = completed[C.TARGET].round().astype("int64")
    completed.to_csv(args.out, index=False)

    # Final integrity checks
    n_total = len(completed)
    n_null = int(completed[C.TARGET].isna().sum())
    n_nonpos = int((completed[C.TARGET] <= 0).sum())
    n_unchanged = 0
    n_anchors = 0
    if C.TARGET in raw.columns:
        orig_anchors = raw[C.TARGET].copy()
        new_prices = completed[C.TARGET].copy()
        anchor_mask = orig_anchors.notna()
        n_anchors = int(anchor_mask.sum())
        if n_anchors > 0:
            n_unchanged = int((orig_anchors[anchor_mask].astype(float).to_numpy()
                               == new_prices[anchor_mask].to_numpy()).sum())

    print(f"[infer] wrote {args.out}  shape={completed.shape}")
    print(f"[infer] integrity: nulls={n_null} non-pos={n_nonpos} "
          f"anchors-unchanged={n_unchanged}/{n_anchors}")
    if n_unchanged != n_anchors:
        raise AssertionError(
            f"anchor prices changed: {n_unchanged}/{n_anchors} "
            f"({n_anchors - n_unchanged} anchors were overwritten)"
        )


if __name__ == "__main__":
    main()
