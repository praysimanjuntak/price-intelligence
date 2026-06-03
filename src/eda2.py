"""Phase 1b — deeper checks that decide the modelling/calibration design.

Answers:
  A. Train temporal range vs the test days. Is train strictly before test?
  B. Is cat_id / brand deterministic per modelId/itemId? (recover for blank rows)
  C. How well does "last-known train price per modelId" predict the 300 anchors?
     -> this is the NO-calibration baseline error on real outage data.
  D. Does a global multiplicative shift estimated from the anchors reduce error?
  E. Time structure of train per modelId: is price a step function or trend?

Usage:
    python -m src.eda2
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src import data_io


def mape(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return float(np.mean(np.abs(yhat - y) / np.clip(np.abs(y), 1, None)) * 100)


def smape(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(np.mean(2 * np.abs(yhat - y) / np.clip(np.abs(y) + np.abs(yhat), 1, None)) * 100)


def run() -> None:
    train = data_io.load_train()
    test = data_io.load_test()

    print("\n=== A. Temporal ranges ===")
    print("train capturedAt:", train[C.TIME_COL].min(), "->", train[C.TIME_COL].max())
    print("test  capturedAt:", test[C.TIME_COL].min(), "->", test[C.TIME_COL].max())
    print("train distinct days:", train[C.TIME_COL].dt.date.nunique())
    days = sorted(train[C.TIME_COL].dt.date.unique())
    print("first/last train days:", days[0], days[-1])
    print("rows per train day (head):")
    print(train.groupby(train[C.TIME_COL].dt.date).size().tail(10))

    print("\n=== B. Is cat_id / brand deterministic per modelId / itemId? ===")
    for key in ["modelId", "itemId"]:
        for attr in ["cat_id", "brand"]:
            nun = train.groupby(key)[attr].nunique(dropna=True)
            multi = (nun > 1).sum()
            print(f"  {attr} per {key}: {multi} keys have >1 distinct value "
                  f"(out of {nun.shape[0]}) -> {'OK deterministic' if multi==0 else 'NOT unique'}")

    print("\n=== C. No-calibration baseline: last-known modelId price -> anchors ===")
    # Build last-known price per modelId from train (max capturedAt)
    tr = train.dropna(subset=[C.TARGET]).sort_values(C.TIME_COL)
    last_price = tr.groupby("modelId").agg(
        last_price=(C.TARGET, "last"),
        median_price=(C.TARGET, "median"),
    )
    anchors = test[test[C.TARGET].notna()].copy()
    anchors = anchors.merge(last_price, on="modelId", how="left")
    cov = anchors["last_price"].notna().mean() * 100
    a = anchors.dropna(subset=["last_price"])
    print(f"  anchors with modelId history: {len(a)}/{len(anchors)} ({cov:.1f}%)")
    for name, col in [("last", "last_price"), ("median", "median_price")]:
        print(f"  [{name}] MAPE={mape(a[C.TARGET], a[col]):.2f}%  "
              f"sMAPE={smape(a[C.TARGET], a[col]):.2f}%  "
              f"median|ratio-1|={np.median(np.abs(a[C.TARGET]/a[col]-1))*100:.2f}%")

    print("\n=== D. Per-day anchor shift (does a global factor help?) ===")
    a = a.assign(day=a[C.TIME_COL].dt.date, ratio=a[C.TARGET] / a["last_price"])
    for day, sub in a.groupby("day"):
        # leave-one-out style: use median ratio of OTHER anchors as the day factor
        gfac = sub["ratio"].median()
        base_mape = mape(sub[C.TARGET], sub["last_price"])
        cal_mape = mape(sub[C.TARGET], sub["last_price"] * gfac)
        print(f"  {day}: n={len(sub)}  median ratio (anchor/last)={gfac:.4f}  "
              f"MAPE no-cal={base_mape:.2f}%  global-cal={cal_mape:.2f}%")

    print("\n=== E. Price time-structure per modelId (step vs trend) ===")
    g = tr.groupby("modelId")[C.TARGET]
    nunq = g.nunique()
    cnt = g.size()
    multi = nunq[cnt >= 5]
    print(f"  modelIds with >=5 obs: {len(multi):,}")
    print(f"  share with a single distinct price (perfectly flat): "
          f"{(multi == 1).mean()*100:.1f}%")
    print(f"  share with <=3 distinct prices: {(multi <= 3).mean()*100:.1f}%")
    print(f"  median distinct prices among multi-obs modelIds: {multi.median():.0f}")


if __name__ == "__main__":
    run()
