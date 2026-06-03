"""Exploratory data analysis.

Produces a markdown report (reports/eda.md) and a few plots answering the
questions that drive the modelling strategy:

  1. On blank-price TEST rows, which feature columns are populated?
     (If priceBeforeDiscount + show_discount/raw_discount are present, price is
      near-deterministic and the task is mostly reconstruction + calibration.)
  2. Price distribution, scale, and outliers.
  3. Per-entity (modelId / itemId / shopId) history depth -> cold-start rate.
  4. Day-to-day price volatility per entity (how good is "last known price"?).
  5. Anchor-row representativeness vs the full day.

Usage:
    python -m src.eda
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from src import config as C
from src import data_io


def _section(buf: io.StringIO, title: str) -> None:
    buf.write(f"\n\n## {title}\n\n")


def _fmt_df(df: pd.DataFrame) -> str:
    return df.to_markdown()


def run() -> None:
    train = data_io.load_train()
    test = data_io.load_test()
    buf = io.StringIO()
    buf.write("# EDA Report — MrScraper Price Intelligence\n")
    buf.write(f"\nTrain shape: {train.shape}  |  Test (3 days) shape: {test.shape}\n")

    # ---------------------------------------------------------------- #
    # 1. Column population on blank-price test rows  (THE key question)
    # ---------------------------------------------------------------- #
    _section(buf, "1. Column population: anchor vs blank-price test rows")
    is_anchor = test[C.TARGET].notna()
    pop = pd.DataFrame(
        {
            "anchor_nonnull_%": (test[is_anchor].notna().mean() * 100).round(1),
            "blank_nonnull_%": (test[~is_anchor].notna().mean() * 100).round(1),
            "train_nonnull_%": (train.notna().mean() * 100).round(1),
        }
    )
    buf.write(_fmt_df(pop))

    # Is price reconstructable from discount columns where they exist?
    _section(buf, "1b. Is price ~ priceBeforeDiscount * (1 - show_discount/100)?")
    tr = train.dropna(subset=[C.TARGET, "priceBeforeDiscount", "show_discount"]).copy()
    tr = tr[(tr["priceBeforeDiscount"] > 0)]
    if len(tr):
        recon = tr["priceBeforeDiscount"] * (1 - tr["show_discount"] / 100.0)
        rel_err = ((recon - tr[C.TARGET]).abs() / tr[C.TARGET].clip(lower=1))
        buf.write(
            f"- rows with pbd>0 & show_discount present: {len(tr):,}\n"
            f"- median relative error of discount-reconstruction: {rel_err.median():.4f}\n"
            f"- share within 1% of true price: {(rel_err < 0.01).mean()*100:.1f}%\n"
            f"- share with show_discount==0 among these: {(tr['show_discount']==0).mean()*100:.1f}%\n"
        )
    # raw_discount reconstruction
    tr2 = train.dropna(subset=[C.TARGET, "priceBeforeDiscount", "raw_discount"]).copy()
    tr2 = tr2[tr2["priceBeforeDiscount"] > 0]
    if len(tr2):
        recon2 = tr2["priceBeforeDiscount"] - tr2["raw_discount"]
        rel_err2 = ((recon2 - tr2[C.TARGET]).abs() / tr2[C.TARGET].clip(lower=1))
        buf.write(
            f"- price ~ priceBeforeDiscount - raw_discount: median rel err "
            f"{rel_err2.median():.4f}, within 1%: {(rel_err2 < 0.01).mean()*100:.1f}%\n"
        )

    # ---------------------------------------------------------------- #
    # 2. Price distribution & outliers
    # ---------------------------------------------------------------- #
    _section(buf, "2. Price distribution (train)")
    p = train[C.TARGET].dropna()
    desc = p.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(2)
    buf.write(_fmt_df(desc.to_frame()))
    buf.write(
        f"\n\n- zeros: {(p == 0).sum():,}  |  negatives: {(p < 0).sum():,}\n"
        f"- min: {p.min():,.0f}  max: {p.max():,.0f}\n"
        f"- log10 range: [{np.log10(p[p>0].min()):.2f}, {np.log10(p.max()):.2f}]\n"
    )

    # ---------------------------------------------------------------- #
    # 3. Entity history depth -> cold start
    # ---------------------------------------------------------------- #
    _section(buf, "3. Entity history depth in train (cold-start risk)")
    rows = []
    for col in ["modelId", "itemId", "shopId", "cat_id", "brand"]:
        if col not in train.columns:
            continue
        vc = train[col].value_counts()
        n_unique = vc.shape[0]
        # how many TEST entities are unseen in train?
        test_ids = set(test[col].dropna().unique())
        train_ids = set(train[col].dropna().unique())
        unseen = len(test_ids - train_ids)
        # history depth percentiles
        rows.append(
            {
                "col": col,
                "n_unique_train": n_unique,
                "median_hist_rows": int(vc.median()),
                "p90_hist_rows": int(vc.quantile(0.9)),
                "test_unique": len(test_ids),
                "test_unseen_in_train": unseen,
                "test_unseen_%": round(100 * unseen / max(len(test_ids), 1), 2),
            }
        )
    buf.write(_fmt_df(pd.DataFrame(rows)))

    # share of test BLANK rows whose modelId / itemId has >=1 train obs
    _section(buf, "3b. Coverage of blank test rows by train history")
    blank = test[~is_anchor]
    for col in ["modelId", "itemId", "shopId"]:
        train_ids = set(train[col].dropna().unique())
        cov = blank[col].isin(train_ids).mean() * 100
        buf.write(f"- blank rows with {col} seen in train: {cov:.2f}%\n")

    # ---------------------------------------------------------------- #
    # 4. Price volatility per modelId (how strong is last-known price?)
    # ---------------------------------------------------------------- #
    _section(buf, "4. Per-modelId price volatility in train")
    g = train.dropna(subset=[C.TARGET]).groupby("modelId")[C.TARGET]
    stats = g.agg(["count", "mean", "std", "min", "max"])
    multi = stats[stats["count"] >= 3].copy()
    multi["cv"] = (multi["std"] / multi["mean"]).replace([np.inf, -np.inf], np.nan)
    multi["range_ratio"] = multi["max"] / multi["min"].clip(lower=1)
    buf.write(
        f"- modelIds with >=3 obs: {len(multi):,}\n"
        f"- median coefficient of variation: {multi['cv'].median():.4f}\n"
        f"- share of modelIds with CV < 1% (essentially constant price): "
        f"{(multi['cv'] < 0.01).mean()*100:.1f}%\n"
        f"- share with CV < 5%: {(multi['cv'] < 0.05).mean()*100:.1f}%\n"
        f"- median max/min price ratio: {multi['range_ratio'].median():.4f}\n"
    )

    # ---------------------------------------------------------------- #
    # 5. Anchor representativeness
    # ---------------------------------------------------------------- #
    _section(buf, "5. Anchor representativeness (per test day)")
    test_day = test[C.TIME_COL].dt.date
    for day, sub in test.assign(_day=test_day).groupby("_day"):
        anc = sub[sub[C.TARGET].notna()]
        buf.write(
            f"\n**{day}** — anchors={len(anc)}, blank={len(sub)-len(anc)}\n"
            f"- anchor price median={anc[C.TARGET].median():,.0f}, "
            f"mean={anc[C.TARGET].mean():,.0f}\n"
            f"- anchor distinct shops={anc['shopId'].nunique()}, "
            f"distinct cats={anc['cat_id'].nunique()}\n"
        )

    out = C.REPORTS_DIR / "eda.md"
    out.write_text(buf.getvalue())
    print(f"[eda] wrote {out} ({len(buf.getvalue()):,} chars)")
    # Also echo to stdout for immediate inspection
    print(buf.getvalue())


if __name__ == "__main__":
    run()
