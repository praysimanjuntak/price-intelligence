"""Aggregate all backtest results into a single comparison table + plots.

Reads the per-phase JSON reports and produces:
  - reports/results_summary.md   (model comparison table + insights)
  - reports/fig_model_comparison.png
  - reports/fig_calibration.png
  - injects the same tables into README.md between <!-- AUTOGEN:* --> markers
    so the README can never drift from the actual run.

Usage:
    python -m src.results_summary
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C


def _load(name):
    p = C.REPORTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def build_table() -> pd.DataFrame:
    rows = []

    def add(name, tier, res):
        o = res["overall"]
        rows.append({
            "model": name, "tier": tier,
            "MAPE_%": o["mape"], "sMAPE_%": o["smape"], "MedAPE_%": o["medape"],
            "MAE_IDR": o["mae"], "RMSE_IDR": o["rmse"], "R2": o["r2"], "n": o["n"],
        })

    t1 = _load("tier1_backtest.json")
    if t1:
        add("LOCF (last price)", "baseline", t1["locf"])
        add("Tier1 Global CatBoost", "tier1", t1["global_catboost"])
        if "gated" in t1:
            add("Tier1 Gated (LOCF+CatBoost)", "tier1", t1["gated"])
        if "cb_global_cal" in t1:
            add("Tier1 CatBoost + global anchor cal", "tier1", t1["cb_global_cal"])
        if "cb_category_cal" in t1:
            add("Tier1 CatBoost + per-category anchor cal", "tier1", t1["cb_category_cal"])

    t2 = _load("tier2_backtest.json")
    if t2:
        add("Tier2 Hierarchical fallback", "tier2", t2["hier"])
        add("Tier2 Robust recent-K median", "tier2", t2["robust_recent"])
        add("Tier2 Hier + global anchor cal", "tier2", t2["cal_global"])
        add("Tier2 Hier + per-category anchor cal", "tier2", t2["cal_category"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["model"]).sort_values("MAPE_%")
    return df


def plot_comparison(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    d = df.sort_values("MAPE_%")
    colors = {"baseline": "#444", "tier1": "#1f77b4", "tier2": "#2ca02c"}
    ax.barh(d["model"], d["MAPE_%"], color=[colors.get(t, "#888") for t in d["tier"]])
    ax.set_xlabel("Backtest MAPE (%)  — lower is better")
    ax.set_title("Model comparison (5-day outage backtest)")
    ax.invert_yaxis()
    for i, v in enumerate(d["MAPE_%"]):
        ax.text(v, i, f" {v:.3f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "fig_model_comparison.png", dpi=130)
    plt.close(fig)


def plot_calibration():
    cal = _load("calibration.json")
    if not cal:
        return
    syn = cal["synthetic_shift"]["results"]
    real = cal["real_days"]
    strategies = ["none", "global", "category"]
    syn_mape = [syn[s]["mape"] for s in strategies]
    real_mape = [real[s]["overall"]["mape"] for s in strategies]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(strategies, real_mape, color="#2ca02c")
    axes[0].set_title("Real shared days (≈ no drift)")
    axes[0].set_ylabel("MAPE (%)")
    for i, v in enumerate(real_mape):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(strategies, syn_mape, color="#d62728")
    axes[1].set_title("Synthetic +15% global / +10% half-cats shift")
    axes[1].set_ylabel("MAPE (%)")
    for i, v in enumerate(syn_mape):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Anchor-set calibration: corrects day-level drift, harmless when none exists")
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "fig_calibration.png", dpi=130)
    plt.close(fig)


def _fmt_int(x):
    return f"{x:,.0f}"


def leaderboard_md(df: pd.DataFrame) -> str:
    """README leaderboard table (MAE, RMSE, MAPE, sMAPE, MedAPE, R2)."""
    lines = ["| model | tier | MAE (IDR) | RMSE (IDR) | MAPE % | sMAPE % | MedAPE % | R2 |",
             "|---|---|---|---|---|---|---|---|"]
    best = df["MAPE_%"].min()
    for _, r in df.iterrows():
        star = "**" if r["MAPE_%"] == best else ""
        lines.append(
            f"| {star}{r['model']}{star} | {r['tier']} | {_fmt_int(r['MAE_IDR'])} | "
            f"{_fmt_int(r['RMSE_IDR'])} | {star}{r['MAPE_%']:.3f}{star} | "
            f"{r['sMAPE_%']:.3f} | {r['MedAPE_%']:.3f} | {r['R2']:.4f} |"
        )
    return "\n".join(lines)


def headtohead_md(df: pd.DataFrame) -> str:
    """Best Tier 1 vs best Tier 2 configuration, side by side."""
    def best_of(tier):
        sub = df[df["tier"] == tier]
        return sub.loc[sub["MAPE_%"].idxmin()] if len(sub) else None

    t1, t2 = best_of("tier1"), best_of("tier2")
    lines = ["| | model | MAE (IDR) | RMSE (IDR) | MAPE % | MedAPE % | R2 |",
             "|---|---|---|---|---|---|---|"]
    for label, r in [("**Tier 1** (global)", t1), ("**Tier 2** (per-entity)", t2)]:
        if r is None:
            continue
        lines.append(
            f"| {label} | {r['model']} | {_fmt_int(r['MAE_IDR'])} | "
            f"{_fmt_int(r['RMSE_IDR'])} | {r['MAPE_%']:.3f} | "
            f"{r['MedAPE_%']:.3f} | {r['R2']:.4f} |"
        )
    return "\n".join(lines)


def calibration_md() -> str:
    cal = _load("calibration.json")
    if not cal:
        return "_calibration.json not found_"
    syn = cal["synthetic_shift"]
    lines = [
        f"Injected +{syn['injected_global_shift']*100:.0f}% platform-wide "
        f"(plus +{syn['injected_cat_extra']*100:.0f}% on half the categories); "
        f"recovered global factor **{syn.get('recovered_global_factor', float('nan')):.4f}** "
        f"(injected {syn['injected_global_shift']:.4f}).",
        "",
        "| strategy | MAPE % | MedAPE % | MAE (IDR) |",
        "|---|---|---|---|",
    ]
    for s in ["none", "global", "category"]:
        r = syn["results"][s]
        lines.append(f"| {s} | {r['mape']:.3f} | {r['medape']:.3f} | {_fmt_int(r['mae'])} |")
    return "\n".join(lines)


def inject_readme(df: pd.DataFrame) -> None:
    """Replace content between <!-- AUTOGEN:<key> START/END --> markers."""
    readme = C.ROOT / "README.md"
    if not readme.exists():
        print("[warn] README.md not found; skipping injection")
        return
    text = readme.read_text()
    blocks = {
        "leaderboard": leaderboard_md(df),
        "headtohead": headtohead_md(df),
        "calibration": calibration_md(),
    }
    for key, content in blocks.items():
        pattern = re.compile(
            rf"(<!-- AUTOGEN:{key} START -->\n).*?(\n<!-- AUTOGEN:{key} END -->)",
            re.DOTALL,
        )
        if not pattern.search(text):
            print(f"[warn] README marker AUTOGEN:{key} not found")
            continue
        text = pattern.sub(lambda m, c=content: m.group(1) + c + m.group(2), text)
    readme.write_text(text)
    print(f"[results] injected {len(blocks)} tables into README.md")


def main():
    df = build_table()
    plot_comparison(df)
    plot_calibration()
    inject_readme(df)

    lines = ["# Results Summary — Model Comparison\n",
             "5-day outage backtest on held-out training days "
             "(2025-03-18 .. 2025-03-22). Each day reveals 100 random anchors; "
             "all other prices are predicted from strict history.\n",
             df.to_markdown(index=False), ""]

    cal = _load("calibration.json")
    if cal:
        syn = cal["synthetic_shift"]
        lines += [
            "\n## Anchor calibration — synthetic shift stress test\n",
            f"Injected a known +{syn['injected_global_shift']*100:.0f}% platform-wide "
            f"price shift (plus +{syn['injected_cat_extra']*100:.0f}% on half the "
            "categories) on the held-out day, then recovered it from the 100 anchors.\n",
            f"- Recovered global factor: **{syn.get('recovered_global_factor', float('nan')):.4f}** "
            f"(injected {syn['injected_global_shift']:.4f})\n",
            "| strategy | MAPE % | MedAPE % | MAE (IDR) |",
            "|---|---|---|---|",
        ]
        for s in ["none", "global", "category"]:
            r = syn["results"][s]
            lines.append(f"| {s} | {r['mape']:.3f} | {r['medape']:.3f} | {r['mae']:,.0f} |")

    out = C.REPORTS_DIR / "results_summary.md"
    out.write_text("\n".join(lines))
    print(df.to_string(index=False))
    print(f"\nsaved {out}")
    print("saved reports/fig_model_comparison.png, reports/fig_calibration.png")


if __name__ == "__main__":
    main()
