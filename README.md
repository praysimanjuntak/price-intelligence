# MrScraper — Price Intelligence & Anomaly Detection

Reconstruct product prices on a day when scraping was unavailable, using a
large historical dataset plus a tiny **anchor set** of 100 manually collected
prices for that day.

**TL;DR result.** On a leakage-safe 5-day outage backtest, the winning model is
a hierarchical last-known-price predictor with **per-category anchor
calibration**: **MAPE 0.829%, MedAPE 0.000%, R2 0.991**. The 100 anchors are
used to detect and correct day-level / per-category price drift; a synthetic
stress test shows they recover a known +15% platform shift almost exactly
(15.4% -> 1.9% MAPE).

---

## Headline findings

1. **Price is near-deterministic per `modelId`.** 83% of modelIds have a single
   distinct price across the whole history; 99.6% have <= 3. Median
   coefficient of variation is ~0. So **last-observation-carried-forward
   (LOCF) per modelId is an extremely strong baseline** (backtest MAPE 0.94%).
2. **The cross-sectional signal is useless on its own.** A global-median
   predictor scores MAPE 186% — price cannot be inferred from features without
   entity identity.
3. **Blank test rows expose only 4 columns**: `capturedAt, shopId, itemId,
   modelId`. Every other column (including `priceBeforeDiscount`,
   `show_discount`, `cat_id`, `brand`) is blank at inference. So all features
   must be derived from **history keyed by those IDs** + the anchor set. We
   recover `cat_id`/`brand` per modelId (they are deterministic) for calibration.
4. **The anchor set is the only same-day signal.** A day-of price change is not
   predictable from history; the 100 anchors are what let us correct
   platform/category drift. This is where calibration adds real value.
5. **Flexible learners do not beat LOCF here.** The leakage-safe global
   CatBoost (1.94% raw, 1.71% gated) underperforms LOCF because it injects
   smoothing noise on the 83% of exactly-flat entities and still cannot predict
   unobserved step changes.

---

## Data

| Split | Rows | Days | Anchors |
|---|---|---|---|
| Train | 306,226 | 59 (2025-01-01 .. 2025-03-22) | n/a (all prices known) |
| Test (3 shared days) | 25,900 | 2025-03-22/23/24 | 100/day (300 total) |
| Test (full hidden file) | 76,255 | 16 | 100/day (1,600 total) |

Prices are in IDR smallest unit (range 100,000 .. 1.66e9; log10 5.0 .. 9.2),
heavy right tail. Cold-start is negligible: 99.99% of blank rows' modelIds
(100% of itemId/shopId) appear in train.

Download links live in `src/config.py` (`make download`).

---

## Approach

The blank rows give no same-day features, so the pipeline has two parts: a
**base predictor** built from history, and an **anchor calibration** layer that
applies the only same-day information available.

### Features (leakage-safe) — `src/features.py`
`EntityStats.fit(history)` learns per-entity price statistics from rows
strictly older than the prediction day, then `.transform()` attaches them via a
cold-start fallback chain `modelId -> itemId -> shopId -> cat_id -> global`:

- `last_price`, `median/mean/min/max`, `std`, `n_obs` per modelId/itemId/shopId
- `price_cv`, `price_range_ratio` (volatility), `days_since_last` (recency)
- recovered `cat_id`/`brand` per modelId (deterministic) for calibration grouping
- temporal features from `capturedAt` (dow, dom, month, hour, weekend). These
  use only the row timestamp; target-day price movement is learned from anchors.

### Tier 1 — Global Marketplace Model — `src/model_global.py`
A single CatBoost model trained on all history. It predicts the **log residual
against LOCF**, `log1p(price) - log1p(last_price)`, so capacity is spent on the
~17% of entities whose price actually moves. A **gated** variant trusts LOCF
exactly where the entity is flat (`CV < 1%`) and only uses the model on
volatile/sparse entities. SHAP top features: `price_cv`, `shopId`,
`log_last_price`, `t_month` — i.e. the model is largely re-deriving LOCF.

To avoid look-ahead leakage, CatBoost training features are built day by day:
for rows on day D, `EntityStats` is fitted only on rows strictly before D. The
validation and inference paths therefore follow the same temporal rule.

### Tier 2 — Shop / Product-level Model — `src/model_entity.py`
- **Hierarchical fallback**: per-entity LOCF with the cold-start chain above.
- **Per-entity + anchor calibration** (the winner, see below).
- **Robust recent-K median**: median of the last K prices per modelId (damps
  flash-sale spikes).

### Anchor calibration (the scored centerpiece) — `src/calibration.py`
From the 100 anchors we compute the log-ratio of true vs LOCF-predicted price
and estimate a multiplicative correction:

- **none** — no correction.
- **global** — one factor = `median(anchor_true / anchor_pred)` (log space).
- **category** — per `cat_id` factor, **shrunk** toward the global factor with
  weight `n/(n + 10)` to guard thin groups.

Because the 3 shared days have ~0 real drift, we also run a **synthetic-shift
stress test**: inject a known +15% platform-wide shift (plus +10% on half the
categories) on a held-out day and verify the anchors recover it.

---

## Validation methodology

Real test blank rows have no labels, so all error numbers come from a
**leakage-safe outage backtest** (`src/validation.py`): hold out the last 5
*training* days entirely. For each held-out day, reveal a random 100 rows as
the anchor set, hide all other prices, fit features on strict history (rows
before that day), predict, and score. We blank exactly the same columns the
real test blanks, so the backtest mirrors inference conditions. Metrics: MAE,
RMSE, MAPE, sMAPE, MedAPE, R2 (`src/metrics.py`). Seed = 42 everywhere.

---

## Results

5-day outage backtest (2025-03-18 .. 2025-03-22), 40,055 predicted rows.

Metrics are reported on the raw IDR price scale. **MAE** and **RMSE** are in
IDR; **MAPE / sMAPE / MedAPE** are percentages; **R²** is on raw price. The two
required tiers are labelled `tier1` (global) and `tier2` (per-shop/product).

The tables below are auto-generated from `reports/*.json` by
`python -m src.results_summary` — they are never edited by hand, so they can
never drift from the actual run.

**Leakage control.** Both tiers are evaluated in the same outage backtest:
features are fitted from strict pre-day history, target prices are hidden except
for the 100 anchors, and anchors calibrate only their own day. Tier 1 CatBoost
uses day-wise training features to avoid within-fold look-ahead.

<!-- AUTOGEN:leaderboard START -->
| model | tier | MAE (IDR) | RMSE (IDR) | MAPE % | sMAPE % | MedAPE % | R2 |
|---|---|---|---|---|---|---|---|
| **Tier2 Hier + per-category anchor cal** | tier2 | 241,666 | 8,033,049 | **0.829** | 0.408 | 0.000 | 0.9907 |
| LOCF (last price) | baseline | 247,996 | 8,067,211 | 0.939 | 0.417 | 0.000 | 0.9906 |
| Tier2 Hierarchical fallback | tier2 | 247,996 | 8,067,211 | 0.939 | 0.417 | 0.000 | 0.9906 |
| Tier2 Hier + global anchor cal | tier2 | 247,996 | 8,067,211 | 0.939 | 0.417 | 0.000 | 0.9906 |
| Tier2 Robust recent-K median | tier2 | 350,932 | 8,143,065 | 1.095 | 0.555 | 0.000 | 0.9904 |
| Tier1 Gated (LOCF+CatBoost) | tier1 | 338,766 | 8,768,850 | 1.712 | 0.488 | 0.000 | 0.9889 |
| Tier1 CatBoost + per-category anchor cal | tier1 | 403,791 | 8,721,888 | 1.824 | 0.684 | 0.098 | 0.9890 |
| Tier1 CatBoost + global anchor cal | tier1 | 412,449 | 8,776,903 | 1.921 | 0.699 | 0.103 | 0.9889 |
| Tier1 Global CatBoost | tier1 | 416,193 | 8,775,077 | 1.937 | 0.715 | 0.124 | 0.9889 |
<!-- AUTOGEN:leaderboard END -->

### Tier 1 (global) vs Tier 2 (per-entity) — head-to-head

The brief asks for metrics reported **separately** for the global model and the
per-shop/product model. Best configuration of each:

<!-- AUTOGEN:headtohead START -->
| | model | MAE (IDR) | RMSE (IDR) | MAPE % | MedAPE % | R2 |
|---|---|---|---|---|---|---|
| **Tier 1** (global) | Tier1 Gated (LOCF+CatBoost) | 338,766 | 8,768,850 | 1.712 | 0.000 | 0.9889 |
| **Tier 2** (per-entity) | Tier2 Hier + per-category anchor cal | 241,666 | 8,033,049 | 0.829 | 0.000 | 0.9907 |
<!-- AUTOGEN:headtohead END -->

Per-day breakdowns for both tiers live in `reports/tier1_backtest.json` and
`reports/tier2_backtest.json`.

**Anchor calibration — synthetic +15% shift stress test:**

<!-- AUTOGEN:calibration START -->
Injected +15% platform-wide (plus +10% on half the categories); recovered global factor **0.1500** (injected 0.1500).

| strategy | MAPE % | MedAPE % | MAE (IDR) |
|---|---|---|---|
| none | 15.437 | 13.043 | 5,837,339 |
| global | 2.752 | 0.000 | 1,539,979 |
| category | 1.914 | 0.000 | 1,188,950 |
<!-- AUTOGEN:calibration END -->

Plots: `reports/fig_model_comparison.png`, `reports/fig_calibration.png`.

### When does each approach win?
- **No-drift days** (the 3 shared days): LOCF and per-entity models are already
  near-perfect; global calibration correctly does nothing; per-category gives a
  small genuine gain (0.939% -> 0.829%) by catching minor category movement.
- **Drift days** (simulated outage with promo/currency shift): calibration is
  decisive — it cuts MAPE from 15.4% to 1.9%. This is the realistic production
  scenario the anchor set is designed for.
- **Tier 1 vs Tier 2**: Tier 2 (entity-level) is structurally better here
  because price is entity-specific and cross-sectional features carry almost no
  signal. Tier 1 is still useful as a global benchmark for sparse/cold-start
  settings, but the hierarchical fallback already covers that path better in
  this dataset.

---

## Reproducing

**Default workflow is CPU-safe.** Tier 1 CatBoost also supports GPU training if
you pass `TIER1_FLAGS=--gpu`.

The repo uses `uv` for dependency management with a pinned `uv.lock` for exact
reproduction. If you prefer pip, `requirements.txt` and `requirements-lock.txt`
are also provided.

```bash
# 1. environment (creates .venv with pinned versions)
uv sync --locked

# 2. data
make download                # train.csv + test_3days.csv from Google Drive
make prepare                 # dtype coercion + parquet cache + sanity report
# Expected output: train 306,226 rows, test 25,900 rows, 300 anchors (100/day)

# 3. analysis + models (all CPU-safe)
make eda                     # reports/eda.md (+ deeper checks)
make backtest                # LOCF / median baselines
make tier1                   # Tier 1 global CatBoost (CPU) + SHAP
make calib                   # anchor calibration + synthetic-shift test
make tier2                   # Tier 2 hierarchical entity models
make results                 # reports/results_summary.md + figures + README tables

# 4. produce the completed file
make infer                   # data/test_completed.csv (all prices filled)

# 5. one-shot (optional: a single command to run everything)
make all                     # download prepare eda backtest tier1 calib tier2 results infer
```

### CatBoost GPU Variant (optional)

If you have a CUDA GPU and want to train Tier 1 CatBoost on GPU:

```bash
make tier1 TIER1_FLAGS="--gpu"    # Tier 1 CatBoost with GPU
make results                       # update tables and figures
```

### Repository layout

```
src/
  config.py            paths, seeds, schema, Drive IDs
  data_io.py           download / coerce / parquet cache / sanity report
  eda.py, eda2.py      exploratory analysis
  metrics.py           MAE / RMSE / MAPE / sMAPE / MedAPE / R2
  features.py          leakage-safe EntityStats + fallback chain
  validation.py        outage-day backtest harness + LOCF baselines
  model_global.py      Tier 1 global CatBoost (+ gated) + SHAP
  model_entity.py      Tier 2 hierarchical entity models
  calibration.py       anchor calibration + synthetic-shift stress test
  infer.py             fill blank prices -> completed CSV
  results_summary.py   aggregate all results + plots
reports/               eda.md, *_backtest.json, results_summary.md, figures
models/                global_catboost.cbm
```

---

## Outliers, missing values, anomalies

- **Outliers**: prices span 5 orders of magnitude. All models work in log space;
  calibration factors use the **median** ratio (robust to flash-sale spikes);
  the robust recent-K variant damps spikes further. No rows are dropped.
- **Missing values**: blank test columns are reconstructed from history where
  deterministic (`cat_id`, `brand`); numeric gaps fall back through the entity
  chain to the global median.
- **Anomaly note**: the day-to-day stability of prices (83% flat) is itself the
  key anomaly — this dataset behaves like a slowly-updated catalog, not a
  volatile market, which is why LOCF dominates and the anchor set matters most
  precisely when that stability breaks (promotions, currency shifts).

## Limitations

- The 3 shared days contain **no real day-level drift**, so the live benefit of
  calibration on them is ~0; its value is demonstrated via the synthetic stress
  test and will show up on hidden days that contain promos/shifts.
- LOCF being this strong is partly a property of this snapshot; on a more
  volatile marketplace the learned models and calibration would matter more.
Reproducible: fixed seed (42), pinned `requirements.txt`, deterministic
feature construction.
