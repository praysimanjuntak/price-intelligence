# Results Summary — Model Comparison

5-day outage backtest on held-out training days (2025-03-18 .. 2025-03-22). Each day reveals 100 random anchors; all other prices are predicted from strict history.

| model                                    | tier     |   MAPE_% |   sMAPE_% |   MedAPE_% |   MAE_IDR |    RMSE_IDR |       R2 |     n |
|:-----------------------------------------|:---------|---------:|----------:|-----------:|----------:|------------:|---------:|------:|
| Tier2 Hier + per-category anchor cal     | tier2    |   0.8286 |    0.4082 |     0      |    241666 | 8.03305e+06 | 0.990693 | 40055 |
| LOCF (last price)                        | baseline |   0.9385 |    0.417  |     0      |    247996 | 8.06721e+06 | 0.990614 | 40055 |
| Tier2 Hierarchical fallback              | tier2    |   0.9385 |    0.417  |     0      |    247996 | 8.06721e+06 | 0.990614 | 40055 |
| Tier2 Hier + global anchor cal           | tier2    |   0.9385 |    0.417  |     0      |    247996 | 8.06721e+06 | 0.990614 | 40055 |
| Tier2 Robust recent-K median             | tier2    |   1.0953 |    0.5549 |     0      |    350932 | 8.14307e+06 | 0.990437 | 40055 |
| Tier1 Gated (LOCF+CatBoost)              | tier1    |   1.7124 |    0.4881 |     0      |    338766 | 8.76885e+06 | 0.98891  | 40055 |
| Tier1 CatBoost + per-category anchor cal | tier1    |   1.8241 |    0.6841 |     0.0977 |    403791 | 8.72189e+06 | 0.989029 | 40055 |
| Tier1 CatBoost + global anchor cal       | tier1    |   1.9206 |    0.6986 |     0.1027 |    412449 | 8.7769e+06  | 0.98889  | 40055 |
| Tier1 Global CatBoost                    | tier1    |   1.9369 |    0.7148 |     0.124  |    416193 | 8.77508e+06 | 0.988895 | 40055 |


## Anchor calibration — synthetic shift stress test

Injected a known +15% platform-wide price shift (plus +10% on half the categories) on the held-out day, then recovered it from the 100 anchors.

- Recovered global factor: **0.1500** (injected 0.1500)

| strategy | MAPE % | MedAPE % | MAE (IDR) |
|---|---|---|---|
| none | 15.437 | 13.043 | 5,837,339 |
| global | 2.752 | 0.000 | 1,539,979 |
| category | 1.914 | 0.000 | 1,188,950 |