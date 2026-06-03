"""Leakage-safe feature engineering.

At inference time the blank TEST rows expose only four columns:
    capturedAt, shopId, itemId, modelId
Everything else (priceBeforeDiscount, show_discount, cat_id, brand, ...) is
blank. Therefore every predictive feature must be derived from HISTORICAL
data keyed by those IDs, plus temporal signals from capturedAt, plus the
anchor set (handled separately in calibration.py).

`EntityStats.fit(history)` learns per-entity price statistics from rows with
known prices that are strictly older than the prediction day. `.transform(df)`
attaches those statistics to the rows we need to predict, using a fallback
chain modelId -> itemId -> shopId -> cat_id -> global for cold-start safety.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

EPS = 1.0  # IDR; prices are large integers, avoids div-by-zero
_NS_PER_DAY = 86_400_000_000_000


def _to_epoch_days(s: pd.Series) -> pd.Series:
    """Convert a datetime Series to integer epoch-days (pandas 2/3 safe)."""
    ns = s.astype("int64")  # nanoseconds since epoch; NaT -> large negative, masked later
    return (ns // _NS_PER_DAY).astype("int64")


# --------------------------------------------------------------------------- #
# Temporal features (available for any row with a timestamp, no leakage)
# --------------------------------------------------------------------------- #
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df[C.TIME_COL]
    out = pd.DataFrame(index=df.index)
    out["t_dow"] = t.dt.dayofweek
    out["t_dom"] = t.dt.day
    out["t_month"] = t.dt.month
    out["t_hour"] = t.dt.hour
    out["t_is_weekend"] = (t.dt.dayofweek >= 5).astype("int8")
    out["t_epoch_days"] = _to_epoch_days(t)
    return out


class EntityStats:
    """Fit per-entity price statistics from a history slice; attach to targets."""

    def __init__(self) -> None:
        self.model_stats: pd.DataFrame | None = None
        self.item_stats: pd.DataFrame | None = None
        self.shop_stats: pd.DataFrame | None = None
        self.cat_stats: pd.DataFrame | None = None
        self.model_attr: pd.DataFrame | None = None  # cat_id, brand per modelId
        self.item_attr: pd.DataFrame | None = None
        self.global_median: float = np.nan
        self.global_last_epoch: int = 0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _agg(df: pd.DataFrame, key: str) -> pd.DataFrame:
        d = df.dropna(subset=[C.TARGET]).sort_values(C.TIME_COL)
        g = d.groupby(key)
        out = g.agg(
            **{
                f"{key}__last_price": (C.TARGET, "last"),
                f"{key}__median_price": (C.TARGET, "median"),
                f"{key}__mean_price": (C.TARGET, "mean"),
                f"{key}__std_price": (C.TARGET, "std"),
                f"{key}__min_price": (C.TARGET, "min"),
                f"{key}__max_price": (C.TARGET, "max"),
                f"{key}__n_obs": (C.TARGET, "size"),
                f"{key}__last_epoch": (C.TIME_COL, "last"),
            }
        )
        out[f"{key}__last_epoch"] = _to_epoch_days(out[f"{key}__last_epoch"])
        return out

    def fit(self, history: pd.DataFrame) -> "EntityStats":
        h = history.dropna(subset=[C.TARGET]).copy()
        self.model_stats = self._agg(h, "modelId")
        self.item_stats = self._agg(h, "itemId")
        self.shop_stats = self._agg(h, "shopId")
        if "cat_id" in h.columns:
            self.cat_stats = self._agg(h.dropna(subset=["cat_id"]), "cat_id")

        # Deterministic attributes recovered per modelId / itemId (mode).
        def _mode_map(df, key, attr):
            sub = df.dropna(subset=[attr])
            if sub.empty:
                return pd.DataFrame(columns=[attr]).rename_axis(key)
            return (
                sub.groupby(key)[attr]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
                .to_frame()
            )

        attrs = [a for a in ["cat_id", "brand"] if a in h.columns]
        if attrs:
            self.model_attr = pd.concat(
                [_mode_map(h, "modelId", a) for a in attrs], axis=1
            )
            self.item_attr = pd.concat(
                [_mode_map(h, "itemId", a) for a in attrs], axis=1
            )
        self.global_median = float(h[C.TARGET].median())
        self.global_last_epoch = int(_to_epoch_days(pd.Series([h[C.TIME_COL].max()])).iloc[0])
        return self

    # ------------------------------------------------------------------ #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a numeric feature matrix aligned to df.index."""
        out = df[["modelId", "itemId", "shopId"]].copy()

        # Recover cat_id / brand for blank rows from the modelId attribute map.
        if self.model_attr is not None:
            rec = df[["modelId"]].merge(
                self.model_attr, on="modelId", how="left"
            )
            for a in self.model_attr.columns:
                out[a] = rec[a].to_numpy()
        # itemId fallback for cat/brand
        if self.item_attr is not None:
            rec_i = df[["itemId"]].merge(self.item_attr, on="itemId", how="left")
            for a in self.item_attr.columns:
                if a in out.columns:
                    out[a] = out[a].fillna(pd.Series(rec_i[a].to_numpy(), index=out.index))

        # Merge entity stats
        out = out.merge(self.model_stats, on="modelId", how="left")
        out = out.merge(self.item_stats, on="itemId", how="left")
        out = out.merge(self.shop_stats, on="shopId", how="left")
        if self.cat_stats is not None and "cat_id" in out.columns:
            out = out.merge(self.cat_stats, on="cat_id", how="left")

        # Fallback chain for the primary predictor: last/median price.
        def _fallback(*cols):
            res = pd.Series(np.nan, index=out.index)
            for c in cols:
                if c in out.columns:
                    res = res.fillna(out[c])
            return res.fillna(self.global_median)

        out["last_price"] = _fallback(
            "modelId__last_price", "itemId__last_price",
            "shopId__last_price", "cat_id__median_price",
        )
        out["median_price"] = _fallback(
            "modelId__median_price", "itemId__median_price",
            "shopId__median_price", "cat_id__median_price",
        )

        # Recency: days since the entity's last known observation.
        epoch = _to_epoch_days(df[C.TIME_COL]).to_numpy()
        last_epoch = out.get("modelId__last_epoch")
        if last_epoch is None:
            last_epoch = pd.Series(self.global_last_epoch, index=out.index)
        out["days_since_last"] = (epoch - last_epoch.fillna(self.global_last_epoch)).clip(lower=0)

        # Derived signals
        out["price_cv"] = (out["modelId__std_price"] / out["modelId__mean_price"]).replace(
            [np.inf, -np.inf], np.nan
        )
        out["price_range_ratio"] = (
            out["modelId__max_price"] / out["modelId__min_price"].clip(lower=EPS)
        )
        out["log_last_price"] = np.log1p(out["last_price"].clip(lower=0))
        out["log_median_price"] = np.log1p(out["median_price"].clip(lower=0))
        out["model_n_obs"] = out.get("modelId__n_obs", pd.Series(0, index=out.index)).fillna(0)
        out["item_n_obs"] = out.get("itemId__n_obs", pd.Series(0, index=out.index)).fillna(0)
        out["shop_n_obs"] = out.get("shopId__n_obs", pd.Series(0, index=out.index)).fillna(0)
        out["is_cold_model"] = out["model_n_obs"].eq(0).astype("int8")

        # Time features
        tf = add_time_features(df)
        out = pd.concat([out.reset_index(drop=True), tf.reset_index(drop=True)], axis=1)
        out.index = df.index
        return out


# Columns intended as model inputs (numeric). Categorical handled by CatBoost.
NUMERIC_FEATURES = [
    "last_price", "median_price", "log_last_price", "log_median_price",
    "modelId__mean_price", "modelId__std_price", "modelId__min_price",
    "modelId__max_price", "itemId__median_price", "shopId__median_price",
    "days_since_last", "price_cv", "price_range_ratio",
    "model_n_obs", "item_n_obs", "shop_n_obs", "is_cold_model",
    "t_dow", "t_dom", "t_month", "t_hour", "t_is_weekend",
]
CATEGORICAL_FEATURES = ["shopId", "cat_id", "brand"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cols += [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return cols
