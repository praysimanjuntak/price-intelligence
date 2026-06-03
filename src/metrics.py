"""Regression metrics for price prediction.

All metrics operate on the raw price scale (IDR). We report MAE, RMSE,
MAPE, sMAPE and median absolute percentage error (more robust to the
heavy right tail of prices).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


def _arr(x):
    return np.asarray(x, dtype=float)


def mae(y, yhat):
    return float(np.mean(np.abs(_arr(yhat) - _arr(y))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((_arr(yhat) - _arr(y)) ** 2)))


def mape(y, yhat):
    y = _arr(y)
    return float(np.mean(np.abs(_arr(yhat) - y) / np.clip(np.abs(y), 1, None)) * 100)


def smape(y, yhat):
    y, yhat = _arr(y), _arr(yhat)
    denom = np.clip(np.abs(y) + np.abs(yhat), 1, None)
    return float(np.mean(2 * np.abs(yhat - y) / denom) * 100)


def medape(y, yhat):
    y = _arr(y)
    return float(np.median(np.abs(_arr(yhat) - y) / np.clip(np.abs(y), 1, None)) * 100)


def r2(y, yhat):
    y, yhat = _arr(y), _arr(yhat)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


@dataclass
class Metrics:
    n: int
    mae: float
    rmse: float
    mape: float
    smape: float
    medape: float
    r2: float

    def as_dict(self):
        return asdict(self)


def evaluate(y, yhat) -> Metrics:
    y, yhat = _arr(y), _arr(yhat)
    mask = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[mask], yhat[mask]
    return Metrics(
        n=int(len(y)),
        mae=round(mae(y, yhat), 2),
        rmse=round(rmse(y, yhat), 2),
        mape=round(mape(y, yhat), 4),
        smape=round(smape(y, yhat), 4),
        medape=round(medape(y, yhat), 4),
        r2=round(r2(y, yhat), 6),
    )
