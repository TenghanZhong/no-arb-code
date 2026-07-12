from __future__ import annotations

import math
from typing import Dict

import numpy as np


EPS = 1e-12


def _finite_pair(actual: np.ndarray, pred: np.ndarray):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p)
    return a[ok], p[ok]


def regression_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    a, p = _finite_pair(actual, pred)
    if a.size == 0:
        return {"n": 0, "r2": math.nan, "rmse": math.nan, "mae": math.nan}
    err = p - a
    sse = float(np.sum(err**2))
    sst = float(np.sum((a - np.mean(a)) ** 2))
    r2 = math.nan if a.size < 2 or sst <= EPS else 1.0 - sse / sst
    return {
        "n": int(a.size),
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
    }


def error_metrics(actual: np.ndarray, pred: np.ndarray, scale: np.ndarray | float, *, floor: float) -> Dict[str, float]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    s = np.asarray(scale, dtype=float) + np.zeros_like(a, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p) & np.isfinite(s)
    if not np.any(ok):
        return {
            "n": 0,
            "rmse": math.nan,
            "mae": math.nan,
            "median_abs": math.nan,
            "p90_abs": math.nan,
            "rmse_scaled": math.nan,
            "mae_scaled": math.nan,
            "median_abs_scaled": math.nan,
            "p90_abs_scaled": math.nan,
            "gt_one_scale_rate": math.nan,
        }
    err_abs = np.abs(p[ok] - a[ok])
    denom = np.maximum(np.abs(s[ok]), float(floor))
    scaled = err_abs / denom
    return {
        "n": int(err_abs.size),
        "rmse": float(np.sqrt(np.mean(err_abs**2))),
        "mae": float(np.mean(err_abs)),
        "median_abs": float(np.median(err_abs)),
        "p90_abs": float(np.percentile(err_abs, 90)),
        "rmse_scaled": float(np.sqrt(np.mean(scaled**2))),
        "mae_scaled": float(np.mean(scaled)),
        "median_abs_scaled": float(np.median(scaled)),
        "p90_abs_scaled": float(np.percentile(scaled, 90)),
        "gt_one_scale_rate": float(np.mean(scaled > 1.0)),
    }


def outside_box_metrics(pred: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> Dict[str, float]:
    p = np.asarray(pred, dtype=float)
    b = np.asarray(bid, dtype=float)
    a = np.asarray(ask, dtype=float)
    ok = np.isfinite(p) & np.isfinite(b) & np.isfinite(a) & (a >= b)
    if not np.any(ok):
        return {
            "n": 0,
            "outside_rate": math.nan,
            "below_bid_rate": math.nan,
            "above_ask_rate": math.nan,
        }
    below = p[ok] < b[ok]
    above = p[ok] > a[ok]
    return {
        "n": int(np.sum(ok)),
        "outside_rate": float(np.mean(below | above)),
        "below_bid_rate": float(np.mean(below)),
        "above_ask_rate": float(np.mean(above)),
    }


def safe_qlike(actual_variance: np.ndarray, pred_variance: np.ndarray, *, floor: float = 1e-10) -> float:
    a, p = _finite_pair(actual_variance, pred_variance)
    if a.size == 0:
        return math.nan
    a = np.maximum(a, floor)
    p = np.maximum(p, floor)
    return float(np.mean(a / p + np.log(p)))
