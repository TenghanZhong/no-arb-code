#!/usr/bin/env python3
"""Repair-risk analysis with an optional learned target."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass, asdict, replace
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize, least_squares
from scipy.stats import norm

_COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
from full_strike_constraints import (  # noqa: E402
    build_full_constraint_matrix,
    certify_call_curve,
    effective_bounds,
    feasibility_certificate,
    scipy_linear_constraint,
)
try:
    from scipy.interpolate import UnivariateSpline
except Exception:  # pragma: no cover
    UnivariateSpline = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

SECONDS_PER_DAY = 86400
SECONDS_PER_YEAR = 365.0 * SECONDS_PER_DAY
EPS = 1e-12


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RepairConfig:
    """Configuration for the matched repair analysis."""
    currency: str = "BTC"
    snapshot_hour: int = 12

    # Chains repaired at time t.
    dte_min_repair: float = 2.0
    dte_max_repair: float = 30.0

    # Wider DTE range for t+1 marking.
    dte_min_marking: float = 0.0
    dte_max_marking: float = 31.0
    marking_extra_days: int = 1

    price_floor: float = 0.0005
    strike_tol: float = 0.02
    min_strikes: int = 7
    elastic_penalty: float = 1000.0
    elastic_box: bool = False
    qp_backend: str = "cvxpy"
    qp_solver: str = "auto"
    smooth_rho: float = 1e-2
    smooth_rhos: Tuple[float, ...] = ()
    budget_smoothing: Tuple[float, ...] = ()
    spline_smoothing: float = 1e-4
    filter_spread_quantile: float = 0.75
    filter_bad_weight: float = 0.01
    svi_max_nfev: int = 2000
    weight_clip_q_low: float = 0.05
    weight_clip_q_high: float = 0.95
    arms: Tuple[str, ...] = ("L1", "L2", "SMOOTH")
    solver_maxiter: int = 800
    linprog_method: str = "highs"
    pnl_scale_floor_usd: float = 1.0
    hedge_tc_bps: float = 0.0
    wide_quantile: float = 0.75
    sparse_quantile: float = 0.75
    wing_abs_k: float = 0.05
    short_dte_max: float = 7.0
    gamma_material_threshold: float = 0.05
    gamma_rr_threshold: float = 0.20
    price_diff_box_threshold: float = 0.25

    # Slice-ranking score.
    ranking_score_method: str = "relspread_h2"
    ranking_slice_agg: str = "max"
    max_slices: Optional[int] = None
    selected_slices: Optional[int] = None
    slice_selection: str = "top"
    random_state: int = 7
    require_all_arms_for_main: bool = True


# =============================================================================
# Generic helpers
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def finite(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def to_epoch_seconds(series: pd.Series) -> pd.Series:
    """Convert epoch-like or datetime-like series to int64 epoch seconds."""
    if np.issubdtype(series.dtype, np.number):
        s = pd.to_numeric(series, errors="coerce")
        med = np.nanmedian(s.to_numpy(dtype=float))
        if med > 1e17:      # ns
            s = s / 1e9
        elif med > 1e14:    # us
            s = s / 1e6
        elif med > 1e11:    # ms
            s = s / 1e3
        return s.round().astype("Int64")
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return (dt.view("int64") // 10**9).astype("Int64")


def month_mask(epoch_seconds: pd.Series, months: Sequence[str]) -> pd.Series:
    if not months:
        return pd.Series(True, index=epoch_seconds.index)
    dt = pd.to_datetime(epoch_seconds.astype("int64"), unit="s", utc=True)
    return dt.dt.strftime("%Y-%m").isin(set(months))


def month_mask_with_extra_days(epoch_seconds: pd.Series, months: Sequence[str], extra_days: int = 0) -> pd.Series:
    """Select months and optional post-month-end days for t+1 marking."""
    if not months:
        return pd.Series(True, index=epoch_seconds.index)
    dt = pd.to_datetime(epoch_seconds.astype("int64"), unit="s", utc=True)
    mask = pd.Series(False, index=epoch_seconds.index)
    for m in months:
        start = pd.Timestamp(f"{m}-01", tz="UTC")
        end = start + pd.offsets.MonthBegin(1) + pd.Timedelta(days=int(extra_days))
        mask |= (dt >= start) & (dt < end)
    return mask


def dte_bucket_from_dte(dte: pd.Series) -> pd.Series:
    bins = [-np.inf, 1, 7, 14, 30, 45, np.inf]
    labels = ["0-1", "2-7", "8-14", "15-30", "31-45", "46+"]
    return pd.cut(dte, bins=bins, labels=labels).astype(str)


def infer_iv_decimal(iv: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(iv), errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(arr)
    if np.isfinite(med) and med > 3.0:
        arr = arr / 100.0
    return arr


# =============================================================================
# Deribit inverse Black-Scholes helpers
# =============================================================================

def deribit_call_base_price_from_iv(F: np.ndarray, K: np.ndarray, tau: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Normalized/base-currency call premium.

    c = N(d1) - (K/F)N(d2),
    d1 = [ln(F/K) + .5 sigma^2 tau] / [sigma sqrt(tau)].
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    out = np.full(np.broadcast_shapes(F.shape, K.shape, tau.shape, sigma.shape), np.nan, dtype=float)
    F, K, tau, sigma = np.broadcast_arrays(F, K, tau, sigma)
    valid = (F > 0) & (K > 0) & (tau > 0) & (sigma > 0)
    if not np.any(valid):
        return out
    vol_sqrt = sigma[valid] * np.sqrt(tau[valid])
    d1 = (np.log(F[valid] / K[valid]) + 0.5 * sigma[valid] ** 2 * tau[valid]) / vol_sqrt
    d2 = d1 - vol_sqrt
    out[valid] = norm.cdf(d1) - (K[valid] / F[valid]) * norm.cdf(d2)
    return out


def deribit_put_base_price_from_iv(F: np.ndarray, K: np.ndarray, tau: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    call = deribit_call_base_price_from_iv(F, K, tau, sigma)
    return call - 1.0 + np.asarray(K, dtype=float) / np.asarray(F, dtype=float)


def implied_vol_from_base_call_scalar(c: float, F: float, K: float, tau: float,
                                      lo: float = 1e-5, hi: float = 5.0,
                                      max_iter: int = 80) -> float:
    """Invert normalized Deribit call premium to IV by bisection."""
    if not all(np.isfinite(v) for v in [c, F, K, tau]) or F <= 0 or K <= 0 or tau <= 0:
        return np.nan
    intrinsic = max(1.0 - K / F, 0.0)
    c = min(max(float(c), intrinsic + 1e-12), 1.0 - 1e-12)

    def price(sig: float) -> float:
        return float(deribit_call_base_price_from_iv(np.array([F]), np.array([K]), np.array([tau]), np.array([sig]))[0])

    p_lo = price(lo)
    p_hi = price(hi)
    if c <= p_lo + 1e-12:
        return lo
    while c > p_hi and hi < 20.0:
        hi *= 2.0
        p_hi = price(hi)
    if c > p_hi + 1e-8:
        return np.nan
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if price(mid) < c:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def usd_spot_greeks_from_iv(claim: str, X: float, F: float, K: float, tau: float, sigma: float) -> Tuple[float, float, float]:
    """USD-premium spot delta, gamma, and vega under fixed carry."""
    if not all(np.isfinite(v) for v in [X, F, K, tau, sigma]) or X <= 0 or F <= 0 or K <= 0 or tau <= 0 or sigma <= 0:
        return np.nan, np.nan, np.nan
    vol_sqrt = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * tau) / vol_sqrt
    delta_call = norm.cdf(d1)
    gamma = norm.pdf(d1) / (X * sigma * math.sqrt(tau))
    vega = X * norm.pdf(d1) * math.sqrt(tau)
    if str(claim).upper().startswith("P"):
        return delta_call - 1.0, gamma, vega
    return delta_call, gamma, vega


# =============================================================================
# Panel loading and state variables
# =============================================================================

def load_synthetic_panel(path: str | Path, cfg: RepairConfig, months: Sequence[str],
                         *, dte_min: float, dte_max: float,
                         extra_days: int = 0) -> pd.DataFrame:
    """Load a call-equivalent quote panel for a specified DTE window."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    required = [
        "currency", "date", "expiration_date", "strike", "forward_price", "underlyer_spot",
        "tau", "DTE", "k", "C_bid", "C_ask", "C_mid", "intrinsic"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Synthetic call panel missing required columns: {missing}")

    out = df.copy()
    out["date"] = to_epoch_seconds(out["date"]).astype("int64")
    out["expiration_date"] = to_epoch_seconds(out["expiration_date"]).astype("int64")
    out["currency"] = out["currency"].astype(str).str.upper()
    out = out[out["currency"] == cfg.currency.upper()].copy()
    out = out[out["date"] % SECONDS_PER_DAY == cfg.snapshot_hour * 3600].copy()
    if months:
        out = out[month_mask_with_extra_days(out["date"], months, extra_days=extra_days)].copy()
    out = out[(out["DTE"] > dte_min) & (out["DTE"] <= dte_max)].copy()
    out = out[np.isfinite(out["C_bid"]) & np.isfinite(out["C_ask"]) & np.isfinite(out["C_mid"])].copy()
    out = out[(out["C_ask"] >= out["C_bid"]) & (out["C_ask"] > 0)].copy()
    if "dte_bucket" not in out.columns:
        out["dte_bucket"] = dte_bucket_from_dte(out["DTE"])
    else:
        out["dte_bucket"] = out["dte_bucket"].astype(str)
    out["relspread_floor"] = (out["C_ask"] - out["C_bid"]) / np.maximum(out["C_mid"].abs(), cfg.price_floor)
    return out.reset_index(drop=True)


def add_state_flags_base(panel: pd.DataFrame, cfg: RepairConfig) -> pd.DataFrame:
    """Compute slice-ranking state variables on the base panel."""
    df = panel.copy()
    if "relspread_floor" not in df.columns:
        df["relspread_floor"] = (df["C_ask"] - df["C_bid"]) / np.maximum(df["C_mid"].abs(), cfg.price_floor)
    if "dte_bucket" not in df.columns:
        df["dte_bucket"] = dte_bucket_from_dte(df["DTE"])
    df = df.sort_values(["currency", "date", "expiration_date", "strike"]).reset_index(drop=True)
    df["h_abs"] = np.nan
    df["h_norm"] = np.nan
    df["edge_strike"] = False

    for _, idx in df.groupby(["currency", "date", "expiration_date"], sort=False).groups.items():
        loc = np.asarray(list(idx))
        strikes = df.loc[loc, "strike"].to_numpy(dtype=float)
        F = float(df.loc[loc[0], "forward_price"])
        if len(strikes) >= 2:
            h = np.full(len(strikes), np.nan)
            h[0] = strikes[1] - strikes[0]
            h[-1] = strikes[-1] - strikes[-2]
            if len(strikes) >= 3:
                h[1:-1] = 0.5 * (strikes[2:] - strikes[:-2])
            df.loc[loc, "h_abs"] = h
            df.loc[loc, "h_norm"] = h / F if F > 0 else np.nan
            edge = np.zeros(len(strikes), dtype=bool)
            edge[0] = True
            edge[-1] = True
            df.loc[loc, "edge_strike"] = edge

    df["short_dte"] = df["DTE"] <= cfg.short_dte_max
    df["wing"] = df["k"].abs() >= cfg.wing_abs_k

    df["wide_spread"] = False
    df["sparse_strike"] = False
    global_wide = np.nanquantile(df["relspread_floor"], cfg.wide_quantile) if len(df) else np.nan
    global_sparse = np.nanquantile(df["h_norm"], cfg.sparse_quantile) if len(df) else np.nan
    for _, idx in df.groupby("dte_bucket", sort=False).groups.items():
        loc = np.asarray(list(idx))
        rs = df.loc[loc, "relspread_floor"].to_numpy(dtype=float)
        hs = df.loc[loc, "h_norm"].to_numpy(dtype=float)
        wide_thr = np.nanquantile(rs, cfg.wide_quantile) if np.isfinite(rs).sum() >= 20 else global_wide
        sparse_thr = np.nanquantile(hs, cfg.sparse_quantile) if np.isfinite(hs).sum() >= 20 else global_sparse
        df.loc[loc, "wide_spread"] = df.loc[loc, "relspread_floor"] >= wide_thr
        df.loc[loc, "sparse_strike"] = df.loc[loc, "h_norm"] >= sparse_thr

    box_width = (df["C_ask"] - df["C_bid"]).astype(float)
    h_abs = df["h_abs"].astype(float).replace(0.0, np.nan)
    h_norm = df["h_norm"].astype(float).replace(0.0, np.nan)
    df["ranking_score_abs_h2"] = box_width / np.maximum(h_abs ** 2, EPS)
    df["ranking_score_rel_h2"] = df["relspread_floor"] / np.maximum(h_norm ** 2, EPS)
    method = cfg.ranking_score_method.lower()
    if method == "abs_spread_h2":
        df["ranking_score"] = df["ranking_score_abs_h2"]
    else:
        df["ranking_score"] = df["ranking_score_rel_h2"]

    # Threshold-based spread-and-spacing screen used as a comparison.
    df["threshold_screen_cell"] = df["short_dte"] & df["wing"] & df["wide_spread"] & df["sparse_strike"]
    score_thr = np.nanquantile(df["ranking_score"].replace([np.inf, -np.inf], np.nan), 0.90) if len(df) else np.nan
    df["high_score_cell"] = df["ranking_score"] >= score_thr
    return df


def slice_key_cols() -> List[str]:
    return ["currency", "date", "expiration_date"]


def select_repair_slices(panel_full: pd.DataFrame, cfg: RepairConfig) -> pd.DataFrame:
    """Select repair slices by the configured spread-over-mesh score."""
    df = panel_full.copy()
    grp = df.groupby(slice_key_cols(), sort=False)
    agg_func = cfg.ranking_slice_agg.lower()
    if agg_func == "median":
        score_agg = ("ranking_score", "median")
    elif agg_func == "p90":
        score_agg = ("ranking_score", lambda x: np.nanpercentile(x.replace([np.inf, -np.inf], np.nan), 90))
    else:
        score_agg = ("ranking_score", "max")
    slice_stats = grp.agg(
        n_strikes=("strike", "size"),
        ranking_score=score_agg,
        threshold_screen_obs=("threshold_screen_cell", "sum"),
        max_relspread_floor=("relspread_floor", "max"),
        max_h_norm=("h_norm", "max"),
        min_h_norm=("h_norm", "min"),
        med_dte=("DTE", "median"),
        med_abs_k=("k", lambda x: float(np.nanmedian(np.abs(x))))
    ).reset_index()
    slice_stats = slice_stats[slice_stats["n_strikes"] >= cfg.min_strikes].copy()
    if slice_stats.empty:
        return df.iloc[0:0].copy()

    rng = np.random.default_rng(cfg.random_state)
    if cfg.selected_slices is not None:
        slice_stats = slice_stats.sort_values("ranking_score", ascending=False).head(cfg.selected_slices)
    elif cfg.max_slices is not None:
        if len(slice_stats) > cfg.max_slices:
            take = rng.choice(slice_stats.index.to_numpy(), size=cfg.max_slices, replace=False)
            slice_stats = slice_stats.loc[take]

    keys = set(map(tuple, slice_stats[slice_key_cols()].to_numpy()))
    mask = df[slice_key_cols()].apply(lambda r: tuple(r) in keys, axis=1)
    selected = df[mask].copy().sort_values(slice_key_cols() + ["strike"]).reset_index(drop=True)
    selected = selected.merge(slice_stats[slice_key_cols() + ["ranking_score", "threshold_screen_obs"]].rename(columns={"ranking_score":"slice_ranking_score"}),
                              on=slice_key_cols(), how="left")
    return selected


# =============================================================================
# Derivatives and admissible constraints
# =============================================================================

def first_second_derivatives_nonuniform(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return first and second derivative on a nonuniform grid.

    Interior formulas are three-point nonuniform finite differences. Edges use one-sided
    first derivative and NaN second derivative.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    d1 = np.full(n, np.nan)
    d2 = np.full(n, np.nan)
    if n < 2:
        return d1, d2
    d1[0] = (y[1] - y[0]) / (x[1] - x[0]) if x[1] != x[0] else np.nan
    d1[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2]) if x[-1] != x[-2] else np.nan
    for i in range(1, n - 1):
        hm = x[i] - x[i - 1]
        hp = x[i + 1] - x[i]
        if hm <= 0 or hp <= 0:
            continue
        d1[i] = (hm * hm * (y[i + 1] - y[i]) + hp * hp * (y[i] - y[i - 1])) / (hm * hp * (hm + hp))
        d2[i] = 2.0 / (hm + hp) * ((y[i + 1] - y[i]) / hp - (y[i] - y[i - 1]) / hm)
    return d1, d2


def convex_rows_for_c(strikes: np.ndarray, n_total: int, offset: int = 0) -> List[np.ndarray]:
    """Rows for slope_left - slope_right <= 0."""
    rows = []
    n = len(strikes)
    for i in range(1, n - 1):
        hm = strikes[i] - strikes[i - 1]
        hp = strikes[i + 1] - strikes[i]
        if hm <= 0 or hp <= 0:
            continue
        row = np.zeros(n_total)
        row[offset + i - 1] = -1.0 / hm
        row[offset + i] = 1.0 / hm + 1.0 / hp
        row[offset + i + 1] = -1.0 / hp
        rows.append(row)
    return rows


def monotone_rows_for_c(n: int, n_total: int, offset: int = 0) -> List[np.ndarray]:
    """Rows for -c_i + c_{i+1} <= 0, i.e. c_i >= c_{i+1}."""
    rows = []
    for i in range(n - 1):
        row = np.zeros(n_total)
        row[offset + i] = -1.0
        row[offset + i + 1] = 1.0
        rows.append(row)
    return rows


def second_diff_matrix(x: np.ndarray) -> np.ndarray:
    """Matrix D such that D @ c gives nonuniform second differences on interior points."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    rows = []
    for i in range(1, n - 1):
        hm = x[i] - x[i - 1]
        hp = x[i + 1] - x[i]
        if hm <= 0 or hp <= 0:
            continue
        row = np.zeros(n)
        row[i - 1] = 2.0 / (hm + hp) * (1.0 / hm)
        row[i] = 2.0 / (hm + hp) * (-1.0 / hp - 1.0 / hm)
        row[i + 1] = 2.0 / (hm + hp) * (1.0 / hp)
        rows.append(row)
    return np.asarray(rows) if rows else np.zeros((0, n))


def verify_solution(c: np.ndarray, bid: np.ndarray, ask: np.ndarray, intrinsic: np.ndarray, strikes: np.ndarray,
                    tol: float = 2e-6) -> Dict[str, float | bool]:
    c = np.asarray(c, dtype=float)
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    intrinsic = np.asarray(intrinsic, dtype=float)
    mono_v = np.nanmax(np.maximum(0.0, c[1:] - c[:-1])) if len(c) >= 2 else 0.0
    conv_v = 0.0
    if len(c) >= 3:
        vals = []
        for i in range(1, len(c) - 1):
            hm = strikes[i] - strikes[i - 1]
            hp = strikes[i + 1] - strikes[i]
            if hm > 0 and hp > 0:
                vals.append(max(0.0, (c[i] - c[i - 1]) / hm - (c[i + 1] - c[i]) / hp))
        conv_v = max(vals) if vals else 0.0
    intr_v = np.nanmax(np.maximum(0.0, intrinsic - c)) if len(c) else 0.0
    upper_v = np.nanmax(np.maximum(0.0, c - 1.0)) if len(c) else 0.0
    box_low = np.maximum(0.0, bid - c)
    box_high = np.maximum(0.0, c - ask)
    scale = np.maximum(ask - bid, 0.0005)
    box_slack_sum = float(np.nansum((box_low + box_high) / scale))
    feasible = bool((mono_v <= tol) and (conv_v <= tol) and (intr_v <= tol) and (upper_v <= tol))
    return {
        "verified_feasible": feasible,
        "max_mono_violation": float(mono_v),
        "max_conv_violation": float(conv_v),
        "max_intrinsic_violation": float(intr_v),
        "max_upper_violation": float(upper_v),
        "box_slack_sum": box_slack_sum,
        "max_box_low_violation": float(np.nanmax(box_low) if len(box_low) else 0.0),
        "max_box_high_violation": float(np.nanmax(box_high) if len(box_high) else 0.0),
    }


# =============================================================================
# Repair solvers
# =============================================================================

def prepare_slice_arrays(slc: pd.DataFrame, cfg: RepairConfig) -> Dict[str, np.ndarray]:
    slc = slc.sort_values("strike")
    K = slc["strike"].to_numpy(dtype=float)
    F = slc["forward_price"].to_numpy(dtype=float)
    X = slc["underlyer_spot"].to_numpy(dtype=float)
    tau = slc["tau"].to_numpy(dtype=float)
    bid_raw = slc["C_bid"].to_numpy(dtype=float)
    ask_raw = slc["C_ask"].to_numpy(dtype=float)
    mid_raw = slc["C_mid"].to_numpy(dtype=float)
    intrinsic = np.maximum(1.0 - K / F, 0.0)
    # Retain quoted bounds for diagnostics; enforce effective price bounds.
    bid = np.nan_to_num(bid_raw, nan=0.0, neginf=0.0, posinf=1.0)
    ask = np.nan_to_num(ask_raw, nan=1.0, neginf=0.0, posinf=1.0)
    # Collapse inverted quoted intervals to their midpoint.
    bad = ask < bid
    if np.any(bad):
        m = 0.5 * (ask[bad] + bid[bad])
        bid[bad] = m
        ask[bad] = m
    mid = np.nan_to_num(mid_raw, nan=0.5 * (bid + ask))
    mid = np.minimum(np.maximum(mid, 0.0), 1.0)
    scale = np.maximum(ask - bid, cfg.price_floor)
    x_norm = K / np.nanmedian(F)
    # Clip inverse-width weights to limit single-quote influence.
    w_l1 = 1.0 / scale
    w_l2 = 1.0 / (scale ** 2)
    lo_q, hi_q = np.nanquantile(w_l2, [cfg.weight_clip_q_low, cfg.weight_clip_q_high])
    w_l2 = np.clip(w_l2, lo_q, hi_q)
    return dict(K=K, F=F, X=X, tau=tau, bid=bid, ask=ask, mid=mid, intrinsic=intrinsic,
                scale=scale, x_norm=x_norm, w_l1=w_l1, w_l2=w_l2)


def solve_l1(arr: Dict[str, np.ndarray], cfg: RepairConfig) -> Dict:
    """Solve the L1 projection with a common or elastic bid--ask box."""
    K, bid, ask, mid, intrinsic, scale, x_norm = (
        arr[k] for k in ["K", "bid", "ask", "mid", "intrinsic", "scale", "x_norm"]
    )
    n = len(K)
    if not cfg.elastic_box:
        lower = np.maximum(intrinsic, bid)
        upper = np.minimum(1.0, ask)
        if np.any(lower > upper + 1e-12):
            return {"success": False, "status": "hard_box_empty", "message": "lower bound exceeds upper bound", "c": None}
        N = 2 * n
        c_obj = np.zeros(N)
        c_obj[n:2*n] = arr["w_l1"]
        A_ub = []
        b_ub = []
        for i in range(n):
            row = np.zeros(N); row[i] = 1.0; row[n+i] = -1.0
            A_ub.append(row); b_ub.append(mid[i])
            row = np.zeros(N); row[i] = -1.0; row[n+i] = -1.0
            A_ub.append(row); b_ub.append(-mid[i])
        full_A, full_b, _ = build_full_constraint_matrix(x_norm)
        for base_row, bound in zip(full_A, full_b):
            row = np.zeros(N); row[:n] = base_row
            A_ub.append(row); b_ub.append(float(bound))
        bounds = [(float(lower[i]), float(upper[i])) for i in range(n)] + [(0.0, None)] * n
        try:
            res = linprog(c_obj, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), bounds=bounds, method=cfg.linprog_method)
        except Exception as e:
            return {"success": False, "status": "exception", "message": str(e), "c": None}
        if not res.success:
            return {"success": False, "status": str(res.status), "message": res.message, "c": None}
        c = res.x[:n]
        ver = verify_solution(c, bid, ask, intrinsic, x_norm)
        success = bool(ver["verified_feasible"])
        return {"success": success, "status": str(res.status), "message": res.message, "objective": float(res.fun), "c": c, **ver}

    # Elastic-box L1 diagnostic.
    N = 4 * n
    c_obj = np.zeros(N)
    c_obj[n:2*n] = arr["w_l1"]
    c_obj[2*n:3*n] = cfg.elastic_penalty / scale
    c_obj[3*n:4*n] = cfg.elastic_penalty / scale
    A_ub = []
    b_ub = []
    for i in range(n):
        row = np.zeros(N); row[i] = 1.0; row[n+i] = -1.0
        A_ub.append(row); b_ub.append(mid[i])
        row = np.zeros(N); row[i] = -1.0; row[n+i] = -1.0
        A_ub.append(row); b_ub.append(-mid[i])
        row = np.zeros(N); row[i] = -1.0; row[2*n+i] = -1.0
        A_ub.append(row); b_ub.append(-bid[i])
        row = np.zeros(N); row[i] = 1.0; row[3*n+i] = -1.0
        A_ub.append(row); b_ub.append(ask[i])
    full_A, full_b, _ = build_full_constraint_matrix(x_norm)
    for base_row, bound in zip(full_A, full_b):
        row = np.zeros(N); row[:n] = base_row
        A_ub.append(row); b_ub.append(float(bound))
    bounds = [(float(intrinsic[i]), 1.0) for i in range(n)] + [(0.0, None)] * (3 * n)
    try:
        res = linprog(c_obj, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), bounds=bounds, method=cfg.linprog_method)
    except Exception as e:
        return {"success": False, "status": "exception", "message": str(e), "c": None}
    if not res.success:
        return {"success": False, "status": str(res.status), "message": res.message, "c": None}
    c = res.x[:n]
    ver = verify_solution(c, bid, ask, intrinsic, x_norm)
    return {"success": bool(ver["verified_feasible"]), "status": str(res.status), "message": res.message, "objective": float(res.fun), "c": c, **ver}


def linear_constraints_c_only(x_norm: np.ndarray) -> LinearConstraint:
    """Complete normalized-call strike constraints for a c-only problem."""
    return scipy_linear_constraint(x_norm)


def build_c_only_constraints(arr: Dict[str, np.ndarray]) -> List[LinearConstraint]:
    """Complete normalized-call strike constraints for a c-only problem."""
    return [scipy_linear_constraint(arr["x_norm"])]



def label_float(x: float) -> str:
    s = f"{float(x):g}".replace("-", "m").replace(".", "p")
    return s


def parse_smooth_rho_from_arm(arm: str, cfg: RepairConfig) -> float:
    a = arm.upper()
    if a.startswith("SMOOTH_RHO_"):
        token = a.replace("SMOOTH_RHO_", "").replace("P", ".").replace("M", "-")
        try:
            return float(token)
        except Exception:
            return cfg.smooth_rho
    if a == "SMOOTH":
        return cfg.smooth_rho
    return 0.0


def parse_budget_from_arm(arm: str) -> Optional[float]:
    a = arm.upper()
    if a.startswith("BUDGET_"):
        token = a.replace("BUDGET_", "").replace("P", ".").replace("M", "-")
        try:
            return float(token)
        except Exception:
            return None
    return None


def solve_qp_like(arr: Dict[str, np.ndarray], cfg: RepairConfig, arm: str, l1_c: Optional[np.ndarray],
                  *, target_mid: Optional[np.ndarray] = None,
                  target_weights: Optional[np.ndarray] = None,
                  smooth_rho: Optional[float] = None,
                  cfg_override: Optional[RepairConfig] = None) -> Dict:
    """Solve an L2, smoothness, or target-projection QP."""
    cfg2 = cfg_override or cfg
    K, bid, ask, mid0, intrinsic, x_norm = (arr[k] for k in ["K", "bid", "ask", "mid", "intrinsic", "x_norm"])
    n = len(K)
    mid = np.asarray(target_mid if target_mid is not None else mid0, dtype=float)
    w = np.asarray(target_weights if target_weights is not None else arr["w_l2"], dtype=float)
    w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
    if np.nanmax(w) <= 0:
        w = np.ones(n)
    arm_upper = arm.upper()
    rho = parse_smooth_rho_from_arm(arm_upper, cfg2) if smooth_rho is None else smooth_rho
    use_smooth = (arm_upper.startswith("SMOOTH") or arm_upper in {"IVSPLINE", "SVI", "FILTER_SMOOTH"} or (smooth_rho is not None and smooth_rho > 0))
    D2 = second_diff_matrix(x_norm) if use_smooth else np.zeros((0, n))

    if cfg2.qp_backend.lower() == "cvxpy":
        try:
            import cvxpy as cp  # type: ignore
        except Exception:
            return {"success": False, "status": "cvxpy_not_installed", "message": "Install cvxpy/osqp/clarabel for QP support", "c": None}
        c = cp.Variable(n)
        constraints = [c >= intrinsic, c <= 1.0]
        if cfg2.elastic_box:
            xi_low = cp.Variable(n, nonneg=True)
            xi_high = cp.Variable(n, nonneg=True)
            constraints += [c + xi_low >= bid, c - xi_high <= ask]
            slack_pen = cfg2.elastic_penalty * cp.sum(cp.multiply(1.0 / arr["scale"], xi_low + xi_high))
        else:
            constraints += [c >= bid, c <= ask]
            slack_pen = 0.0
        full_A, full_b, _ = build_full_constraint_matrix(x_norm)
        constraints += [full_A @ c <= full_b]
        obj = cp.sum(cp.multiply(w, cp.square(c - mid))) + slack_pen
        if D2.size and rho > 0:
            obj += rho * cp.sum_squares(D2 @ c)
        prob = cp.Problem(cp.Minimize(obj), constraints)
        installed = set(cp.installed_solvers())
        solver_order = [cfg2.qp_solver.upper()] if cfg2.qp_solver.lower() != "auto" else ["OSQP", "CLARABEL", "ECOS", "SCS"]
        last_msg = ""
        for solver in solver_order:
            if solver not in installed:
                continue
            try:
                if solver == "OSQP":
                    prob.solve(solver=solver, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000, verbose=False)
                else:
                    prob.solve(solver=solver, verbose=False)
                if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and c.value is not None:
                    cval = np.asarray(c.value, dtype=float).reshape(-1)
                    ver = verify_solution(cval, bid, ask, intrinsic, x_norm)
                    success = bool(ver["verified_feasible"])
                    return {"success": success, "status": prob.status, "message": f"cvxpy:{solver}", "objective": float(prob.value), "c": cval, **ver}
                last_msg = str(prob.status)
            except Exception as e:
                last_msg = f"{solver}: {e}"
                continue
        return {"success": False, "status": "cvxpy_failed", "message": last_msg or "No installed cvxpy solver succeeded", "c": None}

    # SciPy fallback.
    lower = np.maximum(intrinsic, bid) if not cfg2.elastic_box else intrinsic
    upper = np.minimum(1.0, ask) if not cfg2.elastic_box else np.ones(n)
    if np.any(lower > upper + 1e-12):
        return {"success": False, "status": "hard_box_empty", "message": "lower bound exceeds upper bound", "c": None}
    c0 = np.minimum(np.maximum(l1_c if l1_c is not None else mid, lower), upper)
    cons = [linear_constraints_c_only(x_norm)]
    bounds = Bounds(lower, upper)
    def obj_np(c_: np.ndarray) -> float:
        val = float(np.sum(w * (c_ - mid) ** 2))
        if D2.size and rho > 0:
            val += float(rho * np.sum((D2 @ c_) ** 2))
        return val
    def jac_np(c_: np.ndarray) -> np.ndarray:
        grad = 2.0 * w * (c_ - mid)
        if D2.size and rho > 0:
            grad = grad + 2.0 * rho * (D2.T @ (D2 @ c_))
        return grad
    try:
        res = minimize(obj_np, c0, jac=jac_np, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": cfg2.solver_maxiter, "ftol": 1e-10, "disp": False})
    except Exception as e:
        return {"success": False, "status": "exception", "message": str(e), "c": None}
    if not res.success:
        return {"success": False, "status": str(res.status), "message": str(res.message), "objective": np.nan, "c": None}
    cval = res.x
    ver = verify_solution(cval, bid, ask, intrinsic, x_norm)
    success = bool(ver["verified_feasible"])
    return {"success": success, "status": str(res.status), "message": str(res.message), "objective": float(res.fun), "c": cval, **ver}


def solve_budget_smoothing(arr: Dict[str, np.ndarray], cfg: RepairConfig, budget: float, l1_c: Optional[np.ndarray]) -> Dict:
    """Minimize curvature under no-arbitrage, quote-box, and displacement constraints."""
    K, bid, ask, mid, intrinsic, x_norm = (arr[k] for k in ["K", "bid", "ask", "mid", "intrinsic", "x_norm"])
    n = len(K)
    D2 = second_diff_matrix(x_norm)
    try:
        import cvxpy as cp  # type: ignore
    except Exception:
        return {"success": False, "status": "cvxpy_not_installed", "message": "Budget smoothing requires cvxpy", "c": None}
    lower = np.maximum(np.maximum(intrinsic, bid), mid - budget * arr["scale"])
    upper = np.minimum(np.minimum(1.0, ask), mid + budget * arr["scale"])
    if np.any(lower > upper + 1e-12):
        return {"success": False, "status": "budget_box_empty", "message": f"No feasible no-arb curve within budget {budget}", "c": None}
    c = cp.Variable(n)
    full_A, full_b, _ = build_full_constraint_matrix(x_norm)
    constraints = [c >= lower, c <= upper, full_A @ c <= full_b]
    # Small quote-fit term breaks curvature ties.
    obj = cp.sum_squares(D2 @ c) + 1e-6 * cp.sum(cp.multiply(arr["w_l2"], cp.square(c - mid)))
    prob = cp.Problem(cp.Minimize(obj), constraints)
    solver_order = [cfg.qp_solver.upper()] if cfg.qp_solver.lower() != "auto" else ["OSQP", "CLARABEL", "ECOS", "SCS"]
    installed = set(cp.installed_solvers())
    last_msg = ""
    for solver in solver_order:
        if solver not in installed:
            continue
        try:
            if solver == "OSQP":
                prob.solve(solver=solver, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000, verbose=False)
            else:
                prob.solve(solver=solver, verbose=False)
            if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and c.value is not None:
                cval = np.asarray(c.value, dtype=float).reshape(-1)
                ver = verify_solution(cval, bid, ask, intrinsic, x_norm)
                return {"success": bool(ver["verified_feasible"]), "status": prob.status, "message": f"cvxpy:{solver};budget={budget}", "objective": float(prob.value), "c": cval, **ver}
            last_msg = str(prob.status)
        except Exception as e:
            last_msg = f"{solver}: {e}"
    return {"success": False, "status": "budget_failed", "message": last_msg, "c": None}


def filtered_weights_for_slice(slc: pd.DataFrame, arr: Dict[str, np.ndarray], cfg: RepairConfig) -> np.ndarray:
    """Desk-style filtering: downweight wide-spread/stale/zero-volume-OI quotes."""
    w = arr["w_l2"].copy()
    rel = slc["relspread_floor"].to_numpy(dtype=float)
    thr = np.nanquantile(rel, cfg.filter_spread_quantile) if np.isfinite(rel).sum() else np.inf
    bad = rel >= thr
    for col in ["volume", "open_interest"]:
        if col in slc.columns:
            vals = pd.to_numeric(slc[col], errors="coerce").to_numpy(dtype=float)
            bad |= np.isfinite(vals) & (vals <= 0)
    # Preserve at least one quote term.
    if bad.sum() >= len(bad) - 2:
        bad = rel >= np.nanquantile(rel, 0.90)
    w[bad] = cfg.filter_bad_weight * np.nanmedian(w[~bad]) if np.any(~bad) else cfg.filter_bad_weight * np.nanmedian(w)
    return w


def mid_iv_from_base_prices(arr: Dict[str, np.ndarray]) -> np.ndarray:
    return np.array([implied_vol_from_base_call_scalar(arr["mid"][i], arr["F"][i], arr["K"][i], arr["tau"][i]) for i in range(len(arr["K"]))])


def svi_total_variance(k: np.ndarray, params: np.ndarray) -> np.ndarray:
    a, b, rho, m, sig = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))


def svi_projected_target(arr: Dict[str, np.ndarray], cfg: RepairConfig) -> Optional[np.ndarray]:
    """Fit raw SVI total variance to the slice's midpoint IV and return target call prices."""
    K, F, tau = arr["K"], arr["F"], arr["tau"]
    k = np.log(K / np.nanmedian(F))
    sig = mid_iv_from_base_prices(arr)
    valid = np.isfinite(sig) & (sig > 0) & np.isfinite(k)
    if valid.sum() < 7:
        return None
    w_obs = (sig[valid] ** 2) * np.nanmedian(tau[valid])
    k_obs = k[valid]
    if np.nanmax(w_obs) <= 0:
        return None
    p0 = np.array([max(np.nanmin(w_obs) * 0.5, 1e-5), 0.1, -0.2, 0.0, 0.2])
    lb = np.array([1e-8, 1e-8, -0.999, -2.0, 1e-4])
    ub = np.array([10.0, 10.0, 0.999, 2.0, 5.0])
    def resid(p):
        return svi_total_variance(k_obs, p) - w_obs
    try:
        res = least_squares(resid, p0, bounds=(lb, ub), max_nfev=cfg.svi_max_nfev)
        if not res.success:
            return None
        w_fit = np.maximum(svi_total_variance(k, res.x), 1e-10)
        sigma_fit = np.sqrt(w_fit / np.maximum(tau, 1e-10))
        target = deribit_call_base_price_from_iv(F, K, tau, sigma_fit)
        return np.minimum(np.maximum(target, arr["intrinsic"]), 1.0)
    except Exception:
        return None


def ivspline_projected_target(arr: Dict[str, np.ndarray], cfg: RepairConfig) -> Optional[np.ndarray]:
    """Smooth total variance in IV-space with a spline, then return target call prices."""
    if UnivariateSpline is None:
        return None
    K, F, tau = arr["K"], arr["F"], arr["tau"]
    k = np.log(K / np.nanmedian(F))
    sig = mid_iv_from_base_prices(arr)
    w_obs = sig ** 2 * tau
    valid = np.isfinite(k) & np.isfinite(w_obs) & (w_obs > 0)
    if valid.sum() < 5:
        return None
    order = np.argsort(k[valid])
    k_obs = k[valid][order]
    wv = w_obs[valid][order]
    try:
        spline = UnivariateSpline(k_obs, wv, k=min(3, len(k_obs)-1), s=cfg.spline_smoothing * len(k_obs))
        w_fit = np.maximum(spline(k), 1e-10)
        sigma_fit = np.sqrt(w_fit / np.maximum(tau, 1e-10))
        target = deribit_call_base_price_from_iv(F, K, tau, sigma_fit)
        return np.minimum(np.maximum(target, arr["intrinsic"]), 1.0)
    except Exception:
        return None


def solve_slice_repairs(slc: pd.DataFrame, cfg: RepairConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return repaired rows and diagnostics for a single date-expiry slice."""
    slc = slc.sort_values("strike").copy()
    arr = prepare_slice_arrays(slc, cfg)
    n = len(arr["K"])
    if n < cfg.min_strikes:
        return pd.DataFrame(), pd.DataFrame()

    arms = tuple(a.upper() for a in cfg.arms)
    solutions: Dict[str, Dict] = {}
    l1_c = None
    if "L1" in arms:
        sol = solve_l1(arr, cfg)
        solutions["L1"] = sol
        if sol.get("success") and sol.get("c") is not None:
            l1_c = sol["c"]

    for arm in arms:
        if arm == "L1":
            continue
        if arm == "L2":
            solutions[arm] = solve_qp_like(arr, cfg, arm, l1_c)
        elif arm == "SMOOTH" or arm.startswith("SMOOTH_RHO_"):
            rho = parse_smooth_rho_from_arm(arm, cfg)
            solutions[arm] = solve_qp_like(arr, cfg, arm, l1_c, smooth_rho=rho)
        elif arm.startswith("BUDGET_"):
            b = parse_budget_from_arm(arm)
            solutions[arm] = solve_budget_smoothing(arr, cfg, b, l1_c) if b is not None else {"success": False, "status": "bad_budget_arm", "message": arm, "c": None}
        elif arm == "FILTER_L2":
            w = filtered_weights_for_slice(slc, arr, cfg)
            solutions[arm] = solve_qp_like(arr, cfg, arm, l1_c, target_weights=w)
        elif arm == "FILTER_SMOOTH":
            w = filtered_weights_for_slice(slc, arr, cfg)
            solutions[arm] = solve_qp_like(arr, cfg, arm, l1_c, target_weights=w, smooth_rho=cfg.smooth_rho)
        elif arm == "IVSPLINE":
            target = ivspline_projected_target(arr, cfg)
            if target is None:
                solutions[arm] = {"success": False, "status": "ivspline_failed", "message": "could not build IV spline target", "c": None}
            else:
                solutions[arm] = solve_qp_like(arr, cfg, arm, l1_c, target_mid=target, smooth_rho=cfg.smooth_rho)
        elif arm == "SVI":
            target = svi_projected_target(arr, cfg)
            if target is None:
                solutions[arm] = {"success": False, "status": "svi_failed", "message": "could not fit SVI target", "c": None}
            else:
                solutions[arm] = solve_qp_like(arr, cfg, arm, l1_c, target_mid=target, smooth_rho=cfg.smooth_rho)
        elif arm == "ELASTIC_L2":
            cfg_el = replace(cfg, elastic_box=True)
            solutions[arm] = solve_qp_like(arr, cfg_el, arm, l1_c, cfg_override=cfg_el)
        else:
            solutions[arm] = {"success": False, "status": "unknown_arm", "message": f"Unknown arm {arm}", "c": None}

    key_vals = {c: slc.iloc[0][c] for c in slice_key_cols()}
    diag_rows = []
    surface_rows = []
    for arm, sol in solutions.items():
        diag = {
            **key_vals,
            "arm": arm,
            "n_strikes": n,
            "success": bool(sol.get("success", False)),
            "status": sol.get("status"),
            "message": sol.get("message"),
            "objective": finite(sol.get("objective")),
        }
        for k in ["verified_feasible", "max_mono_violation", "max_conv_violation",
                  "max_vertical_spread_violation", "max_vertical_slope", "max_constraint_violation",
                  "max_intrinsic_violation", "max_upper_violation", "box_slack_sum",
                  "max_box_low_violation", "max_box_high_violation"]:
            diag[k] = sol.get(k, np.nan)
        diag_rows.append(diag)
        if not sol.get("success", False) or sol.get("c") is None:
            continue
        c_hat = np.asarray(sol["c"], dtype=float)
        for j, (_, row) in enumerate(slc.iterrows()):
            surface_rows.append({
                **{col: row[col] for col in [
                    "currency", "date", "expiration_date", "strike", "forward_price", "underlyer_spot", "tau", "DTE",
                    "dte_bucket", "k", "C_bid", "C_ask", "C_mid", "intrinsic", "relspread_floor", "h_abs", "h_norm",
                    "ranking_score", "ranking_score_rel_h2", "ranking_score_abs_h2", "slice_ranking_score", "threshold_screen_obs",
                    "short_dte", "wing", "wide_spread", "sparse_strike", "high_score_cell", "threshold_screen_cell", "edge_strike"
                ] if col in row.index},
                "arm": arm,
                "C_repaired": float(c_hat[j]),
                "repair_abs_mid_diff": float(abs(c_hat[j] - arr["mid"][j])),
                "repair_norm_mid_diff": float(abs(c_hat[j] - arr["mid"][j]) / (arr["scale"][j] + EPS)),
            })
    return pd.DataFrame(surface_rows), pd.DataFrame(diag_rows)


def run_repairs(panel_repair: pd.DataFrame, cfg: RepairConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    surfaces = []
    diags = []
    groups = list(panel_repair.groupby(slice_key_cols(), sort=False))
    for i, (_, slc) in enumerate(groups, 1):
        srf, d = solve_slice_repairs(slc, cfg)
        if len(srf):
            surfaces.append(srf)
        if len(d):
            diags.append(d)
        if i % 25 == 0:
            print(f"[repair] solved {i}/{len(groups)} slices", flush=True)
    surf_df = pd.concat(surfaces, ignore_index=True) if surfaces else pd.DataFrame()
    diag_df = pd.concat(diags, ignore_index=True) if diags else pd.DataFrame()
    return surf_df, diag_df


# =============================================================================
# Greeks / strike dispersion
# =============================================================================

def compute_greeks(surfaces: pd.DataFrame) -> pd.DataFrame:
    if surfaces.empty:
        return pd.DataFrame()
    rows = []
    for _, slc in surfaces.groupby(slice_key_cols() + ["arm"], sort=False):
        slc = slc.sort_values("strike").copy()
        K = slc["strike"].to_numpy(dtype=float)
        C = slc["C_repaired"].to_numpy(dtype=float)
        F = slc["forward_price"].to_numpy(dtype=float)
        X = slc["underlyer_spot"].to_numpy(dtype=float)
        tau = slc["tau"].to_numpy(dtype=float)
        x_norm = K / np.nanmedian(F)
        c_k, c_kk = first_second_derivatives_nonuniform(C, K)
        c_x, c_xx = first_second_derivatives_nonuniform(C, x_norm)
        ivs = np.array([implied_vol_from_base_call_scalar(C[i], F[i], K[i], tau[i]) for i in range(len(K))])
        call_delta = np.full(len(K), np.nan)
        put_delta = np.full(len(K), np.nan)
        gamma = np.full(len(K), np.nan)
        vega = np.full(len(K), np.nan)
        for i in range(len(K)):
            dc, g, v = usd_spot_greeks_from_iv("C", X[i], F[i], K[i], tau[i], ivs[i])
            dp, _, _ = usd_spot_greeks_from_iv("P", X[i], F[i], K[i], tau[i], ivs[i])
            call_delta[i] = dc
            put_delta[i] = dp
            gamma[i] = g
            vega[i] = v
        tmp = slc.copy()
        tmp["iv_repaired"] = ivs
        tmp["C_K"] = c_k
        tmp["C_KK"] = c_kk
        tmp["C_x"] = c_x
        tmp["C_xx"] = c_xx
        tmp["delta_call_usd"] = call_delta
        tmp["delta_put_usd"] = put_delta
        tmp["gamma_usd"] = gamma
        tmp["vega_usd"] = vega
        # Local curvature proxy without density normalization.
        tmp["rnd_proxy"] = np.maximum(c_kk * tmp["underlyer_spot"].to_numpy(dtype=float), 0.0)
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_strike_pairwise_dispersion(greeks: pd.DataFrame, cfg: RepairConfig) -> pd.DataFrame:
    if greeks.empty:
        return pd.DataFrame()
    rows = []
    group_cols = slice_key_cols() + ["strike"]
    for _, g in greeks.groupby(group_cols, sort=False):
        if g["arm"].nunique() < 2:
            continue
        slice_gamma_scale = np.nanmedian(np.abs(
            greeks[(greeks["currency"] == g.iloc[0]["currency"]) &
                   (greeks["date"] == g.iloc[0]["date"]) &
                   (greeks["expiration_date"] == g.iloc[0]["expiration_date"])]
            ["gamma_usd"].to_numpy(dtype=float)
        ))
        if not np.isfinite(slice_gamma_scale) or slice_gamma_scale < EPS:
            slice_gamma_scale = np.nanmedian(np.abs(greeks["gamma_usd"].to_numpy(dtype=float)))
        slice_gamma_scale = max(finite(slice_gamma_scale, 0.0), EPS)
        records = {r["arm"]: r for _, r in g.iterrows()}
        arms = sorted(records)
        for a, b in combinations(arms, 2):
            ra, rb = records[a], records[b]
            price_diff = abs(float(ra["C_repaired"]) - float(rb["C_repaired"]))
            box_width = max(float(ra["C_ask"] - ra["C_bid"]), cfg.price_floor)
            gamma_diff = abs(float(ra["gamma_usd"]) - float(rb["gamma_usd"])) if np.isfinite(ra["gamma_usd"]) and np.isfinite(rb["gamma_usd"]) else np.nan
            gamma_pair_scale = 0.5 * (abs(finite(ra["gamma_usd"], 0.0)) + abs(finite(rb["gamma_usd"], 0.0)))
            gamma_rr = gamma_diff / (gamma_pair_scale + EPS) if np.isfinite(gamma_diff) else np.nan
            gamma_diff_scaled = gamma_diff / (slice_gamma_scale + EPS) if np.isfinite(gamma_diff) else np.nan
            ckk_diff = abs(finite(ra["C_KK"], np.nan) - finite(rb["C_KK"], np.nan))
            h_abs = finite(ra.get("h_abs"), np.nan)
            amp_h2 = ckk_diff * (h_abs ** 2) / (price_diff + EPS) if np.isfinite(ckk_diff) and np.isfinite(h_abs) else np.nan
            cxx_diff = abs(finite(ra["C_xx"], np.nan) - finite(rb["C_xx"], np.nan))
            h_norm = finite(ra.get("h_norm"), np.nan)
            amp_x_h2 = cxx_diff * (h_norm ** 2) / (price_diff + EPS) if np.isfinite(cxx_diff) and np.isfinite(h_norm) else np.nan
            small_large = (
                (price_diff / box_width < cfg.price_diff_box_threshold)
                and np.isfinite(gamma_rr) and gamma_rr > cfg.gamma_rr_threshold
                and np.isfinite(gamma_diff_scaled) and gamma_diff_scaled > cfg.gamma_material_threshold
            )
            rows.append({
                "currency": ra["currency"], "date": int(ra["date"]), "expiration_date": int(ra["expiration_date"]),
                "strike": float(ra["strike"]), "arm_a": a, "arm_b": b,
                "DTE": float(ra["DTE"]), "dte_bucket": ra.get("dte_bucket"), "k": float(ra["k"]),
                "short_dte": bool(ra.get("short_dte", False)), "wing": bool(ra.get("wing", False)),
                "wide_spread": bool(ra.get("wide_spread", False)), "sparse_strike": bool(ra.get("sparse_strike", False)),
                "high_score_cell": bool(ra.get("high_score_cell", False)), "edge_strike": bool(ra.get("edge_strike", False)),
                "relspread_floor": float(ra.get("relspread_floor", np.nan)), "ranking_score": float(ra.get("ranking_score", np.nan)), "ranking_score_rel_h2": float(ra.get("ranking_score_rel_h2", np.nan)), "ranking_score_abs_h2": float(ra.get("ranking_score_abs_h2", np.nan)), "h_abs": h_abs, "h_norm": h_norm,
                "price_diff": price_diff,
                "price_diff_box": price_diff / (box_width + EPS),
                "gamma_diff": gamma_diff,
                "gamma_rr": gamma_rr,
                "gamma_diff_scaled": gamma_diff_scaled,
                "slice_gamma_scale": slice_gamma_scale,
                "ckk_diff": ckk_diff,
                "amp_h2": amp_h2,
                "cxx_diff": cxx_diff,
                "amp_x_h2": amp_x_h2,
                "small_price_large_gamma": bool(small_large),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Standardized books and P&L
# =============================================================================

BOOKS = {
    "ATM_STRADDLE": [("C", 0.00, 1.0), ("P", 0.00, 1.0)],
    "PUT_SPREAD_5_10": [("P", -0.05, 1.0), ("P", -0.10, -1.0)],
    "RISK_REVERSAL_5": [("C", 0.05, 1.0), ("P", -0.05, -1.0)],
    "PUT_FLY_0_5_10": [("P", -0.10, 1.0), ("P", -0.05, -2.0), ("P", 0.00, 1.0)],
}


def select_book_legs(slice_base: pd.DataFrame, book_name: str, cfg: RepairConfig) -> Optional[List[Dict]]:
    legs = BOOKS[book_name]
    slc = slice_base.sort_values("strike")
    chosen = []
    for claim, target_k, qty in legs:
        idx = (slc["k"] - target_k).abs().idxmin()
        row = slc.loc[idx]
        dist = abs(float(row["k"]) - target_k)
        if dist > cfg.strike_tol:
            return None
        chosen.append({
            "claim": claim, "target_k": target_k, "qty": qty, "strike": float(row["strike"]),
            "k": float(row["k"]), "strike_dist": dist,
        })
    if book_name != "ATM_STRADDLE":
        # Prevent degenerate spread/fly books on sparse grids.
        if len({leg["strike"] for leg in chosen}) < len(chosen):
            return None
    return chosen


def option_price_base_from_call(c_call: float, K: float, F: float, claim: str) -> float:
    if str(claim).upper().startswith("P"):
        return c_call - 1.0 + K / F
    return c_call


def common_mark_for_leg(panel_full: pd.DataFrame, date: int, expiration: int, strike: float, claim: str) -> Optional[Dict]:
    row = panel_full[(panel_full["date"] == date) & (panel_full["expiration_date"] == expiration) & (panel_full["strike"] == strike)]
    if row.empty:
        return None
    r = row.iloc[0]
    c = float(r["C_mid"])
    F = float(r["forward_price"])
    X = float(r["underlyer_spot"])
    price_base = option_price_base_from_call(c, strike, F, claim)
    return {"price_base": price_base, "price_usd": price_base * X, "F": F, "X": X}


def compute_book_outputs(greeks: pd.DataFrame, panel_full: pd.DataFrame, cfg: RepairConfig) -> pd.DataFrame:
    if greeks.empty:
        return pd.DataFrame()
    rows = []
    # One row per base slice.
    base_cols = ["currency", "date", "expiration_date", "strike", "forward_price", "underlyer_spot", "tau", "DTE", "k", "C_mid"]
    base = panel_full[base_cols].drop_duplicates(["currency", "date", "expiration_date", "strike"])

    for key, base_slc in base.groupby(slice_key_cols(), sort=False):
        currency, date, expiration = key
        for book_name in BOOKS:
            legs = select_book_legs(base_slc, book_name, cfg)
            if legs is None:
                continue
            next_date = int(date) + SECONDS_PER_DAY
            # Common marks at t and t+1.
            common_t = []
            common_tp1 = []
            for leg in legs:
                mt = common_mark_for_leg(panel_full, int(date), int(expiration), leg["strike"], leg["claim"])
                mn = common_mark_for_leg(panel_full, next_date, int(expiration), leg["strike"], leg["claim"])
                common_t.append(mt)
                common_tp1.append(mn)
            nextday_available = all(m is not None for m in common_tp1) and all(m is not None for m in common_t)
            V_common_t = np.nan
            V_common_tp1 = np.nan
            X_t = float(base_slc.iloc[0]["underlyer_spot"])
            X_tp1 = np.nan
            if nextday_available:
                V_common_t = sum(leg["qty"] * m["price_usd"] for leg, m in zip(legs, common_t))
                V_common_tp1 = sum(leg["qty"] * m["price_usd"] for leg, m in zip(legs, common_tp1))
                X_tp1 = float(common_tp1[0]["X"])
            # Repair-specific valuation and hedge.
            g_slice = greeks[(greeks["currency"] == currency) & (greeks["date"] == date) & (greeks["expiration_date"] == expiration)]
            for arm, g_arm in g_slice.groupby("arm", sort=False):
                V_rep_usd = 0.0
                delta = 0.0
                gamma = 0.0
                leg_ok = True
                for leg in legs:
                    rr = g_arm[g_arm["strike"] == leg["strike"]]
                    if rr.empty:
                        leg_ok = False
                        break
                    r = rr.iloc[0]
                    c_call = float(r["C_repaired"])
                    F = float(r["forward_price"])
                    X = float(r["underlyer_spot"])
                    price_base = option_price_base_from_call(c_call, leg["strike"], F, leg["claim"])
                    if str(leg["claim"]).upper().startswith("P"):
                        d = float(r["delta_put_usd"])
                    else:
                        d = float(r["delta_call_usd"])
                    gma = float(r["gamma_usd"])
                    V_rep_usd += leg["qty"] * price_base * X
                    delta += leg["qty"] * d
                    gamma += leg["qty"] * gma
                if not leg_ok:
                    continue
                hedge = -delta
                pnl_common = np.nan
                loss_common = np.nan
                tc = 0.0
                if nextday_available and np.isfinite(X_tp1):
                    tc = cfg.hedge_tc_bps * 1e-4 * abs(hedge) * X_t
                    pnl_common = (V_common_tp1 - V_common_t) + hedge * (X_tp1 - X_t) - tc
                    scale = max(abs(V_common_t), cfg.pnl_scale_floor_usd)
                    loss_common = -pnl_common / scale
                rows.append({
                    "currency": currency, "date": int(date), "expiration_date": int(expiration), "book": book_name, "arm": arm,
                    "DTE": float(base_slc.iloc[0]["DTE"]), "n_legs": len(legs),
                    "leg_strikes": json.dumps([leg["strike"] for leg in legs]),
                    "leg_claims": json.dumps([leg["claim"] for leg in legs]),
                    "leg_qty": json.dumps([leg["qty"] for leg in legs]),
                    "value_repaired_usd": V_rep_usd,
                    "value_common_usd": V_common_t,
                    "delta_usd": delta,
                    "gamma_usd": gamma,
                    "hedge_underlying_units": hedge,
                    "nextday_available": bool(nextday_available),
                    "pnl_common": pnl_common,
                    "loss_common": loss_common,
                    "tc_usd": tc,
                    "underlyer_spot": X_t,
                    "underlyer_spot_next": X_tp1,
                })
    return pd.DataFrame(rows)


def compute_book_pairwise_dispersion(book_outputs: pd.DataFrame) -> pd.DataFrame:
    if book_outputs.empty:
        return pd.DataFrame()
    rows = []
    for _, g in book_outputs.groupby(["currency", "date", "expiration_date", "book"], sort=False):
        if g["arm"].nunique() < 2:
            continue
        rec = {r["arm"]: r for _, r in g.iterrows()}
        arms = sorted(rec)
        gamma_scale = max(np.nanmedian(np.abs(g["gamma_usd"].to_numpy(dtype=float))), EPS)
        delta_scale = max(np.nanmedian(np.abs(g["delta_usd"].to_numpy(dtype=float))), EPS)
        for a, b in combinations(arms, 2):
            ra, rb = rec[a], rec[b]
            gamma_diff = abs(float(ra["gamma_usd"]) - float(rb["gamma_usd"]))
            delta_diff = abs(float(ra["delta_usd"]) - float(rb["delta_usd"]))
            hedge_diff = abs(float(ra["hedge_underlying_units"]) - float(rb["hedge_underlying_units"]))
            loss_diff = np.nan
            if np.isfinite(ra["loss_common"]) and np.isfinite(rb["loss_common"]):
                loss_diff = abs(float(ra["loss_common"]) - float(rb["loss_common"]))
            rows.append({
                "currency": ra["currency"], "date": int(ra["date"]), "expiration_date": int(ra["expiration_date"]),
                "book": ra["book"], "arm_a": a, "arm_b": b, "DTE": float(ra["DTE"]),
                "delta_diff": delta_diff,
                "delta_rr": delta_diff / (delta_scale + EPS),
                "gamma_diff": gamma_diff,
                "gamma_rr": gamma_diff / (gamma_scale + EPS),
                "hedge_diff": hedge_diff,
                "loss_common_diff": loss_diff,
                "nextday_available": bool(ra["nextday_available"] and rb["nextday_available"]),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Summaries and figures
# =============================================================================

def summarize(panel_full: pd.DataFrame, panel_repair: pd.DataFrame, repairs: pd.DataFrame, diag: pd.DataFrame,
              greeks: pd.DataFrame, strike_disp: pd.DataFrame, book_outputs: pd.DataFrame,
              book_disp: pd.DataFrame, cfg: RepairConfig) -> pd.DataFrame:
    requested_arms = tuple(a.upper() for a in cfg.arms)
    attempted = len(diag)
    successes = int(diag["success"].sum()) if len(diag) and "success" in diag else 0
    n_slices = len(panel_repair.groupby(slice_key_cols())) if len(panel_repair) else 0
    success_rate = successes / attempted if attempted else np.nan

    # Slices with all requested arms.
    complete = np.nan
    if len(diag):
        ok = diag[diag["success"]].groupby(slice_key_cols())["arm"].nunique()
        complete = float((ok >= len(requested_arms)).mean()) if len(ok) else 0.0

    high_score = strike_disp[strike_disp.get("high_score_cell", False).astype(bool)] if len(strike_disp) and "high_score_cell" in strike_disp else pd.DataFrame()
    summary = {
        "currency": cfg.currency.upper(),
        "snapshot_hour": cfg.snapshot_hour,
        "dte_min_repair": cfg.dte_min_repair,
        "dte_max_repair": cfg.dte_max_repair,
        "dte_min_marking": cfg.dte_min_marking,
        "dte_max_marking": cfg.dte_max_marking,
        "requested_arms": ",".join(requested_arms),
        "panel_full_rows": len(panel_full),
        "panel_repair_rows": len(panel_repair),
        "candidate_slices": n_slices,
        "repair_attempts": attempted,
        "repair_successes": successes,
        "repair_success_rate": success_rate,
        "complete_slice_rate": complete,
        "repaired_surface_rows": len(repairs),
        "greek_rows": len(greeks),
        "strike_pairwise_rows": len(strike_disp),
        "book_output_rows": len(book_outputs),
        "book_pairwise_rows": len(book_disp),
    }
    for name, df in [("all", strike_disp), ("high_score", high_score)]:
        if len(df):
            summary[f"{name}_median_gamma_rr"] = float(np.nanmedian(df["gamma_rr"]))
            summary[f"{name}_p90_gamma_rr"] = float(np.nanpercentile(df["gamma_rr"], 90))
            summary[f"{name}_median_gamma_diff_scaled"] = float(np.nanmedian(df["gamma_diff_scaled"]))
            summary[f"{name}_p90_gamma_diff_scaled"] = float(np.nanpercentile(df["gamma_diff_scaled"], 90))
            summary[f"{name}_small_price_large_gamma_rate"] = float(np.nanmean(df["small_price_large_gamma"]))
            summary[f"{name}_median_price_diff_box"] = float(np.nanmedian(df["price_diff_box"]))
            summary[f"{name}_p90_price_diff_box"] = float(np.nanpercentile(df["price_diff_box"], 90))
        else:
            for k in ["median_gamma_rr", "p90_gamma_rr", "median_gamma_diff_scaled", "p90_gamma_diff_scaled", "small_price_large_gamma_rate", "median_price_diff_box", "p90_price_diff_box"]:
                summary[f"{name}_{k}"] = np.nan
    if len(book_disp):
        summary["book_p90_gamma_rr"] = float(np.nanpercentile(book_disp["gamma_rr"], 90))
        summary["book_p90_delta_rr"] = float(np.nanpercentile(book_disp["delta_rr"], 90))
        summary["book_p90_loss_common_diff"] = float(np.nanpercentile(book_disp["loss_common_diff"], 90))
        summary["book_nextday_pair_rate"] = float(np.nanmean(book_disp["nextday_available"]))
    else:
        summary["book_p90_gamma_rr"] = np.nan
        summary["book_p90_delta_rr"] = np.nan
        summary["book_p90_loss_common_diff"] = np.nan
        summary["book_nextday_pair_rate"] = np.nan

    # Minimum-support status indicators.
    summary["pass_greek_rr"] = bool(
        (finite(summary.get("high_score_p90_gamma_rr"), -np.inf) > cfg.gamma_rr_threshold)
        or (finite(summary.get("high_score_median_gamma_rr"), -np.inf) > 0.10)
    )
    summary["pass_material_scaled"] = bool(finite(summary.get("high_score_p90_gamma_diff_scaled"), -np.inf) > cfg.gamma_material_threshold)
    summary["pass_small_price_large_gamma"] = bool(finite(summary.get("high_score_small_price_large_gamma_rate"), 0.0) > 0.0)
    summary["analysis_signal"] = bool(
        summary["pass_greek_rr"] and summary["pass_material_scaled"] and summary["pass_small_price_large_gamma"]
    )
    return pd.DataFrame([summary])


def make_figures(strike_disp: pd.DataFrame, book_disp: pd.DataFrame, greeks: pd.DataFrame, out_dir: Path) -> None:
    if plt is None:
        return
    fig_dir = ensure_dir(out_dir / "figures")
    if len(strike_disp):
        sample = strike_disp.copy()
        if len(sample) > 20000:
            sample = sample.sample(20000, random_state=7)
        plt.figure(figsize=(7, 5))
        x = np.log10(sample["price_diff_box"].clip(lower=1e-8))
        y = np.log10(sample["gamma_diff_scaled"].clip(lower=1e-8))
        plt.scatter(x, y, s=5, alpha=0.25)
        plt.xlabel("log10(price diff / bid-ask box width)")
        plt.ylabel("log10(gamma diff / slice gamma scale)")
        plt.title("Price-space closeness vs material gamma dispersion")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig1_fan_price_vs_gamma_scaled.png", dpi=180)
        plt.close()

        # Compact state-summary table.
        tmp = strike_disp.copy()
        tmp["state"] = np.where(tmp["high_score_cell"], "high_score", "other")
        agg = tmp.groupby("state")["gamma_diff_scaled"].agg(["median", lambda x: np.nanpercentile(x, 90)]).rename(columns={"<lambda_0>": "p90"})
        plt.figure(figsize=(5, 4))
        agg[["median", "p90"]].plot(kind="bar", ax=plt.gca())
        plt.ylabel("Gamma diff / slice gamma scale")
        plt.title("High-score amplification")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig2_high_score_vs_other_gamma.png", dpi=180)
        plt.close()

        # Select the largest qualifying gamma gap.
        cand = strike_disp[strike_disp["small_price_large_gamma"]].copy()
        if cand.empty:
            cand = strike_disp.copy()
        if len(cand):
            row = cand.sort_values("gamma_diff_scaled", ascending=False).iloc[0]
            slc = greeks[(greeks["currency"] == row["currency"]) & (greeks["date"] == row["date"]) &
                         (greeks["expiration_date"] == row["expiration_date"]) &
                         (greeks["arm"].isin([row["arm_a"], row["arm_b"]]))].copy()
            if len(slc):
                plt.figure(figsize=(7, 5))
                for arm, g in slc.groupby("arm"):
                    g = g.sort_values("strike")
                    plt.plot(g["k"], g["gamma_usd"], marker="o", markersize=2, label=str(arm))
                plt.axvline(row["k"], linestyle="--", linewidth=1)
                plt.xlabel("log-moneyness k")
                plt.ylabel("USD spot gamma")
                plt.title("Real-panel case study: repaired gamma curves")
                plt.legend()
                plt.tight_layout()
                plt.savefig(fig_dir / "fig3_case_study_gamma.png", dpi=180)
                plt.close()
    if len(book_disp):
        plt.figure(figsize=(7, 5))
        bd = book_disp.copy()
        bd["loss_common_diff"] = bd["loss_common_diff"].replace([np.inf, -np.inf], np.nan)
        agg = bd.groupby("book")[["gamma_rr", "loss_common_diff"]].median(numeric_only=True)
        agg.plot(kind="bar", ax=plt.gca())
        plt.title("Book-level repair dispersion")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig4_book_dispersion.png", dpi=180)
        plt.close()


# =============================================================================
# CLI / main
# =============================================================================


def expand_arms_from_args(args: argparse.Namespace) -> Tuple[str, ...]:
    arms = [a.upper() for a in args.arms]
    if args.smooth_rhos:
        arms = [a for a in arms if a != "SMOOTH"]
        arms.extend([f"SMOOTH_RHO_{label_float(x)}".upper() for x in args.smooth_rhos])
    if args.budget_smoothing:
        arms.extend([f"BUDGET_{label_float(x)}".upper() for x in args.budget_smoothing])
    if args.include_broader_arms:
        arms.extend(["FILTER_L2", "IVSPLINE", "SVI", "ELASTIC_L2"])
    if args.add_arms:
        arms.extend([a.upper() for a in args.add_arms])
    # Stable deduplication.
    seen = set()
    out = []
    for a in arms:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return tuple(out)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repair-risk analysis for Deribit options")
    p.add_argument("--synthetic-call-panel", required=True, help="Call-equivalent quote panel in Parquet format")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--currency", default="BTC")
    p.add_argument("--snapshot-hour", type=int, default=12)
    p.add_argument("--month", action="append", default=[], help="YYYY-MM; repeatable")

    # DTE aliases.
    p.add_argument("--dte-min", type=float, default=None, help="Alias for --dte-min-repair")
    p.add_argument("--dte-max", type=float, default=None, help="Alias for --dte-max-repair")
    p.add_argument("--dte-min-repair", type=float, default=2.0)
    p.add_argument("--dte-max-repair", type=float, default=30.0)
    p.add_argument("--dte-min-marking", type=float, default=0.0)
    p.add_argument("--dte-max-marking", type=float, default=31.0)
    p.add_argument("--marking-extra-days", type=int, default=1)

    p.add_argument("--price-floor", type=float, default=0.0005)
    p.add_argument("--strike-tol", type=float, default=0.02)
    p.add_argument("--min-strikes", type=int, default=7)
    p.add_argument("--arms", nargs="+", default=["L1", "L2", "SMOOTH"],
                   help="Base arms. Supported: L1 L2 SMOOTH FILTER_L2 FILTER_SMOOTH IVSPLINE SVI ELASTIC_L2 plus generated SMOOTH_RHO_* and BUDGET_*.")
    p.add_argument("--add-arms", nargs="*", default=[], help="Additional raw arm names to append")
    p.add_argument("--include-broader-arms", action="store_true", help="Append FILTER_L2, IVSPLINE, SVI, ELASTIC_L2")
    p.add_argument("--smooth-rho", type=float, default=0.01, help="Default rho for a generic SMOOTH arm")
    p.add_argument("--smooth-rhos", nargs="*", type=float, default=[], help="Generate SMOOTH_RHO_* arms, e.g. 0.01 0.1 1 10")
    p.add_argument("--budget-smoothing", nargs="*", type=float, default=[], help="Generate BUDGET_* arms, e.g. 0.1 0.25 0.5")
    p.add_argument("--spline-smoothing", type=float, default=1e-4)
    p.add_argument("--filter-spread-quantile", type=float, default=0.75)
    p.add_argument("--filter-bad-weight", type=float, default=0.01)
    p.add_argument("--svi-max-nfev", type=int, default=2000)

    p.add_argument("--elastic-penalty", type=float, default=1000.0)
    p.add_argument("--elastic-box", action="store_true", help="Use an elastic bid-ask box for all repair arms.")
    p.add_argument("--qp-backend", choices=["cvxpy", "scipy"], default="cvxpy", help="QP backend for L2/SMOOTH")
    p.add_argument("--qp-solver", default="auto", help="cvxpy solver name or auto")
    p.add_argument("--solver-maxiter", type=int, default=800)

    p.add_argument("--ranking-score-method", choices=["relspread_h2", "abs_spread_h2"], default="relspread_h2")
    p.add_argument("--ranking-slice-agg", choices=["max", "p90", "median"], default="max")
    p.add_argument("--selected-slices", type=int, default=None, help="Select a fixed number of slices by the ranking score")
    p.add_argument("--max-slices", type=int, default=None, help="Optional random slice cap")
    p.add_argument("--random-state", type=int, default=7)
    p.add_argument("--hedge-tc-bps", type=float, default=0.0)
    p.add_argument("--allow-partial-arms", action="store_true",
                   help="Keep slices with any successful arm pairs. Used when an analysis includes repair rules with incomplete support.")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dte_min_repair = args.dte_min if args.dte_min is not None else args.dte_min_repair
    dte_max_repair = args.dte_max if args.dte_max is not None else args.dte_max_repair
    effective_arms = expand_arms_from_args(args)
    cfg = RepairConfig(
        currency=args.currency.upper(), snapshot_hour=args.snapshot_hour,
        dte_min_repair=dte_min_repair, dte_max_repair=dte_max_repair,
        dte_min_marking=args.dte_min_marking, dte_max_marking=args.dte_max_marking,
        marking_extra_days=args.marking_extra_days,
        price_floor=args.price_floor, strike_tol=args.strike_tol, min_strikes=args.min_strikes,
        arms=effective_arms, smooth_rho=args.smooth_rho,
        smooth_rhos=tuple(args.smooth_rhos), budget_smoothing=tuple(args.budget_smoothing),
        spline_smoothing=args.spline_smoothing, filter_spread_quantile=args.filter_spread_quantile,
        filter_bad_weight=args.filter_bad_weight, svi_max_nfev=args.svi_max_nfev,
        elastic_penalty=args.elastic_penalty, elastic_box=args.elastic_box,
        qp_backend=args.qp_backend, qp_solver=args.qp_solver, solver_maxiter=args.solver_maxiter,
        ranking_score_method=args.ranking_score_method, ranking_slice_agg=args.ranking_slice_agg,
        selected_slices=args.selected_slices, max_slices=args.max_slices,
        random_state=args.random_state, hedge_tc_bps=args.hedge_tc_bps,
        require_all_arms_for_main=not args.allow_partial_arms,
    )
    out_dir = ensure_dir(args.output_dir)
    (out_dir / "repair_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"[load] {args.synthetic_call_panel}", flush=True)
    print(f"[arms] {', '.join(cfg.arms)}", flush=True)

    panel_repair_raw = load_synthetic_panel(
        args.synthetic_call_panel, cfg, args.month,
        dte_min=cfg.dte_min_repair, dte_max=cfg.dte_max_repair,
        extra_days=0,
    )
    panel_marking_raw = load_synthetic_panel(
        args.synthetic_call_panel, cfg, args.month,
        dte_min=cfg.dte_min_marking, dte_max=cfg.dte_max_marking,
        extra_days=cfg.marking_extra_days,
    )
    if panel_repair_raw.empty:
        raise SystemExit("No repair rows after currency/hour/month/DTE filters. Check snapshot hour and repair DTE.")
    if panel_marking_raw.empty:
        raise SystemExit("No marking rows after currency/hour/month/DTE filters. Check marking DTE and months.")

    panel_repair_base = add_state_flags_base(panel_repair_raw, cfg)
    panel_marking = add_state_flags_base(panel_marking_raw, cfg)
    panel_repair_base.to_parquet(out_dir / "repair_panel_base.parquet", index=False)
    panel_marking.to_parquet(out_dir / "marking_panel_full.parquet", index=False)
    print(f"[panel] repair base rows={len(panel_repair_base):,}; slices={len(panel_repair_base.groupby(slice_key_cols())):,}", flush=True)
    print(f"[panel] marking rows={len(panel_marking):,}; slices={len(panel_marking.groupby(slice_key_cols())):,}", flush=True)

    panel_repair = select_repair_slices(panel_repair_base, cfg)
    panel_repair.to_parquet(out_dir / "repair_panel_selected.parquet", index=False)
    print(f"[panel] selected repair rows={len(panel_repair):,}; slices={len(panel_repair.groupby(slice_key_cols())) if len(panel_repair) else 0:,}", flush=True)
    if panel_repair.empty:
        raise SystemExit("No repair slices selected.")

    surfaces, diag = run_repairs(panel_repair, cfg)
    diag.to_parquet(out_dir / "repair_diagnostics.parquet", index=False)
    diag.to_csv(out_dir / "repair_diagnostics.csv", index=False)
    surfaces.to_parquet(out_dir / "repaired_surfaces.parquet", index=False)
    print(f"[repair] successful surface rows={len(surfaces):,}; diagnostics={len(diag):,}", flush=True)
    if surfaces.empty:
        print("No successful repair surfaces. Stop.", file=sys.stderr)
        return 2

    if cfg.require_all_arms_for_main:
        ok_counts = diag[diag["success"]].groupby(slice_key_cols())["arm"].nunique().reset_index(name="n_success_arms")
        ok_keys = set(map(tuple, ok_counts[ok_counts["n_success_arms"] >= len(cfg.arms)][slice_key_cols()].to_numpy()))
        before = len(surfaces)
        surfaces = surfaces[surfaces[slice_key_cols()].apply(lambda r: tuple(r) in ok_keys, axis=1)].copy()
        print(f"[filter] complete-arm surfaces {before:,} -> {len(surfaces):,}", flush=True)
        surfaces.to_parquet(out_dir / "repaired_surfaces_complete_arms.parquet", index=False)
        if surfaces.empty:
            print("No complete-arm slices after filtering; use --allow-partial-arms to retain partial support.", file=sys.stderr)
            return 3

    greeks = compute_greeks(surfaces)
    greeks.to_parquet(out_dir / "greeks.parquet", index=False)
    print(f"[greeks] rows={len(greeks):,}", flush=True)

    strike_disp = compute_strike_pairwise_dispersion(greeks, cfg)
    strike_disp.to_parquet(out_dir / "strike_pairwise_dispersion.parquet", index=False)
    strike_disp.to_csv(out_dir / "strike_pairwise_dispersion.csv", index=False)
    print(f"[strike dispersion] rows={len(strike_disp):,}", flush=True)

    book_outputs = compute_book_outputs(greeks, panel_marking, cfg)
    book_outputs.to_parquet(out_dir / "book_outputs.parquet", index=False)
    print(f"[books] rows={len(book_outputs):,}", flush=True)

    book_disp = compute_book_pairwise_dispersion(book_outputs)
    book_disp.to_parquet(out_dir / "book_pairwise_dispersion.parquet", index=False)
    book_disp.to_csv(out_dir / "book_pairwise_dispersion.csv", index=False)
    print(f"[book dispersion] rows={len(book_disp):,}", flush=True)

    summary = summarize(panel_marking, panel_repair, surfaces, diag, greeks, strike_disp, book_outputs, book_disp, cfg)
    summary.to_csv(out_dir / "repair_summary.csv", index=False)
    print("[summary]")
    print(summary.T.to_string(header=False))

    if len(strike_disp):
        state_summary = strike_disp.groupby(["high_score_cell", "dte_bucket"], dropna=False).agg(
            n=("gamma_rr", "size"),
            median_gamma_rr=("gamma_rr", "median"),
            p90_gamma_rr=("gamma_rr", lambda x: np.nanpercentile(x, 90)),
            median_gamma_diff_scaled=("gamma_diff_scaled", "median"),
            p90_gamma_diff_scaled=("gamma_diff_scaled", lambda x: np.nanpercentile(x, 90)),
            small_price_large_gamma_rate=("small_price_large_gamma", "mean"),
            median_price_diff_box=("price_diff_box", "median"),
            median_ranking_score=("ranking_score", "median") if "ranking_score" in strike_disp.columns else ("price_diff_box", "median"),
        ).reset_index()
        state_summary.to_csv(out_dir / "state_summary.csv", index=False)
    if len(diag):
        diag_summary = diag.groupby("arm").agg(
            attempts=("success", "size"), successes=("success", "sum"), success_rate=("success", "mean"),
            median_box_slack=("box_slack_sum", "median"), p90_box_slack=("box_slack_sum", lambda x: np.nanpercentile(x, 90)),
            median_mono_v=("max_mono_violation", "median"), median_conv_v=("max_conv_violation", "median")
        ).reset_index()
        diag_summary.to_csv(out_dir / "repair_arm_summary.csv", index=False)

    if not args.no_plots:
        make_figures(strike_disp, book_disp, greeks, out_dir)
    print(f"[done] outputs in {out_dir}", flush=True)
    return 0




# =============================================================================
# Analysis extensions
# =============================================================================


def _cfg_get(cfg: RepairConfig, name: str, default):
    return getattr(cfg, name, default)


def verify_solution(c: np.ndarray, bid: np.ndarray, ask: np.ndarray, intrinsic: np.ndarray, x_norm: np.ndarray,
                    tol: float = 2e-6, price_floor: float = 0.0005) -> Dict[str, float | bool]:
    """Check effective price bounds, call-spread bounds, and convexity."""
    c = np.asarray(c, dtype=float)
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    intrinsic = np.asarray(intrinsic, dtype=float)
    lower, upper = effective_bounds(x_norm, bid, ask, intrinsic)
    cert = certify_call_curve(c, x_norm, lower, upper, tolerance=tol)
    intr_v = np.nanmax(np.maximum(0.0, intrinsic - c)) if len(c) else 0.0
    upper_v = np.nanmax(np.maximum(0.0, c - 1.0)) if len(c) else 0.0
    box_low = np.maximum(0.0, bid - c)
    box_high = np.maximum(0.0, c - ask)
    scale = np.maximum(ask - bid, price_floor)
    box_slack_sum = float(np.nansum((box_low + box_high) / scale))
    feasible = bool(cert["certified_feasible"])
    return {
        "verified_feasible": feasible,
        "max_mono_violation": float(cert["max_monotonicity_violation"]),
        "max_conv_violation": float(cert["max_convexity_violation"]),
        "max_vertical_spread_violation": float(cert["max_vertical_spread_violation"]),
        "max_vertical_slope": float(cert["max_vertical_slope"]),
        "max_constraint_violation": float(cert["max_constraint_violation"]),
        "max_intrinsic_violation": float(intr_v),
        "max_upper_violation": float(upper_v),
        "box_slack_sum": box_slack_sum,
        "max_box_low_violation": float(np.nanmax(box_low) if len(box_low) else 0.0),
        "max_box_high_violation": float(np.nanmax(box_high) if len(box_high) else 0.0),
    }


def prepare_slice_arrays(slc: pd.DataFrame, cfg: RepairConfig) -> Dict[str, np.ndarray]:
    """Prepare finite arrays and effective price bounds for one slice."""
    slc = slc.sort_values("strike")
    K = slc["strike"].to_numpy(dtype=float)
    F = slc["forward_price"].to_numpy(dtype=float)
    X = slc["underlyer_spot"].to_numpy(dtype=float)
    tau = slc["tau"].to_numpy(dtype=float)
    bid_raw = slc["C_bid"].to_numpy(dtype=float)
    ask_raw = slc["C_ask"].to_numpy(dtype=float)
    mid_raw = slc["C_mid"].to_numpy(dtype=float)
    forward = float(np.nanmedian(F))
    x_norm = K / forward
    intrinsic = np.maximum(1.0 - x_norm, 0.0)

    bid = np.nan_to_num(bid_raw, nan=np.nan, neginf=np.nan, posinf=np.nan)
    ask = np.nan_to_num(ask_raw, nan=np.nan, neginf=np.nan, posinf=np.nan)
    bad_nonfinite = (~np.isfinite(bid)) | (~np.isfinite(ask))
    bad_inverted = np.isfinite(bid) & np.isfinite(ask) & (ask < bid)
    bad_box_count = int(np.sum(bad_nonfinite | bad_inverted))

    # Count invalid intervals before repairing their bounds.
    bid = np.where(np.isfinite(bid), bid, 0.0)
    ask = np.where(np.isfinite(ask), ask, 1.0)
    if np.any(ask < bid):
        m = 0.5 * (ask + bid)
        bad = ask < bid
        bid[bad] = m[bad]
        ask[bad] = m[bad]

    mid = np.nan_to_num(mid_raw, nan=0.5 * (bid + ask), neginf=0.0, posinf=1.0)
    mid_clip_count = int(np.sum((mid < 0.0) | (mid > 1.0)))
    mid = np.minimum(np.maximum(mid, 0.0), 1.0)
    scale = np.maximum(ask - bid, cfg.price_floor)
    forward_dispersion_rel = float(np.nanmax(np.abs(F - forward)) / max(abs(forward), EPS))
    w_l1 = 1.0 / scale
    w_l2 = 1.0 / (scale ** 2)
    if np.isfinite(w_l2).sum() >= 2:
        lo_q, hi_q = np.nanquantile(w_l2, [cfg.weight_clip_q_low, cfg.weight_clip_q_high])
        w_l2 = np.clip(w_l2, lo_q, hi_q)
    else:
        w_l2 = np.ones_like(scale)
    return dict(K=K, F=F, X=X, tau=tau, bid=bid, ask=ask, mid=mid, intrinsic=intrinsic,
                scale=scale, x_norm=x_norm, w_l1=w_l1, w_l2=w_l2,
                bad_box_count=bad_box_count, mid_clip_count=mid_clip_count,
                forward_dispersion_rel=forward_dispersion_rel)


def add_state_flags_base(panel: pd.DataFrame, cfg: RepairConfig) -> pd.DataFrame:
    """Compute state variables once on the base panel with finite interior-strike ranking scores."""
    df = panel.copy()
    if "relspread_floor" not in df.columns:
        df["relspread_floor"] = (df["C_ask"] - df["C_bid"]) / np.maximum(df["C_mid"].abs(), cfg.price_floor)
    if "dte_bucket" not in df.columns:
        df["dte_bucket"] = dte_bucket_from_dte(df["DTE"])
    df = df.sort_values(["currency", "date", "expiration_date", "strike"]).reset_index(drop=True)
    df["h_abs"] = np.nan
    df["h_norm"] = np.nan
    df["edge_strike"] = False

    for _, idx in df.groupby(["currency", "date", "expiration_date"], sort=False).groups.items():
        loc = np.asarray(list(idx))
        strikes = df.loc[loc, "strike"].to_numpy(dtype=float)
        F = float(df.loc[loc[0], "forward_price"])
        if len(strikes) >= 2:
            h = np.full(len(strikes), np.nan)
            h[0] = strikes[1] - strikes[0]
            h[-1] = strikes[-1] - strikes[-2]
            if len(strikes) >= 3:
                h[1:-1] = 0.5 * (strikes[2:] - strikes[:-2])
            df.loc[loc, "h_abs"] = h
            df.loc[loc, "h_norm"] = h / F if F > 0 else np.nan
            edge = np.zeros(len(strikes), dtype=bool)
            edge[0] = True
            edge[-1] = True
            df.loc[loc, "edge_strike"] = edge

    df["short_dte"] = df["DTE"] <= cfg.short_dte_max
    df["wing"] = df["k"].abs() >= cfg.wing_abs_k
    df["wide_spread"] = False
    df["sparse_strike"] = False
    global_wide = np.nanquantile(df["relspread_floor"], cfg.wide_quantile) if len(df) else np.nan
    global_sparse = np.nanquantile(df["h_norm"], cfg.sparse_quantile) if len(df) else np.nan
    for _, idx in df.groupby("dte_bucket", sort=False).groups.items():
        loc = np.asarray(list(idx))
        rs = df.loc[loc, "relspread_floor"].to_numpy(dtype=float)
        hs = df.loc[loc, "h_norm"].to_numpy(dtype=float)
        wide_thr = np.nanquantile(rs, cfg.wide_quantile) if np.isfinite(rs).sum() >= 20 else global_wide
        sparse_thr = np.nanquantile(hs, cfg.sparse_quantile) if np.isfinite(hs).sum() >= 20 else global_sparse
        df.loc[loc, "wide_spread"] = df.loc[loc, "relspread_floor"] >= wide_thr
        df.loc[loc, "sparse_strike"] = df.loc[loc, "h_norm"] >= sparse_thr

    box_width = (df["C_ask"] - df["C_bid"]).astype(float)
    h_abs = df["h_abs"].astype(float).replace(0.0, np.nan)
    h_norm = df["h_norm"].astype(float).replace(0.0, np.nan)
    df["ranking_score_abs_h2"] = box_width / np.maximum(h_abs ** 2, EPS)
    df["ranking_score_rel_h2"] = df["relspread_floor"] / np.maximum(h_norm ** 2, EPS)
    method = cfg.ranking_score_method.lower()
    df["ranking_score"] = df["ranking_score_abs_h2"] if method == "abs_spread_h2" else df["ranking_score_rel_h2"]

    # Rank only interior strikes with defined central curvature.
    invalid_ranking = df["edge_strike"].astype(bool) | (~np.isfinite(df["h_norm"])) | (df["h_norm"] <= 0)
    df.loc[invalid_ranking, ["ranking_score", "ranking_score_abs_h2", "ranking_score_rel_h2"]] = np.nan

    df["threshold_screen_cell"] = df["short_dte"] & df["wing"] & df["wide_spread"] & df["sparse_strike"]
    scores = df["ranking_score"].replace([np.inf, -np.inf], np.nan)
    score_thr = np.nanquantile(scores, 0.90) if np.isfinite(scores).sum() else np.nan
    df["high_score_cell"] = np.isfinite(df["ranking_score"]) & (df["ranking_score"] >= score_thr)
    return df


def select_repair_slices(panel_full: pd.DataFrame, cfg: RepairConfig, panel_helper: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Select repair slices and optional t+1 marking helpers."""
    df = panel_full.copy()
    grp = df.groupby(slice_key_cols(), sort=False)
    agg_func = _cfg_get(cfg, "ranking_slice_agg", "p90").lower()
    def _clean(x):
        return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if agg_func == "median":
        score_agg = ("ranking_score", lambda x: float(np.nanmedian(_clean(x))) if len(_clean(x)) else np.nan)
    elif agg_func == "max":
        score_agg = ("ranking_score", lambda x: float(np.nanmax(_clean(x))) if len(_clean(x)) else np.nan)
    else:
        score_agg = ("ranking_score", lambda x: float(np.nanpercentile(_clean(x), 90)) if len(_clean(x)) else np.nan)
    slice_stats = grp.agg(
        n_strikes=("strike", "size"),
        ranking_score=score_agg,
        threshold_screen_obs=("threshold_screen_cell", "sum"),
        max_relspread_floor=("relspread_floor", "max"),
        max_h_norm=("h_norm", "max"),
        min_h_norm=("h_norm", "min"),
        med_dte=("DTE", "median"),
        med_abs_k=("k", lambda x: float(np.nanmedian(np.abs(x))))
    ).reset_index()
    slice_stats = slice_stats[(slice_stats["n_strikes"] >= cfg.min_strikes) & np.isfinite(slice_stats["ranking_score"])].copy()
    if slice_stats.empty:
        return df.iloc[0:0].copy()

    rng = np.random.default_rng(cfg.random_state)
    if cfg.selected_slices is not None:
        slice_stats = slice_stats.sort_values("ranking_score", ascending=False).head(cfg.selected_slices)
    elif cfg.max_slices is not None and len(slice_stats) > cfg.max_slices:
        take = rng.choice(slice_stats.index.to_numpy(), size=cfg.max_slices, replace=False)
        slice_stats = slice_stats.loc[take]

    core_keys = set(map(tuple, slice_stats[slice_key_cols()].to_numpy()))
    mask_core = df[slice_key_cols()].apply(lambda r: tuple(r) in core_keys, axis=1)
    selected = df[mask_core].copy()
    selected["core_repair_slice"] = True

    if _cfg_get(cfg, "include_nextday_repair_slices", False):
        helper_source = panel_helper if panel_helper is not None else df
        helper_rows = []
        for _, r in slice_stats.iterrows():
            key_next = (r["currency"], int(r["date"]) + SECONDS_PER_DAY, r["expiration_date"])
            m = ((helper_source["currency"] == key_next[0]) &
                 (helper_source["date"].astype(int) == key_next[1]) &
                 (helper_source["expiration_date"].astype(int) == int(key_next[2])))
            if m.any():
                tmp = helper_source[m].copy()
                tmp["core_repair_slice"] = False
                helper_rows.append(tmp)
        if helper_rows:
            selected = pd.concat([selected] + helper_rows, ignore_index=True)
            selected = selected.drop_duplicates(["currency", "date", "expiration_date", "strike"], keep="first")
            selected["core_repair_slice"] = selected["core_repair_slice"].fillna(False).astype(bool)

    selected = selected.sort_values(slice_key_cols() + ["strike"]).reset_index(drop=True)
    selected = selected.merge(slice_stats[slice_key_cols() + ["ranking_score", "threshold_screen_obs"]].rename(columns={"ranking_score":"slice_ranking_score"}),
                              on=slice_key_cols(), how="left")
    selected["slice_ranking_score"] = selected["slice_ranking_score"].fillna(selected.get("ranking_score"))
    return selected


def _filtered_bad_mask_for_slice(slc: pd.DataFrame, cfg: RepairConfig) -> np.ndarray:
    rel = pd.to_numeric(slc.get("relspread_floor", pd.Series(np.nan, index=slc.index)), errors="coerce").to_numpy(dtype=float)
    thr = np.nanquantile(rel, cfg.filter_spread_quantile) if np.isfinite(rel).sum() >= 3 else np.nan
    bad = np.isfinite(rel) & np.isfinite(thr) & (rel >= thr)
    # Prefer volume and open interest; fall back to spread.
    for col in ["volume", "open_interest"]:
        if col in slc.columns:
            vals = pd.to_numeric(slc[col], errors="coerce").fillna(0).to_numpy(dtype=float)
            bad |= vals <= 0
    return bad


def solve_filter_relaxed_l2(arr: Dict[str, np.ndarray], slc: pd.DataFrame, cfg: RepairConfig, l1_c: Optional[np.ndarray]) -> Dict:
    """Downweight flagged quotes and relax their box constraints."""
    try:
        import cvxpy as cp  # type: ignore
    except Exception:
        return {"success": False, "status": "cvxpy_not_installed", "message": "FILTER_RELAXED_L2 requires cvxpy", "c": None}
    K, bid, ask, mid, intrinsic, x_norm = (
        arr[k] for k in ["K", "bid", "ask", "mid", "intrinsic", "x_norm"]
    )
    n = len(K)
    bad = _filtered_bad_mask_for_slice(slc, cfg)
    w = np.array(arr["w_l2"], dtype=float)
    w[bad] *= cfg.filter_bad_weight
    slack_mult = np.ones(n)
    slack_mult[bad] = max(float(cfg.filter_bad_weight), 1e-4)
    c = cp.Variable(n)
    xi_low = cp.Variable(n, nonneg=True)
    xi_high = cp.Variable(n, nonneg=True)
    full_A, full_b, _ = build_full_constraint_matrix(x_norm)
    constraints = [
        c >= intrinsic,
        c <= 1.0,
        c + xi_low >= bid,
        c - xi_high <= ask,
        full_A @ c <= full_b,
    ]
    slack_pen = cfg.elastic_penalty * cp.sum(cp.multiply(slack_mult / arr["scale"], xi_low + xi_high))
    obj = cp.sum(cp.multiply(w, cp.square(c - mid))) + slack_pen
    prob = cp.Problem(cp.Minimize(obj), constraints)
    solver_order = [cfg.qp_solver.upper()] if cfg.qp_solver.lower() != "auto" else ["OSQP", "CLARABEL", "ECOS", "SCS"]
    installed = set(cp.installed_solvers())
    last_msg = ""
    for solver in solver_order:
        if solver not in installed:
            continue
        try:
            if solver == "OSQP":
                prob.solve(solver=solver, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000, verbose=False)
            else:
                prob.solve(solver=solver, verbose=False)
            if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and c.value is not None:
                cval = np.asarray(c.value, dtype=float).reshape(-1)
                ver = verify_solution(cval, bid, ask, intrinsic, x_norm, price_floor=cfg.price_floor)
                success = bool(ver["verified_feasible"])
                return {"success": success, "status": prob.status, "message": f"cvxpy:{solver}; bad_quote_share={bad.mean():.4f}",
                        "objective": float(prob.value), "c": cval, "filtered_quote_share": float(bad.mean()), **ver}
            last_msg = str(prob.status)
        except Exception as e:
            last_msg = f"{solver}: {e}"
    return {"success": False, "status": "cvxpy_failed", "message": last_msg, "c": None, "filtered_quote_share": float(bad.mean())}


def solve_slice_repairs(slc: pd.DataFrame, cfg: RepairConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    slc = slc.sort_values("strike").copy()
    arr = prepare_slice_arrays(slc, cfg)
    n = len(arr["K"])
    if n < cfg.min_strikes:
        return pd.DataFrame(), pd.DataFrame()
    arms = tuple(a.upper() for a in cfg.arms)
    solutions: Dict[str, Dict] = {}
    l1_c = None
    if "L1" in arms:
        sol = solve_l1(arr, cfg)
        solutions["L1"] = sol
        if sol.get("success") and sol.get("c") is not None:
            l1_c = sol["c"]
    for arm in arms:
        if arm == "L1":
            continue
        if arm == "L2":
            solutions[arm] = solve_qp_like(arr, cfg, arm, l1_c)
        elif arm in {"SPREAD_WEIGHTED_L2", "FILTER_L2"}:
            w = filtered_weights_for_slice(slc, arr, cfg)
            solutions["SPREAD_WEIGHTED_L2" if arm == "FILTER_L2" else arm] = solve_qp_like(arr, cfg, arm, l1_c, target_weights=w)
        elif arm == "FILTER_RELAXED_L2":
            solutions[arm] = solve_filter_relaxed_l2(arr, slc, cfg, l1_c)
        elif arm == "SMOOTH" or arm.startswith("SMOOTH_RHO_"):
            rho = parse_smooth_rho_from_arm(arm, cfg)
            solutions[arm] = solve_qp_like(arr, cfg, arm, l1_c, smooth_rho=rho)
        elif arm.startswith("BUDGET_"):
            b = parse_budget_from_arm(arm)
            solutions[arm] = solve_budget_smoothing(arr, cfg, b, l1_c) if b is not None else {"success": False, "status": "bad_budget_arm", "message": arm, "c": None}
        elif arm == "IVSPLINE":
            target = ivspline_projected_target(arr, cfg)
            solutions[arm] = solve_qp_like(arr, cfg, arm, l1_c, target_mid=target, smooth_rho=cfg.smooth_rho) if target is not None else {"success": False, "status": "ivspline_failed", "message": "could not build IV spline target", "c": None}
        elif arm == "SVI":
            target = svi_projected_target(arr, cfg)
            solutions[arm] = solve_qp_like(arr, cfg, arm, l1_c, target_mid=target, smooth_rho=cfg.smooth_rho) if target is not None else {"success": False, "status": "svi_failed", "message": "could not fit SVI target", "c": None}
        elif arm == "ELASTIC_L2":
            cfg_el = replace(cfg, elastic_box=True)
            solutions[arm] = solve_qp_like(arr, cfg_el, arm, l1_c, cfg_override=cfg_el)
        else:
            solutions[arm] = {"success": False, "status": "unknown_arm", "message": f"Unknown arm {arm}", "c": None}

    key_vals = {c: slc.iloc[0][c] for c in slice_key_cols()}
    diag_rows, surface_rows = [], []
    for arm, sol in solutions.items():
        if _cfg_get(cfg, "exclude_optimal_inaccurate", False) and str(sol.get("status", "")).lower() == "optimal_inaccurate":
            sol = dict(sol)
            sol["success"] = False
            sol["message"] = str(sol.get("message", "")) + "; excluded optimal_inaccurate"
        diag = {
            **key_vals,
            "arm": arm,
            "n_strikes": n,
            "success": bool(sol.get("success", False)),
            "status": sol.get("status"),
            "solver_status_inaccurate": "inaccurate" in str(sol.get("status", "")).lower(),
            "message": sol.get("message"),
            "objective": finite(sol.get("objective")),
            "bad_box_count": int(arr.get("bad_box_count", 0)),
            "mid_clip_count": int(arr.get("mid_clip_count", 0)),
            "filtered_quote_share": finite(sol.get("filtered_quote_share"), np.nan),
        }
        for k in ["verified_feasible", "max_mono_violation", "max_conv_violation",
                  "max_vertical_spread_violation", "max_vertical_slope", "max_constraint_violation",
                  "max_intrinsic_violation", "max_upper_violation", "box_slack_sum",
                  "max_box_low_violation", "max_box_high_violation"]:
            diag[k] = sol.get(k, np.nan)
        diag_rows.append(diag)
        if not sol.get("success", False) or sol.get("c") is None:
            continue
        c_hat = np.asarray(sol["c"], dtype=float)
        for j, (_, row) in enumerate(slc.iterrows()):
            surface_rows.append({
                **{col: row[col] for col in [
                    "currency", "date", "expiration_date", "strike", "forward_price", "underlyer_spot", "tau", "DTE",
                    "dte_bucket", "k", "C_bid", "C_ask", "C_mid", "intrinsic", "relspread_floor", "h_abs", "h_norm",
                    "ranking_score", "ranking_score_rel_h2", "ranking_score_abs_h2", "slice_ranking_score", "threshold_screen_obs",
                    "short_dte", "wing", "wide_spread", "sparse_strike", "high_score_cell", "threshold_screen_cell", "edge_strike", "core_repair_slice"
                ] if col in row.index},
                "arm": arm,
                "C_repaired": float(c_hat[j]),
                "repair_abs_mid_diff": float(abs(c_hat[j] - arr["mid"][j])),
                "repair_norm_mid_diff": float(abs(c_hat[j] - arr["mid"][j]) / (arr["scale"][j] + EPS)),
            })
    return pd.DataFrame(surface_rows), pd.DataFrame(diag_rows)


def compute_strike_pairwise_dispersion(greeks: pd.DataFrame, cfg: RepairConfig) -> pd.DataFrame:
    if greeks.empty:
        return pd.DataFrame()
    rows = []
    group_cols = slice_key_cols() + ["strike"]
    # Slice-level scales.
    scale_map = {}
    for skey, s in greeks.groupby(slice_key_cols(), sort=False):
        def medabs(col):
            vals = pd.to_numeric(s[col], errors="coerce").to_numpy(dtype=float) if col in s else np.array([])
            val = np.nanmedian(np.abs(vals)) if np.isfinite(vals).sum() else np.nan
            return max(finite(val, 0.0), EPS)
        scale_map[skey] = {
            "gamma": medabs("gamma_usd"),
            "cxx": medabs("C_xx"),
            "rnd": medabs("rnd_proxy"),
        }
    for _, g in greeks.groupby(group_cols, sort=False):
        if g["arm"].nunique() < 2:
            continue
        skey = (g.iloc[0]["currency"], g.iloc[0]["date"], g.iloc[0]["expiration_date"])
        scales = scale_map.get(skey, {"gamma": EPS, "cxx": EPS, "rnd": EPS})
        records = {r["arm"]: r for _, r in g.iterrows()}
        arms = sorted(records)
        for a, b in combinations(arms, 2):
            ra, rb = records[a], records[b]
            price_diff = abs(float(ra["C_repaired"]) - float(rb["C_repaired"]))
            box_width = max(float(ra["C_ask"] - ra["C_bid"]), cfg.price_floor)
            gamma_diff = abs(finite(ra["gamma_usd"], np.nan) - finite(rb["gamma_usd"], np.nan))
            gamma_pair_scale = 0.5 * (abs(finite(ra["gamma_usd"], 0.0)) + abs(finite(rb["gamma_usd"], 0.0)))
            gamma_rr = gamma_diff / (gamma_pair_scale + EPS) if np.isfinite(gamma_diff) else np.nan
            gamma_diff_scaled = gamma_diff / (scales["gamma"] + EPS) if np.isfinite(gamma_diff) else np.nan
            ckk_diff = abs(finite(ra["C_KK"], np.nan) - finite(rb["C_KK"], np.nan))
            cxx_diff = abs(finite(ra["C_xx"], np.nan) - finite(rb["C_xx"], np.nan))
            rnd_diff = abs(finite(ra["rnd_proxy"], np.nan) - finite(rb["rnd_proxy"], np.nan))
            cxx_diff_scaled = cxx_diff / (scales["cxx"] + EPS) if np.isfinite(cxx_diff) else np.nan
            rnd_diff_scaled = rnd_diff / (scales["rnd"] + EPS) if np.isfinite(rnd_diff) else np.nan
            h_abs = finite(ra.get("h_abs"), np.nan)
            h_norm = finite(ra.get("h_norm"), np.nan)
            amp_h2 = ckk_diff * (h_abs ** 2) / (price_diff + EPS) if np.isfinite(ckk_diff) and np.isfinite(h_abs) else np.nan
            amp_x_h2 = cxx_diff * (h_norm ** 2) / (price_diff + EPS) if np.isfinite(cxx_diff) and np.isfinite(h_norm) else np.nan
            small_large_gamma = (
                (price_diff / box_width < cfg.price_diff_box_threshold)
                and np.isfinite(gamma_rr) and gamma_rr > cfg.gamma_rr_threshold
                and np.isfinite(gamma_diff_scaled) and gamma_diff_scaled > _cfg_get(cfg, "gamma_scaled_only_threshold", cfg.gamma_material_threshold)
            )
            small_large_curv = (
                (price_diff / box_width < cfg.price_diff_box_threshold)
                and np.isfinite(cxx_diff_scaled) and cxx_diff_scaled > _cfg_get(cfg, "cxx_material_threshold", cfg.gamma_material_threshold)
            )
            rows.append({
                "currency": ra["currency"], "date": int(ra["date"]), "expiration_date": int(ra["expiration_date"]),
                "strike": float(ra["strike"]), "arm_a": a, "arm_b": b,
                "DTE": float(ra["DTE"]), "dte_bucket": ra.get("dte_bucket"), "k": float(ra["k"]),
                "short_dte": bool(ra.get("short_dte", False)), "wing": bool(ra.get("wing", False)),
                "wide_spread": bool(ra.get("wide_spread", False)), "sparse_strike": bool(ra.get("sparse_strike", False)),
                "high_score_cell": bool(ra.get("high_score_cell", False)), "threshold_screen_cell": bool(ra.get("threshold_screen_cell", False)),
                "edge_strike": bool(ra.get("edge_strike", False)), "core_repair_slice": bool(ra.get("core_repair_slice", True)),
                "relspread_floor": float(ra.get("relspread_floor", np.nan)),
                "ranking_score": float(ra.get("ranking_score", np.nan)),
                "ranking_score_rel_h2": float(ra.get("ranking_score_rel_h2", np.nan)),
                "ranking_score_abs_h2": float(ra.get("ranking_score_abs_h2", np.nan)),
                "h_abs": h_abs, "h_norm": h_norm,
                "price_diff": price_diff,
                "price_diff_box": price_diff / (box_width + EPS),
                "gamma_diff": gamma_diff,
                "gamma_rr": gamma_rr,
                "gamma_diff_scaled": gamma_diff_scaled,
                "slice_gamma_scale": scales["gamma"],
                "ckk_diff": ckk_diff,
                "cxx_diff": cxx_diff,
                "cxx_diff_scaled": cxx_diff_scaled,
                "rnd_diff": rnd_diff,
                "rnd_diff_scaled": rnd_diff_scaled,
                "amp_h2": amp_h2,
                "amp_x_h2": amp_x_h2,
                "small_price_large_gamma": bool(small_large_gamma),
                "small_price_large_curvature": bool(small_large_curv),
            })
    return pd.DataFrame(rows)


def select_local_diagnostic_legs(slice_base: pd.DataFrame, cfg: RepairConfig) -> Optional[List[Dict]]:
    """Local butterfly centered on the highest-ranked interior strike."""
    slc = slice_base.sort_values("strike").reset_index(drop=True)
    eligible = slc[(~slc.get("edge_strike", False).astype(bool)) & np.isfinite(slc.get("ranking_score", np.nan))]
    if eligible.empty or len(slc) < 3:
        return None
    center_idx = int(eligible["ranking_score"].idxmax())
    pos = slc.index.get_loc(center_idx)
    if pos <= 0 or pos >= len(slc) - 1:
        return None
    center = slc.iloc[pos]
    claim = "P" if float(center["k"]) < 0 else "C"
    trip = [(pos - 1, 1.0), (pos, -2.0), (pos + 1, 1.0)]
    legs = []
    for p, qty in trip:
        row = slc.iloc[p]
        legs.append({"claim": claim, "target_k": float(row["k"]), "qty": qty, "strike": float(row["strike"]), "k": float(row["k"]), "strike_dist": 0.0})
    return legs


def compute_book_outputs(greeks: pd.DataFrame, panel_full: pd.DataFrame, cfg: RepairConfig) -> pd.DataFrame:
    if greeks.empty:
        return pd.DataFrame()
    rows = []
    base_cols = ["currency", "date", "expiration_date", "strike", "forward_price", "underlyer_spot", "tau", "DTE", "k", "C_mid", "ranking_score", "edge_strike"]
    base_cols = [c for c in base_cols if c in panel_full.columns]
    # Summarize core slices; retain helper slices only for t+1 marks.
    if "core_repair_slice" in greeks.columns:
        core_slice_keys = greeks.loc[greeks["core_repair_slice"].astype(bool), slice_key_cols()].drop_duplicates()
    else:
        core_slice_keys = greeks[slice_key_cols()].drop_duplicates()
    base = panel_full[base_cols].drop_duplicates(["currency", "date", "expiration_date", "strike"])
    if not core_slice_keys.empty:
        base = base.merge(core_slice_keys, on=slice_key_cols(), how="inner")
    else:
        base = base.iloc[0:0].copy()
    repaired_lookup = greeks.set_index(["currency", "date", "expiration_date", "strike", "arm"], drop=False)

    for key, base_slc in base.groupby(slice_key_cols(), sort=False):
        currency, date, expiration = key
        books_to_run = dict(BOOKS)
        books_to_run["LOCAL_DIAGNOSTIC_FLY"] = None
        for book_name in books_to_run:
            if book_name == "LOCAL_DIAGNOSTIC_FLY":
                legs = select_local_diagnostic_legs(base_slc, cfg)
            else:
                legs = select_book_legs(base_slc, book_name, cfg)
            if legs is None:
                continue
            next_date = int(date) + SECONDS_PER_DAY
            common_t, common_tp1 = [], []
            for leg in legs:
                common_t.append(common_mark_for_leg(panel_full, int(date), int(expiration), leg["strike"], leg["claim"]))
                common_tp1.append(common_mark_for_leg(panel_full, next_date, int(expiration), leg["strike"], leg["claim"]))
            nextday_available = all(m is not None for m in common_tp1) and all(m is not None for m in common_t)
            V_common_t = np.nan; V_common_tp1 = np.nan; X_tp1 = np.nan
            X_t = float(base_slc.iloc[0]["underlyer_spot"])
            if nextday_available:
                V_common_t = sum(leg["qty"] * m["price_usd"] for leg, m in zip(legs, common_t))
                V_common_tp1 = sum(leg["qty"] * m["price_usd"] for leg, m in zip(legs, common_tp1))
                X_tp1 = float(common_tp1[0]["X"])

            g_slice = greeks[(greeks["currency"] == currency) & (greeks["date"] == date) & (greeks["expiration_date"] == expiration)]
            for arm, g_arm in g_slice.groupby("arm", sort=False):
                V_rep_usd = 0.0; delta = 0.0; gamma = 0.0; leg_ok = True
                V_rep_tp1_usd = np.nan
                for leg in legs:
                    rr = g_arm[g_arm["strike"] == leg["strike"]]
                    if rr.empty:
                        leg_ok = False; break
                    r = rr.iloc[0]
                    c_call = float(r["C_repaired"]); F = float(r["forward_price"]); X = float(r["underlyer_spot"])
                    price_base = option_price_base_from_call(c_call, leg["strike"], F, leg["claim"])
                    d = float(r["delta_put_usd"] if str(leg["claim"]).upper().startswith("P") else r["delta_call_usd"])
                    gma = float(r["gamma_usd"])
                    V_rep_usd += leg["qty"] * price_base * X
                    delta += leg["qty"] * d
                    gamma += leg["qty"] * gma
                if not leg_ok:
                    continue
                # Repair-specific t+1 mark.
                pm_available = True
                vpm = 0.0
                for leg in legs:
                    idx = (currency, next_date, int(expiration), float(leg["strike"]), arm)
                    if idx not in repaired_lookup.index:
                        pm_available = False; break
                    rn = repaired_lookup.loc[idx]
                    if isinstance(rn, pd.DataFrame):
                        rn = rn.iloc[0]
                    c_next = float(rn["C_repaired"]); F_next = float(rn["forward_price"]); X_next = float(rn["underlyer_spot"])
                    pbase_next = option_price_base_from_call(c_next, leg["strike"], F_next, leg["claim"])
                    vpm += leg["qty"] * pbase_next * X_next
                if pm_available:
                    V_rep_tp1_usd = vpm
                hedge = -delta
                pnl_common = np.nan; loss_common = np.nan; pnl_pipeline = np.nan; loss_pipeline = np.nan; tc = 0.0
                if nextday_available and np.isfinite(X_tp1):
                    tc = cfg.hedge_tc_bps * 1e-4 * abs(hedge) * X_t
                    pnl_common = (V_common_tp1 - V_common_t) + hedge * (X_tp1 - X_t) - tc
                    loss_common = -pnl_common / max(abs(V_common_t), cfg.pnl_scale_floor_usd)
                if pm_available and np.isfinite(X_tp1):
                    pnl_pipeline = (V_rep_tp1_usd - V_rep_usd) + hedge * (X_tp1 - X_t) - tc
                    loss_pipeline = -pnl_pipeline / max(abs(V_rep_usd), cfg.pnl_scale_floor_usd)
                rows.append({
                    "currency": currency, "date": int(date), "expiration_date": int(expiration), "book": book_name, "arm": arm,
                    "DTE": float(base_slc.iloc[0]["DTE"]), "n_legs": len(legs),
                    "leg_strikes": json.dumps([leg["strike"] for leg in legs]),
                    "leg_claims": json.dumps([leg["claim"] for leg in legs]),
                    "leg_qty": json.dumps([leg["qty"] for leg in legs]),
                    "value_repaired_usd": V_rep_usd,
                    "value_repaired_next_usd": V_rep_tp1_usd,
                    "value_common_usd": V_common_t,
                    "delta_usd": delta,
                    "gamma_usd": gamma,
                    "hedge_underlying_units": hedge,
                    "nextday_available": bool(nextday_available),
                    "pipeline_nextday_available": bool(pm_available),
                    "pnl_common": pnl_common,
                    "loss_common": loss_common,
                    "pnl_pipeline": pnl_pipeline,
                    "loss_pipeline": loss_pipeline,
                    "tc_usd": tc,
                    "underlyer_spot": X_t,
                    "underlyer_spot_next": X_tp1,
                })
    return pd.DataFrame(rows)


def compute_book_pairwise_dispersion(book_outputs: pd.DataFrame) -> pd.DataFrame:
    if book_outputs.empty:
        return pd.DataFrame()
    rows = []
    for _, g in book_outputs.groupby(["currency", "date", "expiration_date", "book"], sort=False):
        if g["arm"].nunique() < 2:
            continue
        rec = {r["arm"]: r for _, r in g.iterrows()}
        arms = sorted(rec)
        gamma_scale = max(np.nanmedian(np.abs(g["gamma_usd"].to_numpy(dtype=float))), EPS)
        delta_scale = max(np.nanmedian(np.abs(g["delta_usd"].to_numpy(dtype=float))), EPS)
        for a, b in combinations(arms, 2):
            ra, rb = rec[a], rec[b]
            gamma_diff = abs(float(ra["gamma_usd"]) - float(rb["gamma_usd"]))
            delta_diff = abs(float(ra["delta_usd"]) - float(rb["delta_usd"]))
            hedge_diff = abs(float(ra["hedge_underlying_units"]) - float(rb["hedge_underlying_units"]))
            loss_common_diff = np.nan
            if np.isfinite(ra.get("loss_common")) and np.isfinite(rb.get("loss_common")):
                loss_common_diff = abs(float(ra["loss_common"]) - float(rb["loss_common"]))
            loss_pipeline_diff = np.nan
            if np.isfinite(ra.get("loss_pipeline")) and np.isfinite(rb.get("loss_pipeline")):
                loss_pipeline_diff = abs(float(ra["loss_pipeline"]) - float(rb["loss_pipeline"]))
            rows.append({
                "currency": ra["currency"], "date": int(ra["date"]), "expiration_date": int(ra["expiration_date"]),
                "book": ra["book"], "arm_a": a, "arm_b": b, "DTE": float(ra["DTE"]),
                "delta_diff": delta_diff,
                "delta_rr": delta_diff / (delta_scale + EPS),
                "gamma_diff": gamma_diff,
                "gamma_rr": gamma_diff / (gamma_scale + EPS),
                "hedge_diff": hedge_diff,
                "loss_common_diff": loss_common_diff,
                "loss_pipeline_diff": loss_pipeline_diff,
                "nextday_available": bool(ra["nextday_available"] and rb["nextday_available"]),
                "pipeline_nextday_available": bool(ra.get("pipeline_nextday_available", False) and rb.get("pipeline_nextday_available", False)),
            })
    return pd.DataFrame(rows)


def make_arm_pair_summary(strike_disp: pd.DataFrame, book_disp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if len(strike_disp):
        for (a, b), g in strike_disp.groupby(["arm_a", "arm_b"], sort=False):
            rows.append({
                "level": "strike", "arm_a": a, "arm_b": b, "book": "",
                "n": len(g), "n_core": int(g.get("core_repair_slice", pd.Series(True, index=g.index)).astype(bool).sum()),
                "p50_gamma_rr": float(np.nanmedian(g["gamma_rr"])),
                "p90_gamma_rr": float(np.nanpercentile(g["gamma_rr"], 90)),
                "p50_gamma_diff_scaled": float(np.nanmedian(g["gamma_diff_scaled"])),
                "p90_gamma_diff_scaled": float(np.nanpercentile(g["gamma_diff_scaled"], 90)),
                "p50_cxx_diff_scaled": float(np.nanmedian(g["cxx_diff_scaled"])),
                "p90_cxx_diff_scaled": float(np.nanpercentile(g["cxx_diff_scaled"], 90)),
                "p50_rnd_diff_scaled": float(np.nanmedian(g["rnd_diff_scaled"])),
                "p90_rnd_diff_scaled": float(np.nanpercentile(g["rnd_diff_scaled"], 90)),
                "small_price_large_gamma_rate": float(np.nanmean(g["small_price_large_gamma"])),
                "small_price_large_curvature_rate": float(np.nanmean(g["small_price_large_curvature"])),
            })
    if len(book_disp):
        for (a, b, book), g in book_disp.groupby(["arm_a", "arm_b", "book"], sort=False):
            rows.append({
                "level": "book", "arm_a": a, "arm_b": b, "book": book,
                "n": len(g), "n_core": len(g),
                "p50_gamma_rr": float(np.nanmedian(g["gamma_rr"])),
                "p90_gamma_rr": float(np.nanpercentile(g["gamma_rr"], 90)),
                "p50_delta_rr": float(np.nanmedian(g["delta_rr"])),
                "p90_delta_rr": float(np.nanpercentile(g["delta_rr"], 90)),
                "p90_loss_common_diff": float(np.nanpercentile(g["loss_common_diff"], 90)),
                "p90_loss_pipeline_diff": float(np.nanpercentile(g["loss_pipeline_diff"], 90)) if "loss_pipeline_diff" in g else np.nan,
                "common_nextday_rate": float(np.nanmean(g["nextday_available"])),
                "pipeline_nextday_rate": float(np.nanmean(g.get("pipeline_nextday_available", pd.Series(False, index=g.index)))),
            })
    return pd.DataFrame(rows)


def make_arm_pair_common_support(diag: pd.DataFrame) -> pd.DataFrame:
    if diag.empty:
        return pd.DataFrame()
    rows = []
    all_arms = sorted(diag["arm"].dropna().unique())
    success_by_slice = {}
    for key, g in diag.groupby(slice_key_cols(), sort=False):
        success_by_slice[key] = set(g.loc[g["success"].astype(bool), "arm"])
    for a, b in combinations(all_arms, 2):
        both = sum(1 for s in success_by_slice.values() if a in s and b in s)
        either = sum(1 for s in success_by_slice.values() if a in s or b in s)
        rows.append({"arm_a": a, "arm_b": b, "common_success_slices": both, "either_success_slices": either})
    return pd.DataFrame(rows)


def summarize(panel_full: pd.DataFrame, panel_repair: pd.DataFrame, repairs: pd.DataFrame, diag: pd.DataFrame,
              greeks: pd.DataFrame, strike_disp: pd.DataFrame, book_outputs: pd.DataFrame,
              book_disp: pd.DataFrame, cfg: RepairConfig) -> pd.DataFrame:
    requested_arms = tuple(a.upper() for a in cfg.arms)
    attempted = len(diag)
    successes = int(diag["success"].sum()) if len(diag) and "success" in diag else 0
    n_slices = len(panel_repair.groupby(slice_key_cols())) if len(panel_repair) else 0
    n_core_slices = len(panel_repair[panel_repair.get("core_repair_slice", True).astype(bool)].groupby(slice_key_cols())) if len(panel_repair) and "core_repair_slice" in panel_repair else n_slices
    success_rate = successes / attempted if attempted else np.nan
    complete = np.nan
    if len(diag):
        ok = diag[diag["success"]].groupby(slice_key_cols())["arm"].nunique()
        complete = float((ok >= len(requested_arms)).mean()) if len(ok) else 0.0
    strike_main = strike_disp[strike_disp.get("core_repair_slice", pd.Series(True, index=strike_disp.index)).astype(bool)] if len(strike_disp) else pd.DataFrame()
    high_score = strike_main[strike_main.get("high_score_cell", False).astype(bool)] if len(strike_main) and "high_score_cell" in strike_main else pd.DataFrame()
    summary = {
        "currency": cfg.currency.upper(), "snapshot_hour": cfg.snapshot_hour,
        "dte_min_repair": cfg.dte_min_repair, "dte_max_repair": cfg.dte_max_repair,
        "dte_min_marking": cfg.dte_min_marking, "dte_max_marking": cfg.dte_max_marking,
        "requested_arms": ",".join(requested_arms),
        "panel_full_rows": len(panel_full), "panel_repair_rows": len(panel_repair),
        "candidate_slices": n_slices, "core_candidate_slices": n_core_slices,
        "repair_attempts": attempted, "repair_successes": successes, "repair_success_rate": success_rate,
        "complete_slice_rate": complete, "repaired_surface_rows": len(repairs), "greek_rows": len(greeks),
        "strike_pairwise_rows": len(strike_disp), "strike_pairwise_core_rows": len(strike_main),
        "book_output_rows": len(book_outputs), "book_pairwise_rows": len(book_disp),
        "optimal_inaccurate_rate": float(np.nanmean(diag["solver_status_inaccurate"])) if len(diag) and "solver_status_inaccurate" in diag else np.nan,
        "bad_box_attempt_rate": float(np.nanmean(pd.to_numeric(diag.get("bad_box_count", pd.Series(0, index=diag.index)), errors="coerce") > 0)) if len(diag) else np.nan,
    }
    metrics = ["gamma_rr", "gamma_diff_scaled", "cxx_diff_scaled", "rnd_diff_scaled", "amp_x_h2", "price_diff_box"]
    for name, df in [("all", strike_main), ("high_score", high_score)]:
        if len(df):
            for m in metrics:
                if m in df:
                    summary[f"{name}_median_{m}"] = float(np.nanmedian(df[m]))
                    summary[f"{name}_p90_{m}"] = float(np.nanpercentile(df[m], 90))
            summary[f"{name}_small_price_large_gamma_rate"] = float(np.nanmean(df["small_price_large_gamma"]))
            summary[f"{name}_small_price_large_curvature_rate"] = float(np.nanmean(df["small_price_large_curvature"]))
        else:
            for m in metrics:
                summary[f"{name}_median_{m}"] = np.nan
                summary[f"{name}_p90_{m}"] = np.nan
            summary[f"{name}_small_price_large_gamma_rate"] = np.nan
            summary[f"{name}_small_price_large_curvature_rate"] = np.nan
    if len(book_disp):
        summary["book_p90_gamma_rr"] = float(np.nanpercentile(book_disp["gamma_rr"], 90))
        summary["book_p90_delta_rr"] = float(np.nanpercentile(book_disp["delta_rr"], 90))
        summary["book_p90_loss_common_diff"] = float(np.nanpercentile(book_disp["loss_common_diff"], 90))
        summary["book_p90_loss_pipeline_diff"] = float(np.nanpercentile(book_disp["loss_pipeline_diff"], 90)) if "loss_pipeline_diff" in book_disp else np.nan
        summary["book_nextday_pair_rate"] = float(np.nanmean(book_disp["nextday_available"]))
        summary["book_pipeline_nextday_pair_rate"] = float(np.nanmean(book_disp.get("pipeline_nextday_available", pd.Series(False, index=book_disp.index))))
    else:
        summary["book_p90_gamma_rr"] = summary["book_p90_delta_rr"] = summary["book_p90_loss_common_diff"] = summary["book_p90_loss_pipeline_diff"] = np.nan
        summary["book_nextday_pair_rate"] = summary["book_pipeline_nextday_pair_rate"] = np.nan

    min_slices = _cfg_get(cfg, "min_analysis_slices", 30)
    min_strike_pairs = _cfg_get(cfg, "min_strike_pairs", 500)
    min_book_pairs = _cfg_get(cfg, "min_book_pairs", 50)
    summary["pass_sample_size"] = bool((n_core_slices >= min_slices) and (len(strike_main) >= min_strike_pairs) and (len(book_disp) >= min_book_pairs))
    summary["pass_greek_rr"] = bool((finite(summary.get("high_score_p90_gamma_rr"), -np.inf) > cfg.gamma_rr_threshold) or (finite(summary.get("high_score_median_gamma_rr"), -np.inf) > 0.10))
    summary["pass_material_scaled"] = bool(finite(summary.get("high_score_p90_gamma_diff_scaled"), -np.inf) > cfg.gamma_material_threshold)
    summary["pass_curvature_scaled"] = bool(finite(summary.get("high_score_p90_cxx_diff_scaled"), -np.inf) > _cfg_get(cfg, "cxx_material_threshold", cfg.gamma_material_threshold))
    summary["pass_small_price_large_gamma"] = bool(finite(summary.get("high_score_small_price_large_gamma_rate"), 0.0) > 0.0)
    summary["pass_small_price_large_curvature"] = bool(finite(summary.get("high_score_small_price_large_curvature_rate"), 0.0) > 0.0)
    summary["analysis_signal"] = bool(
        summary["pass_sample_size"] and
        ((summary["pass_greek_rr"] and summary["pass_material_scaled"] and summary["pass_small_price_large_gamma"]) or
         (summary["pass_curvature_scaled"] and summary["pass_small_price_large_curvature"]))
    )
    return pd.DataFrame([summary])


def expand_arms_from_args(args) -> Tuple[str, ...]:
    arms = [a.upper() for a in args.arms]
    for rho in getattr(args, "smooth_rhos", []) or []:
        arms.append(f"SMOOTH_RHO_{label_float(rho).upper()}")
    for b in getattr(args, "budget_smoothing", []) or []:
        arms.append(f"BUDGET_{label_float(b).upper()}")
    for a in getattr(args, "add_arms", []) or []:
        arms.append(a.upper())
    if getattr(args, "include_broader_arms", False):
        arms.extend(["SPREAD_WEIGHTED_L2", "FILTER_RELAXED_L2", "IVSPLINE", "SVI", "ELASTIC_L2"])
    # FILTER_L2 uses weighted L2.
    arms = ["SPREAD_WEIGHTED_L2" if a == "FILTER_L2" else a for a in arms]
    seen, out = set(), []
    for a in arms:
        if a not in seen:
            out.append(a); seen.add(a)
    return tuple(out)


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Repair-risk analysis for Deribit options")
    p.add_argument("--synthetic-call-panel", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--currency", choices=["BTC", "ETH"], default="BTC")
    p.add_argument("--snapshot-hour", type=int, default=12)
    p.add_argument("--month", nargs="*", default=[])
    p.add_argument("--dte-min", type=float, default=None, help="Minimum DTE for repair slices")
    p.add_argument("--dte-max", type=float, default=None, help="Maximum DTE for repair slices")
    p.add_argument("--dte-min-repair", type=float, default=2.0)
    p.add_argument("--dte-max-repair", type=float, default=30.0)
    p.add_argument("--dte-min-marking", type=float, default=0.0)
    p.add_argument("--dte-max-marking", type=float, default=31.0)
    p.add_argument("--marking-extra-days", type=int, default=1)
    p.add_argument("--include-nextday-repair-slices", action="store_true", help="Also repair t+1 helper slices so pipeline-mark P&L can be populated.")
    p.add_argument("--price-floor", type=float, default=0.0005)
    p.add_argument("--strike-tol", type=float, default=0.02)
    p.add_argument("--min-strikes", type=int, default=7)
    p.add_argument("--arms", nargs="+", default=["L1", "L2", "SMOOTH"],
                   help="Supported: L1 L2 SMOOTH SPREAD_WEIGHTED_L2 FILTER_RELAXED_L2 IVSPLINE SVI ELASTIC_L2 plus SMOOTH_RHO_* and BUDGET_*.")
    p.add_argument("--add-arms", nargs="*", default=[])
    p.add_argument("--include-broader-arms", action="store_true", help="Append SPREAD_WEIGHTED_L2, FILTER_RELAXED_L2, IVSPLINE, SVI, ELASTIC_L2")
    p.add_argument("--smooth-rho", type=float, default=0.01)
    p.add_argument("--smooth-rhos", nargs="*", type=float, default=[])
    p.add_argument("--budget-smoothing", nargs="*", type=float, default=[])
    p.add_argument("--spline-smoothing", type=float, default=1e-4)
    p.add_argument("--filter-spread-quantile", type=float, default=0.75)
    p.add_argument("--filter-bad-weight", type=float, default=0.01)
    p.add_argument("--svi-max-nfev", type=int, default=2000)
    p.add_argument("--elastic-penalty", type=float, default=1000.0)
    p.add_argument("--elastic-box", action="store_true")
    p.add_argument("--qp-backend", choices=["cvxpy", "scipy"], default="cvxpy")
    p.add_argument("--qp-solver", default="auto")
    p.add_argument("--solver-maxiter", type=int, default=800)
    p.add_argument("--exclude-optimal-inaccurate", action="store_true")
    p.add_argument("--ranking-score-method", choices=["relspread_h2", "abs_spread_h2"], default="relspread_h2")
    p.add_argument("--ranking-slice-agg", choices=["max", "p90", "median"], default="p90")
    p.add_argument("--selected-slices", type=int, default=None)
    p.add_argument("--max-slices", type=int, default=None)
    p.add_argument("--random-state", type=int, default=7)
    p.add_argument("--hedge-tc-bps", type=float, default=0.0)
    p.add_argument("--allow-partial-arms", action="store_true")
    p.add_argument("--min-analysis-slices", type=int, default=30)
    p.add_argument("--min-strike-pairs", type=int, default=500)
    p.add_argument("--min-book-pairs", type=int, default=50)
    p.add_argument("--cxx-material-threshold", type=float, default=0.05)
    p.add_argument("--gamma-scaled-only-threshold", type=float, default=0.05, help="Scaled-gamma threshold used with price_diff_box.")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dte_min_repair = args.dte_min if args.dte_min is not None else args.dte_min_repair
    dte_max_repair = args.dte_max if args.dte_max is not None else args.dte_max_repair
    effective_arms = expand_arms_from_args(args)
    cfg = RepairConfig(
        currency=args.currency.upper(), snapshot_hour=args.snapshot_hour,
        dte_min_repair=dte_min_repair, dte_max_repair=dte_max_repair,
        dte_min_marking=args.dte_min_marking, dte_max_marking=args.dte_max_marking,
        marking_extra_days=args.marking_extra_days,
        price_floor=args.price_floor, strike_tol=args.strike_tol, min_strikes=args.min_strikes,
        arms=effective_arms, smooth_rho=args.smooth_rho,
        smooth_rhos=tuple(args.smooth_rhos), budget_smoothing=tuple(args.budget_smoothing),
        spline_smoothing=args.spline_smoothing, filter_spread_quantile=args.filter_spread_quantile,
        filter_bad_weight=args.filter_bad_weight, svi_max_nfev=args.svi_max_nfev,
        elastic_penalty=args.elastic_penalty, elastic_box=args.elastic_box,
        qp_backend=args.qp_backend, qp_solver=args.qp_solver, solver_maxiter=args.solver_maxiter,
        ranking_score_method=args.ranking_score_method, ranking_slice_agg=args.ranking_slice_agg,
        selected_slices=args.selected_slices, max_slices=args.max_slices,
        random_state=args.random_state, hedge_tc_bps=args.hedge_tc_bps,
        require_all_arms_for_main=not args.allow_partial_arms,
    )
    cfg.include_nextday_repair_slices = bool(args.include_nextday_repair_slices)
    cfg.exclude_optimal_inaccurate = bool(args.exclude_optimal_inaccurate)
    cfg.min_analysis_slices = int(args.min_analysis_slices)
    cfg.min_strike_pairs = int(args.min_strike_pairs)
    cfg.min_book_pairs = int(args.min_book_pairs)
    cfg.cxx_material_threshold = float(args.cxx_material_threshold)

    out_dir = ensure_dir(args.output_dir)
    cfg_dict = asdict(cfg)
    for k in ["include_nextday_repair_slices", "exclude_optimal_inaccurate", "min_analysis_slices", "min_strike_pairs", "min_book_pairs", "cxx_material_threshold"]:
        cfg_dict[k] = getattr(cfg, k)
    (out_dir / "repair_config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
    print(f"[load] {args.synthetic_call_panel}", flush=True)
    print(f"[arms] {', '.join(cfg.arms)}", flush=True)

    panel_repair_raw = load_synthetic_panel(args.synthetic_call_panel, cfg, args.month,
                                            dte_min=cfg.dte_min_repair, dte_max=cfg.dte_max_repair,
                                            extra_days=0)
    panel_marking_raw = load_synthetic_panel(args.synthetic_call_panel, cfg, args.month,
                                             dte_min=cfg.dte_min_marking, dte_max=cfg.dte_max_marking,
                                             extra_days=cfg.marking_extra_days)
    if panel_repair_raw.empty:
        raise SystemExit("No repair rows after currency/hour/month/DTE filters. Check snapshot hour and repair DTE.")
    if panel_marking_raw.empty:
        raise SystemExit("No marking rows after currency/hour/month/DTE filters. Check marking DTE and months.")

    panel_repair_base = add_state_flags_base(panel_repair_raw, cfg)
    panel_marking = add_state_flags_base(panel_marking_raw, cfg)
    panel_repair_base.to_parquet(out_dir / "repair_panel_base.parquet", index=False)
    panel_marking.to_parquet(out_dir / "marking_panel_full.parquet", index=False)
    print(f"[panel] repair base rows={len(panel_repair_base):,}; slices={len(panel_repair_base.groupby(slice_key_cols())):,}", flush=True)
    print(f"[panel] marking rows={len(panel_marking):,}; slices={len(panel_marking.groupby(slice_key_cols())):,}", flush=True)

    panel_repair = select_repair_slices(panel_repair_base, cfg, panel_helper=panel_marking)
    panel_repair.to_parquet(out_dir / "repair_panel_selected.parquet", index=False)
    print(f"[panel] selected repair rows={len(panel_repair):,}; slices={len(panel_repair.groupby(slice_key_cols())) if len(panel_repair) else 0:,}", flush=True)
    if panel_repair.empty:
        raise SystemExit("No repair slices selected.")

    surfaces, diag = run_repairs(panel_repair, cfg)
    diag.to_parquet(out_dir / "repair_diagnostics.parquet", index=False)
    diag.to_csv(out_dir / "repair_diagnostics.csv", index=False)
    surfaces.to_parquet(out_dir / "repaired_surfaces.parquet", index=False)
    print(f"[repair] successful surface rows={len(surfaces):,}; diagnostics={len(diag):,}", flush=True)
    if surfaces.empty:
        print("No successful repair surfaces. Stop.", file=sys.stderr)
        return 2

    if cfg.require_all_arms_for_main:
        ok_counts = diag[diag["success"]].groupby(slice_key_cols())["arm"].nunique().reset_index(name="n_success_arms")
        ok_keys = set(map(tuple, ok_counts[ok_counts["n_success_arms"] >= len(cfg.arms)][slice_key_cols()].to_numpy()))
        before = len(surfaces)
        surfaces = surfaces[surfaces[slice_key_cols()].apply(lambda r: tuple(r) in ok_keys, axis=1)].copy()
        print(f"[filter] complete-arm surfaces {before:,} -> {len(surfaces):,}", flush=True)
        surfaces.to_parquet(out_dir / "repaired_surfaces_complete_arms.parquet", index=False)
        if surfaces.empty:
            print("No complete-arm slices after filtering; use --allow-partial-arms to retain partial support.", file=sys.stderr)
            return 3

    greeks = compute_greeks(surfaces)
    greeks.to_parquet(out_dir / "greeks.parquet", index=False)
    print(f"[greeks] rows={len(greeks):,}", flush=True)

    strike_disp = compute_strike_pairwise_dispersion(greeks, cfg)
    strike_disp.to_parquet(out_dir / "strike_pairwise_dispersion.parquet", index=False)
    strike_disp.to_csv(out_dir / "strike_pairwise_dispersion.csv", index=False)
    print(f"[strike dispersion] rows={len(strike_disp):,}", flush=True)

    book_outputs = compute_book_outputs(greeks, panel_marking, cfg)
    book_outputs.to_parquet(out_dir / "book_outputs.parquet", index=False)
    print(f"[books] rows={len(book_outputs):,}", flush=True)

    book_disp = compute_book_pairwise_dispersion(book_outputs)
    book_disp.to_parquet(out_dir / "book_pairwise_dispersion.parquet", index=False)
    book_disp.to_csv(out_dir / "book_pairwise_dispersion.csv", index=False)
    print(f"[book dispersion] rows={len(book_disp):,}", flush=True)

    arm_pair_summary = make_arm_pair_summary(strike_disp, book_disp)
    arm_pair_summary.to_csv(out_dir / "arm_pair_summary.csv", index=False)
    arm_pair_support = make_arm_pair_common_support(diag)
    arm_pair_support.to_csv(out_dir / "arm_pair_common_support.csv", index=False)

    summary = summarize(panel_marking, panel_repair, surfaces, diag, greeks, strike_disp, book_outputs, book_disp, cfg)
    summary.to_csv(out_dir / "repair_summary.csv", index=False)
    print("[summary]")
    print(summary.T.to_string(header=False))

    if len(strike_disp):
        main_strike = strike_disp[strike_disp.get("core_repair_slice", pd.Series(True, index=strike_disp.index)).astype(bool)]
        state_summary = main_strike.groupby(["high_score_cell", "dte_bucket"], dropna=False).agg(
            n=("gamma_rr", "size"),
            median_gamma_rr=("gamma_rr", "median"),
            p90_gamma_rr=("gamma_rr", lambda x: np.nanpercentile(x, 90)),
            median_gamma_diff_scaled=("gamma_diff_scaled", "median"),
            p90_gamma_diff_scaled=("gamma_diff_scaled", lambda x: np.nanpercentile(x, 90)),
            median_cxx_diff_scaled=("cxx_diff_scaled", "median"),
            p90_cxx_diff_scaled=("cxx_diff_scaled", lambda x: np.nanpercentile(x, 90)),
            median_rnd_diff_scaled=("rnd_diff_scaled", "median"),
            p90_rnd_diff_scaled=("rnd_diff_scaled", lambda x: np.nanpercentile(x, 90)),
            small_price_large_gamma_rate=("small_price_large_gamma", "mean"),
            small_price_large_curvature_rate=("small_price_large_curvature", "mean"),
            median_price_diff_box=("price_diff_box", "median"),
            median_ranking_score=("ranking_score", "median"),
        ).reset_index()
        state_summary.to_csv(out_dir / "state_summary.csv", index=False)
    if len(diag):
        diag_summary = diag.groupby("arm").agg(
            attempts=("success", "size"), successes=("success", "sum"), success_rate=("success", "mean"),
            optimal_inaccurate_rate=("solver_status_inaccurate", "mean"),
            median_box_slack=("box_slack_sum", "median"), p90_box_slack=("box_slack_sum", lambda x: np.nanpercentile(x, 90)),
            median_mono_v=("max_mono_violation", "median"), median_conv_v=("max_conv_violation", "median"),
            bad_box_attempt_rate=("bad_box_count", lambda x: float(np.nanmean(pd.to_numeric(x, errors="coerce") > 0))),
            median_filtered_share=("filtered_quote_share", "median")
        ).reset_index()
        diag_summary.to_csv(out_dir / "repair_arm_summary.csv", index=False)

    if not args.no_plots:
        make_figures(strike_disp, book_disp, greeks, out_dir)
    print(f"[done] outputs in {out_dir}", flush=True)
    return 0

# =============================================================================
# Hedge and curvature diagnostics
# =============================================================================
# Repair solvers remain unchanged.


def compute_strike_pairwise_dispersion(greeks: pd.DataFrame, cfg: RepairConfig) -> pd.DataFrame:
    """Compute pairwise strike-level dispersion diagnostics."""
    if greeks.empty:
        return pd.DataFrame()
    rows = []
    group_cols = slice_key_cols() + ["strike"]
    scale_map = {}
    for skey, s in greeks.groupby(slice_key_cols(), sort=False):
        def medabs(col):
            vals = pd.to_numeric(s[col], errors="coerce").to_numpy(dtype=float) if col in s else np.array([])
            val = np.nanmedian(np.abs(vals)) if np.isfinite(vals).sum() else np.nan
            return max(finite(val, 0.0), EPS)
        scale_map[skey] = {
            "gamma": medabs("gamma_usd"),
            "cxx": medabs("C_xx"),
            "rnd": medabs("rnd_proxy"),
        }
    for _, g in greeks.groupby(group_cols, sort=False):
        if g["arm"].nunique() < 2:
            continue
        skey = (g.iloc[0]["currency"], g.iloc[0]["date"], g.iloc[0]["expiration_date"])
        scales = scale_map.get(skey, {"gamma": EPS, "cxx": EPS, "rnd": EPS})
        records = {r["arm"]: r for _, r in g.iterrows()}
        arms = sorted(records)
        for a, b in combinations(arms, 2):
            ra, rb = records[a], records[b]
            price_diff = abs(float(ra["C_repaired"]) - float(rb["C_repaired"]))
            box_width = max(float(ra["C_ask"] - ra["C_bid"]), cfg.price_floor)
            price_diff_box = price_diff / (box_width + EPS)

            gamma_diff = abs(finite(ra["gamma_usd"], np.nan) - finite(rb["gamma_usd"], np.nan))
            gamma_pair_scale = 0.5 * (abs(finite(ra["gamma_usd"], 0.0)) + abs(finite(rb["gamma_usd"], 0.0)))
            gamma_rr = gamma_diff / (gamma_pair_scale + EPS) if np.isfinite(gamma_diff) else np.nan
            gamma_diff_scaled = gamma_diff / (scales["gamma"] + EPS) if np.isfinite(gamma_diff) else np.nan

            ckk_diff = abs(finite(ra["C_KK"], np.nan) - finite(rb["C_KK"], np.nan))
            cxx_diff = abs(finite(ra["C_xx"], np.nan) - finite(rb["C_xx"], np.nan))
            rnd_diff = abs(finite(ra["rnd_proxy"], np.nan) - finite(rb["rnd_proxy"], np.nan))
            cxx_diff_scaled = cxx_diff / (scales["cxx"] + EPS) if np.isfinite(cxx_diff) else np.nan
            rnd_diff_scaled = rnd_diff / (scales["rnd"] + EPS) if np.isfinite(rnd_diff) else np.nan
            h_abs = finite(ra.get("h_abs"), np.nan)
            h_norm = finite(ra.get("h_norm"), np.nan)
            amp_h2 = ckk_diff * (h_abs ** 2) / (price_diff + EPS) if np.isfinite(ckk_diff) and np.isfinite(h_abs) else np.nan
            amp_x_h2 = cxx_diff * (h_norm ** 2) / (price_diff + EPS) if np.isfinite(cxx_diff) and np.isfinite(h_norm) else np.nan

            price_small = price_diff_box < cfg.price_diff_box_threshold
            small_large_gamma_joint = (
                price_small
                and np.isfinite(gamma_rr) and gamma_rr > cfg.gamma_rr_threshold
                and np.isfinite(gamma_diff_scaled) and gamma_diff_scaled > cfg.gamma_material_threshold
            )
            small_large_gamma_scaled = (
                price_small
                and np.isfinite(gamma_diff_scaled) and gamma_diff_scaled > _cfg_get(cfg, "gamma_scaled_only_threshold", cfg.gamma_material_threshold)
            )
            small_large_curv = (
                price_small
                and np.isfinite(cxx_diff_scaled) and cxx_diff_scaled > _cfg_get(cfg, "cxx_material_threshold", cfg.gamma_material_threshold)
            )
            rows.append({
                "currency": ra["currency"], "date": int(ra["date"]), "expiration_date": int(ra["expiration_date"]),
                "strike": float(ra["strike"]), "arm_a": a, "arm_b": b,
                "DTE": float(ra["DTE"]), "dte_bucket": ra.get("dte_bucket"), "k": float(ra["k"]),
                "short_dte": bool(ra.get("short_dte", False)), "wing": bool(ra.get("wing", False)),
                "wide_spread": bool(ra.get("wide_spread", False)), "sparse_strike": bool(ra.get("sparse_strike", False)),
                "high_score_cell": bool(ra.get("high_score_cell", False)), "threshold_screen_cell": bool(ra.get("threshold_screen_cell", False)),
                "edge_strike": bool(ra.get("edge_strike", False)), "core_repair_slice": bool(ra.get("core_repair_slice", True)),
                "relspread_floor": float(ra.get("relspread_floor", np.nan)),
                "ranking_score": float(ra.get("ranking_score", np.nan)),
                "ranking_score_rel_h2": float(ra.get("ranking_score_rel_h2", np.nan)),
                "ranking_score_abs_h2": float(ra.get("ranking_score_abs_h2", np.nan)),
                "h_abs": h_abs, "h_norm": h_norm,
                "price_diff": price_diff,
                "price_diff_box": price_diff_box,
                "gamma_diff": gamma_diff,
                "gamma_rr": gamma_rr,
                "gamma_diff_scaled": gamma_diff_scaled,
                "slice_gamma_scale": scales["gamma"],
                "ckk_diff": ckk_diff,
                "cxx_diff": cxx_diff,
                "cxx_diff_scaled": cxx_diff_scaled,
                "rnd_diff": rnd_diff,
                "rnd_diff_scaled": rnd_diff_scaled,
                "amp_h2": amp_h2,
                "amp_x_h2": amp_x_h2,
                "small_price_large_gamma": bool(small_large_gamma_joint),
                "small_price_large_gamma_scaled": bool(small_large_gamma_scaled),
                "small_price_large_curvature": bool(small_large_curv),
            })
    return pd.DataFrame(rows)


def compute_book_pairwise_dispersion(book_outputs: pd.DataFrame) -> pd.DataFrame:
    """Compute book-level hedge slippage and gamma P&L proxies."""
    if book_outputs.empty:
        return pd.DataFrame()
    rows = []
    for _, g in book_outputs.groupby(["currency", "date", "expiration_date", "book"], sort=False):
        if g["arm"].nunique() < 2:
            continue
        rec = {r["arm"]: r for _, r in g.iterrows()}
        arms = sorted(rec)
        gamma_scale = max(np.nanmedian(np.abs(g["gamma_usd"].to_numpy(dtype=float))), EPS)
        delta_scale = max(np.nanmedian(np.abs(g["delta_usd"].to_numpy(dtype=float))), EPS)
        for a, b in combinations(arms, 2):
            ra, rb = rec[a], rec[b]
            gamma_diff = abs(float(ra["gamma_usd"]) - float(rb["gamma_usd"]))
            delta_diff = abs(float(ra["delta_usd"]) - float(rb["delta_usd"]))
            hedge_diff = abs(float(ra["hedge_underlying_units"]) - float(rb["hedge_underlying_units"]))
            loss_common_diff = np.nan
            if np.isfinite(ra.get("loss_common")) and np.isfinite(rb.get("loss_common")):
                loss_common_diff = abs(float(ra["loss_common"]) - float(rb["loss_common"]))
            loss_pipeline_diff = np.nan
            if np.isfinite(ra.get("loss_pipeline")) and np.isfinite(rb.get("loss_pipeline")):
                loss_pipeline_diff = abs(float(ra["loss_pipeline"]) - float(rb["loss_pipeline"]))

            x_t = finite(ra.get("underlyer_spot"), np.nan)
            x_next = finite(ra.get("underlyer_spot_next"), np.nan)
            dS_abs = abs(x_next - x_t) if np.isfinite(x_t) and np.isfinite(x_next) else np.nan
            value_scale_candidates = [
                abs(finite(ra.get("value_common_usd"), np.nan)),
                abs(finite(rb.get("value_common_usd"), np.nan)),
                abs(finite(ra.get("value_repaired_usd"), np.nan)),
                abs(finite(rb.get("value_repaired_usd"), np.nan)),
                1.0,
            ]
            pnl_scale = max([v for v in value_scale_candidates if np.isfinite(v)] or [1.0])
            hedge_slippage_scaled = hedge_diff * dS_abs / (pnl_scale + EPS) if np.isfinite(dS_abs) else np.nan
            gamma_pnl_proxy = 0.5 * gamma_diff * (dS_abs ** 2) / (pnl_scale + EPS) if np.isfinite(dS_abs) else np.nan

            rows.append({
                "currency": ra["currency"], "date": int(ra["date"]), "expiration_date": int(ra["expiration_date"]),
                "book": ra["book"], "arm_a": a, "arm_b": b, "DTE": float(ra["DTE"]),
                "delta_diff": delta_diff,
                "delta_rr": delta_diff / (delta_scale + EPS),
                "gamma_diff": gamma_diff,
                "gamma_rr": gamma_diff / (gamma_scale + EPS),
                "hedge_diff": hedge_diff,
                "underlyer_move_abs": dS_abs,
                "pnl_scale_usd": pnl_scale,
                "hedge_slippage_scaled": hedge_slippage_scaled,
                "gamma_pnl_proxy": gamma_pnl_proxy,
                "loss_common_diff": loss_common_diff,
                "loss_pipeline_diff": loss_pipeline_diff,
                "nextday_available": bool(ra["nextday_available"] and rb["nextday_available"]),
                "pipeline_nextday_available": bool(ra.get("pipeline_nextday_available", False) and rb.get("pipeline_nextday_available", False)),
            })
    return pd.DataFrame(rows)


def make_arm_pair_summary(strike_disp: pd.DataFrame, book_disp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if len(strike_disp):
        for (a, b), g in strike_disp.groupby(["arm_a", "arm_b"], sort=False):
            rows.append({
                "level": "strike", "arm_a": a, "arm_b": b, "book": "",
                "n": len(g), "n_core": int(g.get("core_repair_slice", pd.Series(True, index=g.index)).astype(bool).sum()),
                "p50_gamma_rr": float(np.nanmedian(g["gamma_rr"])),
                "p90_gamma_rr": float(np.nanpercentile(g["gamma_rr"], 90)),
                "p50_gamma_diff_scaled": float(np.nanmedian(g["gamma_diff_scaled"])),
                "p90_gamma_diff_scaled": float(np.nanpercentile(g["gamma_diff_scaled"], 90)),
                "p50_cxx_diff_scaled": float(np.nanmedian(g["cxx_diff_scaled"])),
                "p90_cxx_diff_scaled": float(np.nanpercentile(g["cxx_diff_scaled"], 90)),
                "p50_rnd_diff_scaled": float(np.nanmedian(g["rnd_diff_scaled"])),
                "p90_rnd_diff_scaled": float(np.nanpercentile(g["rnd_diff_scaled"], 90)),
                "small_price_large_gamma_rate": float(np.nanmean(g["small_price_large_gamma"])),
                "small_price_large_gamma_scaled_rate": float(np.nanmean(g["small_price_large_gamma_scaled"])) if "small_price_large_gamma_scaled" in g else np.nan,
                "small_price_large_curvature_rate": float(np.nanmean(g["small_price_large_curvature"])),
            })
    if len(book_disp):
        for (a, b, book), g in book_disp.groupby(["arm_a", "arm_b", "book"], sort=False):
            rows.append({
                "level": "book", "arm_a": a, "arm_b": b, "book": book,
                "n": len(g), "n_core": len(g),
                "p50_gamma_rr": float(np.nanmedian(g["gamma_rr"])),
                "p90_gamma_rr": float(np.nanpercentile(g["gamma_rr"], 90)),
                "p50_delta_rr": float(np.nanmedian(g["delta_rr"])),
                "p90_delta_rr": float(np.nanpercentile(g["delta_rr"], 90)),
                "p90_hedge_slippage_scaled": float(np.nanpercentile(g["hedge_slippage_scaled"], 90)) if "hedge_slippage_scaled" in g else np.nan,
                "p90_gamma_pnl_proxy": float(np.nanpercentile(g["gamma_pnl_proxy"], 90)) if "gamma_pnl_proxy" in g else np.nan,
                "p90_loss_common_diff": float(np.nanpercentile(g["loss_common_diff"], 90)),
                "p90_loss_pipeline_diff": float(np.nanpercentile(g["loss_pipeline_diff"], 90)) if "loss_pipeline_diff" in g else np.nan,
                "common_nextday_rate": float(np.nanmean(g["nextday_available"])),
                "pipeline_nextday_rate": float(np.nanmean(g.get("pipeline_nextday_available", pd.Series(False, index=g.index)))),
            })
    return pd.DataFrame(rows)


def make_protocol_comparison_summary(strike_disp: pd.DataFrame) -> pd.DataFrame:
    """Compare the sparsity screen with the theory-aligned spread/h^2 score."""
    if strike_disp.empty:
        return pd.DataFrame()
    df0 = strike_disp[strike_disp.get("core_repair_slice", pd.Series(True, index=strike_disp.index)).astype(bool)].copy()
    groups = {
        "all_core": pd.Series(True, index=df0.index),
        "threshold_spacing_screen": df0.get("threshold_screen_cell", pd.Series(False, index=df0.index)).astype(bool),
        "spread_mesh_screen": df0.get("high_score_cell", pd.Series(False, index=df0.index)).astype(bool),
        "threshold_only": df0.get("threshold_screen_cell", pd.Series(False, index=df0.index)).astype(bool) & ~df0.get("high_score_cell", pd.Series(False, index=df0.index)).astype(bool),
        "score_only": df0.get("high_score_cell", pd.Series(False, index=df0.index)).astype(bool) & ~df0.get("threshold_screen_cell", pd.Series(False, index=df0.index)).astype(bool),
    }
    rows = []
    for name, mask in groups.items():
        g = df0[mask]
        row = {"protocol_group": name, "n": len(g)}
        if len(g):
            for col in ["gamma_rr", "gamma_diff_scaled", "cxx_diff_scaled", "rnd_diff_scaled", "amp_x_h2", "price_diff_box"]:
                if col in g:
                    row[f"median_{col}"] = float(np.nanmedian(g[col]))
                    row[f"p90_{col}"] = float(np.nanpercentile(g[col], 90))
            for col in ["small_price_large_gamma", "small_price_large_gamma_scaled", "small_price_large_curvature"]:
                if col in g:
                    row[f"{col}_rate"] = float(np.nanmean(g[col]))
            row["median_ranking_score"] = float(np.nanmedian(g["ranking_score"])) if "ranking_score" in g else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(panel_full: pd.DataFrame, panel_repair: pd.DataFrame, repairs: pd.DataFrame, diag: pd.DataFrame,
              greeks: pd.DataFrame, strike_disp: pd.DataFrame, book_outputs: pd.DataFrame,
              book_disp: pd.DataFrame, cfg: RepairConfig) -> pd.DataFrame:
    requested_arms = tuple(a.upper() for a in cfg.arms)
    attempted = len(diag)
    successes = int(diag["success"].sum()) if len(diag) and "success" in diag else 0
    n_slices = len(panel_repair.groupby(slice_key_cols())) if len(panel_repair) else 0
    n_core_slices = len(panel_repair[panel_repair.get("core_repair_slice", True).astype(bool)].groupby(slice_key_cols())) if len(panel_repair) and "core_repair_slice" in panel_repair else n_slices
    success_rate = successes / attempted if attempted else np.nan
    complete = np.nan
    if len(diag):
        ok = diag[diag["success"]].groupby(slice_key_cols())["arm"].nunique()
        complete = float((ok >= len(requested_arms)).mean()) if len(ok) else 0.0
    strike_main = strike_disp[strike_disp.get("core_repair_slice", pd.Series(True, index=strike_disp.index)).astype(bool)] if len(strike_disp) else pd.DataFrame()
    high_score = strike_main[strike_main.get("high_score_cell", False).astype(bool)] if len(strike_main) and "high_score_cell" in strike_main else pd.DataFrame()
    summary = {
        "currency": cfg.currency.upper(), "snapshot_hour": cfg.snapshot_hour,
        "dte_min_repair": cfg.dte_min_repair, "dte_max_repair": cfg.dte_max_repair,
        "dte_min_marking": cfg.dte_min_marking, "dte_max_marking": cfg.dte_max_marking,
        "requested_arms": ",".join(requested_arms),
        "panel_full_rows": len(panel_full), "panel_repair_rows": len(panel_repair),
        "candidate_slices": n_slices, "core_candidate_slices": n_core_slices,
        "repair_attempts": attempted, "repair_successes": successes, "repair_success_rate": success_rate,
        "complete_slice_rate": complete, "repaired_surface_rows": len(repairs), "greek_rows": len(greeks),
        "strike_pairwise_rows": len(strike_disp), "strike_pairwise_core_rows": len(strike_main),
        "book_output_rows": len(book_outputs), "book_pairwise_rows": len(book_disp),
        "optimal_inaccurate_rate": float(np.nanmean(diag["solver_status_inaccurate"])) if len(diag) and "solver_status_inaccurate" in diag else np.nan,
        "bad_box_attempt_rate": float(np.nanmean(pd.to_numeric(diag.get("bad_box_count", pd.Series(0, index=diag.index)), errors="coerce") > 0)) if len(diag) else np.nan,
    }
    metrics = ["gamma_rr", "gamma_diff_scaled", "cxx_diff_scaled", "rnd_diff_scaled", "amp_x_h2", "price_diff_box"]
    for name, df in [("all", strike_main), ("high_score", high_score)]:
        if len(df):
            for m in metrics:
                if m in df:
                    summary[f"{name}_median_{m}"] = float(np.nanmedian(df[m]))
                    summary[f"{name}_p90_{m}"] = float(np.nanpercentile(df[m], 90))
            summary[f"{name}_small_price_large_gamma_rate"] = float(np.nanmean(df["small_price_large_gamma"]))
            summary[f"{name}_small_price_large_gamma_scaled_rate"] = float(np.nanmean(df["small_price_large_gamma_scaled"])) if "small_price_large_gamma_scaled" in df else np.nan
            summary[f"{name}_small_price_large_curvature_rate"] = float(np.nanmean(df["small_price_large_curvature"]))
        else:
            for m in metrics:
                summary[f"{name}_median_{m}"] = np.nan
                summary[f"{name}_p90_{m}"] = np.nan
            summary[f"{name}_small_price_large_gamma_rate"] = np.nan
            summary[f"{name}_small_price_large_gamma_scaled_rate"] = np.nan
            summary[f"{name}_small_price_large_curvature_rate"] = np.nan
    if len(book_disp):
        summary["book_p90_gamma_rr"] = float(np.nanpercentile(book_disp["gamma_rr"], 90))
        summary["book_p90_delta_rr"] = float(np.nanpercentile(book_disp["delta_rr"], 90))
        summary["book_p90_hedge_slippage_scaled"] = float(np.nanpercentile(book_disp["hedge_slippage_scaled"], 90)) if "hedge_slippage_scaled" in book_disp else np.nan
        summary["book_p90_gamma_pnl_proxy"] = float(np.nanpercentile(book_disp["gamma_pnl_proxy"], 90)) if "gamma_pnl_proxy" in book_disp else np.nan
        summary["book_p90_loss_common_diff"] = float(np.nanpercentile(book_disp["loss_common_diff"], 90))
        summary["book_p90_loss_pipeline_diff"] = float(np.nanpercentile(book_disp["loss_pipeline_diff"], 90)) if "loss_pipeline_diff" in book_disp else np.nan
        summary["book_nextday_pair_rate"] = float(np.nanmean(book_disp["nextday_available"]))
        summary["book_pipeline_nextday_pair_rate"] = float(np.nanmean(book_disp.get("pipeline_nextday_available", pd.Series(False, index=book_disp.index))))
    else:
        summary["book_p90_gamma_rr"] = summary["book_p90_delta_rr"] = np.nan
        summary["book_p90_hedge_slippage_scaled"] = summary["book_p90_gamma_pnl_proxy"] = np.nan
        summary["book_p90_loss_common_diff"] = summary["book_p90_loss_pipeline_diff"] = np.nan
        summary["book_nextday_pair_rate"] = summary["book_pipeline_nextday_pair_rate"] = np.nan

    min_slices = _cfg_get(cfg, "min_analysis_slices", 30)
    min_strike_pairs = _cfg_get(cfg, "min_strike_pairs", 500)
    min_book_pairs = _cfg_get(cfg, "min_book_pairs", 50)
    summary["pass_sample_size"] = bool((n_core_slices >= min_slices) and (len(strike_main) >= min_strike_pairs) and (len(book_disp) >= min_book_pairs))
    summary["pass_greek_rr"] = bool((finite(summary.get("high_score_p90_gamma_rr"), -np.inf) > cfg.gamma_rr_threshold) or (finite(summary.get("high_score_median_gamma_rr"), -np.inf) > 0.10))
    summary["pass_material_scaled"] = bool(finite(summary.get("high_score_p90_gamma_diff_scaled"), -np.inf) > cfg.gamma_material_threshold)
    summary["pass_curvature_scaled"] = bool(finite(summary.get("high_score_p90_cxx_diff_scaled"), -np.inf) > _cfg_get(cfg, "cxx_material_threshold", cfg.gamma_material_threshold))
    summary["pass_small_price_large_gamma"] = bool(finite(summary.get("high_score_small_price_large_gamma_rate"), 0.0) > 0.0)
    summary["pass_small_price_large_gamma_scaled"] = bool(finite(summary.get("high_score_small_price_large_gamma_scaled_rate"), 0.0) > 0.0)
    summary["pass_small_price_large_curvature"] = bool(finite(summary.get("high_score_small_price_large_curvature_rate"), 0.0) > 0.0)
    summary["joint_gamma_signal"] = bool(
        summary["pass_sample_size"] and summary["pass_greek_rr"] and summary["pass_material_scaled"] and summary["pass_small_price_large_gamma"]
    )
    summary["scaled_hedge_signal"] = bool(
        summary["pass_sample_size"] and summary["pass_material_scaled"] and summary["pass_small_price_large_gamma_scaled"]
    )
    summary["curvature_signal"] = bool(
        summary["pass_sample_size"] and summary["pass_curvature_scaled"] and summary["pass_small_price_large_curvature"]
    )
    # Union of diagnostic signals.
    summary["analysis_signal"] = bool(
        summary["joint_gamma_signal"] or summary["scaled_hedge_signal"] or summary["curvature_signal"]
    )
    return pd.DataFrame([summary])


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dte_min_repair = args.dte_min if args.dte_min is not None else args.dte_min_repair
    dte_max_repair = args.dte_max if args.dte_max is not None else args.dte_max_repair
    effective_arms = expand_arms_from_args(args)
    cfg = RepairConfig(
        currency=args.currency.upper(), snapshot_hour=args.snapshot_hour,
        dte_min_repair=dte_min_repair, dte_max_repair=dte_max_repair,
        dte_min_marking=args.dte_min_marking, dte_max_marking=args.dte_max_marking,
        marking_extra_days=args.marking_extra_days,
        price_floor=args.price_floor, strike_tol=args.strike_tol, min_strikes=args.min_strikes,
        arms=effective_arms, smooth_rho=args.smooth_rho,
        smooth_rhos=tuple(args.smooth_rhos), budget_smoothing=tuple(args.budget_smoothing),
        spline_smoothing=args.spline_smoothing, filter_spread_quantile=args.filter_spread_quantile,
        filter_bad_weight=args.filter_bad_weight, svi_max_nfev=args.svi_max_nfev,
        elastic_penalty=args.elastic_penalty, elastic_box=args.elastic_box,
        qp_backend=args.qp_backend, qp_solver=args.qp_solver, solver_maxiter=args.solver_maxiter,
        ranking_score_method=args.ranking_score_method, ranking_slice_agg=args.ranking_slice_agg,
        selected_slices=args.selected_slices, max_slices=args.max_slices,
        random_state=args.random_state, hedge_tc_bps=args.hedge_tc_bps,
        require_all_arms_for_main=not args.allow_partial_arms,
    )
    cfg.include_nextday_repair_slices = bool(args.include_nextday_repair_slices)
    cfg.exclude_optimal_inaccurate = bool(args.exclude_optimal_inaccurate)
    cfg.min_analysis_slices = int(args.min_analysis_slices)
    cfg.min_strike_pairs = int(args.min_strike_pairs)
    cfg.min_book_pairs = int(args.min_book_pairs)
    cfg.cxx_material_threshold = float(args.cxx_material_threshold)
    cfg.gamma_scaled_only_threshold = float(args.gamma_scaled_only_threshold)

    out_dir = ensure_dir(args.output_dir)
    cfg_dict = asdict(cfg)
    for k in ["include_nextday_repair_slices", "exclude_optimal_inaccurate", "min_analysis_slices", "min_strike_pairs", "min_book_pairs", "cxx_material_threshold", "gamma_scaled_only_threshold"]:
        cfg_dict[k] = getattr(cfg, k)
    (out_dir / "repair_config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
    print(f"[load] {args.synthetic_call_panel}", flush=True)
    print(f"[arms] {', '.join(cfg.arms)}", flush=True)

    panel_repair_raw = load_synthetic_panel(args.synthetic_call_panel, cfg, args.month,
                                            dte_min=cfg.dte_min_repair, dte_max=cfg.dte_max_repair,
                                            extra_days=0)
    panel_marking_raw = load_synthetic_panel(args.synthetic_call_panel, cfg, args.month,
                                             dte_min=cfg.dte_min_marking, dte_max=cfg.dte_max_marking,
                                             extra_days=cfg.marking_extra_days)
    if panel_repair_raw.empty:
        raise SystemExit("No repair rows after currency/hour/month/DTE filters. Check snapshot hour and repair DTE.")
    if panel_marking_raw.empty:
        raise SystemExit("No marking rows after currency/hour/month/DTE filters. Check marking DTE and months.")

    panel_repair_base = add_state_flags_base(panel_repair_raw, cfg)
    panel_marking = add_state_flags_base(panel_marking_raw, cfg)
    panel_repair_base.to_parquet(out_dir / "repair_panel_base.parquet", index=False)
    panel_marking.to_parquet(out_dir / "marking_panel_full.parquet", index=False)
    print(f"[panel] repair base rows={len(panel_repair_base):,}; slices={len(panel_repair_base.groupby(slice_key_cols())):,}", flush=True)
    print(f"[panel] marking rows={len(panel_marking):,}; slices={len(panel_marking.groupby(slice_key_cols())):,}", flush=True)

    panel_repair = select_repair_slices(panel_repair_base, cfg, panel_helper=panel_marking)
    panel_repair.to_parquet(out_dir / "repair_panel_selected.parquet", index=False)
    print(f"[panel] selected repair rows={len(panel_repair):,}; slices={len(panel_repair.groupby(slice_key_cols())) if len(panel_repair) else 0:,}", flush=True)
    if panel_repair.empty:
        raise SystemExit("No repair slices selected.")

    surfaces, diag = run_repairs(panel_repair, cfg)
    diag.to_parquet(out_dir / "repair_diagnostics.parquet", index=False)
    diag.to_csv(out_dir / "repair_diagnostics.csv", index=False)
    surfaces.to_parquet(out_dir / "repaired_surfaces.parquet", index=False)
    print(f"[repair] successful surface rows={len(surfaces):,}; diagnostics={len(diag):,}", flush=True)
    if surfaces.empty:
        print("No successful repair surfaces. Stop.", file=sys.stderr)
        return 2

    if cfg.require_all_arms_for_main:
        ok_counts = diag[diag["success"]].groupby(slice_key_cols())["arm"].nunique().reset_index(name="n_success_arms")
        ok_keys = set(map(tuple, ok_counts[ok_counts["n_success_arms"] >= len(cfg.arms)][slice_key_cols()].to_numpy()))
        before = len(surfaces)
        surfaces = surfaces[surfaces[slice_key_cols()].apply(lambda r: tuple(r) in ok_keys, axis=1)].copy()
        print(f"[filter] complete-arm surfaces {before:,} -> {len(surfaces):,}", flush=True)
        surfaces.to_parquet(out_dir / "repaired_surfaces_complete_arms.parquet", index=False)
        if surfaces.empty:
            print("No complete-arm slices after filtering; use --allow-partial-arms to retain partial support.", file=sys.stderr)
            return 3

    greeks = compute_greeks(surfaces)
    greeks.to_parquet(out_dir / "greeks.parquet", index=False)
    print(f"[greeks] rows={len(greeks):,}", flush=True)

    strike_disp = compute_strike_pairwise_dispersion(greeks, cfg)
    strike_disp.to_parquet(out_dir / "strike_pairwise_dispersion.parquet", index=False)
    strike_disp.to_csv(out_dir / "strike_pairwise_dispersion.csv", index=False)
    print(f"[strike dispersion] rows={len(strike_disp):,}", flush=True)

    book_outputs = compute_book_outputs(greeks, panel_marking, cfg)
    book_outputs.to_parquet(out_dir / "book_outputs.parquet", index=False)
    print(f"[books] rows={len(book_outputs):,}", flush=True)

    book_disp = compute_book_pairwise_dispersion(book_outputs)
    book_disp.to_parquet(out_dir / "book_pairwise_dispersion.parquet", index=False)
    book_disp.to_csv(out_dir / "book_pairwise_dispersion.csv", index=False)
    print(f"[book dispersion] rows={len(book_disp):,}", flush=True)

    arm_pair_summary = make_arm_pair_summary(strike_disp, book_disp)
    arm_pair_summary.to_csv(out_dir / "arm_pair_summary.csv", index=False)
    arm_pair_support = make_arm_pair_common_support(diag)
    arm_pair_support.to_csv(out_dir / "arm_pair_common_support.csv", index=False)
    protocol_summary = make_protocol_comparison_summary(strike_disp)
    protocol_summary.to_csv(out_dir / "protocol_comparison_summary.csv", index=False)

    summary = summarize(panel_marking, panel_repair, surfaces, diag, greeks, strike_disp, book_outputs, book_disp, cfg)
    summary.to_csv(out_dir / "repair_summary.csv", index=False)
    print("[summary]")
    print(summary.T.to_string(header=False))

    if len(strike_disp):
        main_strike = strike_disp[strike_disp.get("core_repair_slice", pd.Series(True, index=strike_disp.index)).astype(bool)]
        state_summary = main_strike.groupby(["high_score_cell", "threshold_screen_cell", "dte_bucket"], dropna=False).agg(
            n=("gamma_rr", "size"),
            median_gamma_rr=("gamma_rr", "median"),
            p90_gamma_rr=("gamma_rr", lambda x: np.nanpercentile(x, 90)),
            median_gamma_diff_scaled=("gamma_diff_scaled", "median"),
            p90_gamma_diff_scaled=("gamma_diff_scaled", lambda x: np.nanpercentile(x, 90)),
            median_cxx_diff_scaled=("cxx_diff_scaled", "median"),
            p90_cxx_diff_scaled=("cxx_diff_scaled", lambda x: np.nanpercentile(x, 90)),
            median_rnd_diff_scaled=("rnd_diff_scaled", "median"),
            p90_rnd_diff_scaled=("rnd_diff_scaled", lambda x: np.nanpercentile(x, 90)),
            small_price_large_gamma_rate=("small_price_large_gamma", "mean"),
            small_price_large_gamma_scaled_rate=("small_price_large_gamma_scaled", "mean"),
            small_price_large_curvature_rate=("small_price_large_curvature", "mean"),
            median_price_diff_box=("price_diff_box", "median"),
            median_ranking_score=("ranking_score", "median"),
        ).reset_index()
        state_summary.to_csv(out_dir / "state_summary.csv", index=False)
    if len(diag):
        diag_summary = diag.groupby("arm").agg(
            attempts=("success", "size"), successes=("success", "sum"), success_rate=("success", "mean"),
            optimal_inaccurate_rate=("solver_status_inaccurate", "mean"),
            median_box_slack=("box_slack_sum", "median"), p90_box_slack=("box_slack_sum", lambda x: np.nanpercentile(x, 90)),
            median_mono_v=("max_mono_violation", "median"), median_conv_v=("max_conv_violation", "median"),
            bad_box_attempt_rate=("bad_box_count", lambda x: float(np.nanmean(pd.to_numeric(x, errors="coerce") > 0))),
            median_filtered_share=("filtered_quote_share", "median")
        ).reset_index()
        diag_summary.to_csv(out_dir / "repair_arm_summary.csv", index=False)

    if not args.no_plots:
        make_figures(strike_disp, book_disp, greeks, out_dir)
    print(f"[done] outputs in {out_dir}", flush=True)
    return 0



# =============================================================================
# Slice-selection controls and coverage summaries
# =============================================================================


def _finite_scores(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def audit_full_slice_feasibility(panel: pd.DataFrame, cfg: RepairConfig) -> pd.DataFrame:
    """Certify candidate slices before any score-based selection."""
    rows = []
    for key, slc in panel.groupby(slice_key_cols(), sort=False):
        record = dict(zip(slice_key_cols(), key if isinstance(key, tuple) else (key,)))
        record["n_strikes"] = int(len(slc))
        if len(slc) < cfg.min_strikes:
            record.update({
                "full_feasible": False,
                "feasibility_status": "not_run",
                "feasibility_failure_type": "too_few_strikes",
                "max_constraint_violation": np.nan,
                "max_vertical_slope": np.nan,
            })
            rows.append(record)
            continue
        try:
            arr = prepare_slice_arrays(slc, cfg)
            cert = feasibility_certificate(
                x=arr["x_norm"],
                bid=arr["bid"],
                ask=arr["ask"],
                intrinsic=arr["intrinsic"],
            )
            record.update({
                "full_feasible": bool(cert["feasible"]),
                "feasibility_status": cert.get("solver_status", ""),
                "feasibility_failure_type": cert.get("failure_type", ""),
                "max_constraint_violation": cert.get("max_constraint_violation", np.nan),
                "max_vertical_slope": cert.get("max_vertical_slope", np.nan),
                "bad_box_count": int(arr.get("bad_box_count", 0)),
                "forward_dispersion_rel": float(arr.get("forward_dispersion_rel", np.nan)),
            })
        except Exception as exc:
            record.update({
                "full_feasible": False,
                "feasibility_status": "exception",
                "feasibility_failure_type": "screen_exception",
                "feasibility_message": str(exc),
                "max_constraint_violation": np.nan,
                "max_vertical_slope": np.nan,
            })
        rows.append(record)
    return pd.DataFrame(rows)


def select_repair_slices(panel_full: pd.DataFrame, cfg: RepairConfig, panel_helper: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Select repair slices and optional t+1 marking helpers."""
    df = panel_full.copy()
    grp = df.groupby(slice_key_cols(), sort=False)
    agg_func = _cfg_get(cfg, "ranking_slice_agg", "p90").lower()

    def _clean(x):
        z = _finite_scores(pd.Series(x))
        return z.dropna()

    if agg_func == "median":
        score_agg = ("ranking_score", lambda x: float(np.nanmedian(_clean(x))) if len(_clean(x)) else np.nan)
    elif agg_func == "max":
        score_agg = ("ranking_score", lambda x: float(np.nanmax(_clean(x))) if len(_clean(x)) else np.nan)
    else:
        score_agg = ("ranking_score", lambda x: float(np.nanpercentile(_clean(x), 90)) if len(_clean(x)) else np.nan)

    slice_stats = grp.agg(
        n_strikes=("strike", "size"),
        ranking_score=score_agg,
        threshold_screen_obs=("threshold_screen_cell", "sum"),
        max_relspread_floor=("relspread_floor", "max"),
        max_h_norm=("h_norm", "max"),
        min_h_norm=("h_norm", "min"),
        med_dte=("DTE", "median"),
        med_abs_k=("k", lambda x: float(np.nanmedian(np.abs(x))))
    ).reset_index()
    feasibility = audit_full_slice_feasibility(df, cfg)
    audit_path = _cfg_get(cfg, "full_feasibility_audit_path", None)
    if audit_path:
        Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
        feasibility.to_csv(audit_path, index=False)
    slice_stats = slice_stats.merge(feasibility, on=slice_key_cols() + ["n_strikes"], how="left")
    slice_stats = slice_stats[
        (slice_stats["n_strikes"] >= cfg.min_strikes)
        & np.isfinite(slice_stats["ranking_score"])
        & slice_stats["full_feasible"].fillna(False).astype(bool)
    ].copy()
    if slice_stats.empty:
        return df.iloc[0:0].copy()

    # Rank eligible slices before selection.
    slice_stats = slice_stats.sort_values("ranking_score", ascending=False).reset_index(drop=True)
    slice_stats["slice_rank_desc"] = np.arange(1, len(slice_stats) + 1)
    slice_stats["slice_rank_asc"] = len(slice_stats) - slice_stats["slice_rank_desc"] + 1
    if len(slice_stats) > 1:
        slice_stats["slice_ranking_quantile"] = 1.0 - (slice_stats["slice_rank_desc"] - 1) / (len(slice_stats) - 1)
    else:
        slice_stats["slice_ranking_quantile"] = 1.0

    selection = str(_cfg_get(cfg, "slice_selection", "top")).lower()
    rng = np.random.default_rng(_cfg_get(cfg, "random_state", 7))
    n_take = cfg.selected_slices if cfg.selected_slices is not None else cfg.max_slices

    if n_take is not None:
        n_take = int(max(n_take, 0))
        if len(slice_stats) < n_take:
            raise RuntimeError(
                f"Full-feasibility prefilter leaves {len(slice_stats)} slices; "
                f"the fixed analysis design requires {n_take}."
            )
        if selection == "bottom":
            slice_stats = slice_stats.sort_values("ranking_score", ascending=True).head(n_take)
        elif selection == "random":
            take = rng.choice(slice_stats.index.to_numpy(), size=n_take, replace=False)
            slice_stats = slice_stats.loc[take]
        else:
            slice_stats = slice_stats.sort_values("ranking_score", ascending=False).head(n_take)
    elif selection == "bottom":
            # No cap: retain all eligible slices.
        slice_stats = slice_stats.sort_values("ranking_score", ascending=True)
    elif selection == "random":
        # No cap: retain all eligible slices.
        slice_stats = slice_stats.sample(frac=1.0, random_state=int(_cfg_get(cfg, "random_state", 7)))
    else:
        slice_stats = slice_stats.sort_values("ranking_score", ascending=False)

    slice_stats = slice_stats.copy()
    slice_stats["slice_selection"] = selection

    core_keys = set(map(tuple, slice_stats[slice_key_cols()].to_numpy()))
    mask_core = df[slice_key_cols()].apply(lambda r: tuple(r) in core_keys, axis=1)
    selected = df[mask_core].copy()
    selected["core_repair_slice"] = True

    if _cfg_get(cfg, "include_nextday_repair_slices", False):
        helper_source = panel_helper if panel_helper is not None else df
        helper_rows = []
        for _, r in slice_stats.iterrows():
            key_next = (r["currency"], int(r["date"]) + SECONDS_PER_DAY, r["expiration_date"])
            m = ((helper_source["currency"] == key_next[0]) &
                 (helper_source["date"].astype(int) == key_next[1]) &
                 (helper_source["expiration_date"].astype(int) == int(key_next[2])))
            if m.any():
                tmp = helper_source[m].copy()
                tmp["core_repair_slice"] = False
                helper_rows.append(tmp)
        if helper_rows:
            selected = pd.concat([selected] + helper_rows, ignore_index=True)
            selected = selected.drop_duplicates(["currency", "date", "expiration_date", "strike"], keep="first")
            selected["core_repair_slice"] = selected["core_repair_slice"].fillna(False).astype(bool)

    meta_cols = slice_key_cols() + [
        "ranking_score", "threshold_screen_obs", "slice_rank_desc", "slice_rank_asc",
        "slice_ranking_quantile", "slice_selection", "full_feasible"
    ]
    selected = selected.sort_values(slice_key_cols() + ["strike"]).reset_index(drop=True)
    selected = selected.merge(
        slice_stats[meta_cols].rename(columns={"ranking_score": "slice_ranking_score"}),
        on=slice_key_cols(), how="left"
    )
    selected["slice_ranking_score"] = selected["slice_ranking_score"].fillna(selected.get("ranking_score"))
    selected["slice_selection"] = selected["slice_selection"].fillna(selection)
    return selected


def summarize(panel_full: pd.DataFrame, panel_repair: pd.DataFrame, repairs: pd.DataFrame, diag: pd.DataFrame,
              greeks: pd.DataFrame, strike_disp: pd.DataFrame, book_outputs: pd.DataFrame,
              book_disp: pd.DataFrame, cfg: RepairConfig) -> pd.DataFrame:
    """Summarize repair success, support, and downstream diagnostics."""
    requested_arms = tuple(a.upper() for a in cfg.arms)
    attempted = len(diag)
    successes = int(diag["success"].sum()) if len(diag) and "success" in diag else 0
    n_slices = len(panel_repair.groupby(slice_key_cols())) if len(panel_repair) else 0
    if len(panel_repair) and "core_repair_slice" in panel_repair:
        n_core_slices = len(panel_repair[panel_repair["core_repair_slice"].astype(bool)].groupby(slice_key_cols()))
    else:
        n_core_slices = n_slices
    success_rate = successes / attempted if attempted else np.nan
    complete = np.nan
    if len(diag):
        ok = diag[diag["success"]].groupby(slice_key_cols())["arm"].nunique()
        complete = float((ok >= len(requested_arms)).mean()) if len(ok) else 0.0

    strike_main = strike_disp[strike_disp.get("core_repair_slice", pd.Series(True, index=strike_disp.index)).astype(bool)] if len(strike_disp) else pd.DataFrame()
    high_score = strike_main[strike_main.get("high_score_cell", False).astype(bool)] if len(strike_main) and "high_score_cell" in strike_main else pd.DataFrame()
    selection_values = panel_repair.get("slice_selection", pd.Series([_cfg_get(cfg, "slice_selection", "top")])).dropna().unique().tolist() if len(panel_repair) else [_cfg_get(cfg, "slice_selection", "top")]

    summary = {
        "currency": cfg.currency.upper(), "snapshot_hour": cfg.snapshot_hour,
        "slice_selection": ",".join(map(str, selection_values)),
        "ranking_slice_agg": _cfg_get(cfg, "ranking_slice_agg", "p90"),
        "ranking_score_method": _cfg_get(cfg, "ranking_score_method", "relspread_h2"),
        "selected_slices_requested": _cfg_get(cfg, "selected_slices", np.nan),
        "random_state": _cfg_get(cfg, "random_state", np.nan),
        "dte_min_repair": cfg.dte_min_repair, "dte_max_repair": cfg.dte_max_repair,
        "dte_min_marking": cfg.dte_min_marking, "dte_max_marking": cfg.dte_max_marking,
        "requested_arms": ",".join(requested_arms),
        "panel_full_rows": len(panel_full), "panel_repair_rows": len(panel_repair),
        "candidate_slices": n_slices, "core_candidate_slices": n_core_slices,
        "repair_attempts": attempted, "repair_successes": successes, "repair_success_rate": success_rate,
        "complete_slice_rate": complete, "repaired_surface_rows": len(repairs), "greek_rows": len(greeks),
        "strike_pairwise_rows": len(strike_disp), "strike_pairwise_core_rows": len(strike_main),
        "book_output_rows": len(book_outputs), "book_pairwise_rows": len(book_disp),
        "optimal_inaccurate_rate": float(np.nanmean(diag["solver_status_inaccurate"])) if len(diag) and "solver_status_inaccurate" in diag else np.nan,
        "bad_box_attempt_rate": float(np.nanmean(pd.to_numeric(diag.get("bad_box_count", pd.Series(0, index=diag.index)), errors="coerce") > 0)) if len(diag) else np.nan,
    }
    metrics = ["gamma_rr", "gamma_diff_scaled", "cxx_diff_scaled", "rnd_diff_scaled", "amp_x_h2", "price_diff_box"]
    for name, df in [("all", strike_main), ("high_score", high_score)]:
        if len(df):
            for m in metrics:
                if m in df:
                    summary[f"{name}_median_{m}"] = float(np.nanmedian(df[m]))
                    summary[f"{name}_p90_{m}"] = float(np.nanpercentile(df[m], 90))
            summary[f"{name}_small_price_large_gamma_rate"] = float(np.nanmean(df["small_price_large_gamma"])) if "small_price_large_gamma" in df else np.nan
            summary[f"{name}_small_price_large_gamma_scaled_rate"] = float(np.nanmean(df["small_price_large_gamma_scaled"])) if "small_price_large_gamma_scaled" in df else np.nan
            summary[f"{name}_small_price_large_curvature_rate"] = float(np.nanmean(df["small_price_large_curvature"])) if "small_price_large_curvature" in df else np.nan
        else:
            for m in metrics:
                summary[f"{name}_median_{m}"] = np.nan
                summary[f"{name}_p90_{m}"] = np.nan
            summary[f"{name}_small_price_large_gamma_rate"] = np.nan
            summary[f"{name}_small_price_large_gamma_scaled_rate"] = np.nan
            summary[f"{name}_small_price_large_curvature_rate"] = np.nan

    if len(book_disp):
        summary["book_p90_gamma_rr"] = float(np.nanpercentile(book_disp["gamma_rr"], 90))
        summary["book_p90_delta_rr"] = float(np.nanpercentile(book_disp["delta_rr"], 90))
        summary["book_p90_hedge_slippage_scaled"] = float(np.nanpercentile(book_disp["hedge_slippage_scaled"], 90)) if "hedge_slippage_scaled" in book_disp else np.nan
        summary["book_p90_gamma_pnl_proxy"] = float(np.nanpercentile(book_disp["gamma_pnl_proxy"], 90)) if "gamma_pnl_proxy" in book_disp else np.nan
        summary["book_p90_loss_common_diff"] = float(np.nanpercentile(book_disp["loss_common_diff"], 90))
        summary["book_p90_loss_pipeline_diff"] = float(np.nanpercentile(book_disp["loss_pipeline_diff"], 90)) if "loss_pipeline_diff" in book_disp else np.nan
        summary["book_nextday_pair_rate"] = float(np.nanmean(book_disp["nextday_available"]))
        summary["book_pipeline_nextday_pair_rate"] = float(np.nanmean(book_disp.get("pipeline_nextday_available", pd.Series(False, index=book_disp.index))))
        for thr in (0.70, 0.80, 0.90):
    # Apply next-day coverage by book.
            if "book" in book_disp.columns and "nextday_available" in book_disp.columns:
                rates = book_disp.groupby("book")["nextday_available"].mean()
                keep_books = set(rates[rates >= thr].index)
                bd_thr = book_disp[book_disp["book"].isin(keep_books)]
            else:
                bd_thr = book_disp.iloc[0:0]
            tag = int(thr * 100)
            summary[f"book_coverage{tag}_books_kept"] = int(bd_thr["book"].nunique()) if len(bd_thr) and "book" in bd_thr else 0
            summary[f"book_coverage{tag}_rows"] = int(len(bd_thr))
            summary[f"book_coverage{tag}_p90_hedge_slippage_scaled"] = float(np.nanpercentile(bd_thr["hedge_slippage_scaled"], 90)) if len(bd_thr) and "hedge_slippage_scaled" in bd_thr else np.nan
            summary[f"book_coverage{tag}_p90_gamma_pnl_proxy"] = float(np.nanpercentile(bd_thr["gamma_pnl_proxy"], 90)) if len(bd_thr) and "gamma_pnl_proxy" in bd_thr else np.nan
            summary[f"book_coverage{tag}_p90_loss_common_diff"] = float(np.nanpercentile(bd_thr["loss_common_diff"], 90)) if len(bd_thr) and "loss_common_diff" in bd_thr else np.nan
    else:
        summary["book_p90_gamma_rr"] = summary["book_p90_delta_rr"] = np.nan
        summary["book_p90_hedge_slippage_scaled"] = summary["book_p90_gamma_pnl_proxy"] = np.nan
        summary["book_p90_loss_common_diff"] = summary["book_p90_loss_pipeline_diff"] = np.nan
        summary["book_nextday_pair_rate"] = summary["book_pipeline_nextday_pair_rate"] = np.nan
        for tag in (70, 80, 90):
            summary[f"book_coverage{tag}_books_kept"] = 0
            summary[f"book_coverage{tag}_rows"] = 0
            summary[f"book_coverage{tag}_p90_hedge_slippage_scaled"] = np.nan
            summary[f"book_coverage{tag}_p90_gamma_pnl_proxy"] = np.nan
            summary[f"book_coverage{tag}_p90_loss_common_diff"] = np.nan

    min_slices = _cfg_get(cfg, "min_analysis_slices", 30)
    min_strike_pairs = _cfg_get(cfg, "min_strike_pairs", 500)
    min_book_pairs = _cfg_get(cfg, "min_book_pairs", 50)
    summary["pass_sample_size"] = bool((n_core_slices >= min_slices) and (len(strike_main) >= min_strike_pairs) and (len(book_disp) >= min_book_pairs))
    summary["pass_greek_rr"] = bool((finite(summary.get("high_score_p90_gamma_rr"), -np.inf) > cfg.gamma_rr_threshold) or (finite(summary.get("high_score_median_gamma_rr"), -np.inf) > 0.10))
    summary["pass_material_scaled"] = bool(finite(summary.get("high_score_p90_gamma_diff_scaled"), -np.inf) > cfg.gamma_material_threshold)
    summary["pass_curvature_scaled"] = bool(finite(summary.get("high_score_p90_cxx_diff_scaled"), -np.inf) > _cfg_get(cfg, "cxx_material_threshold", cfg.gamma_material_threshold))
    summary["pass_small_price_large_gamma"] = bool(finite(summary.get("high_score_small_price_large_gamma_rate"), 0.0) > 0.0)
    summary["pass_small_price_large_gamma_scaled"] = bool(finite(summary.get("high_score_small_price_large_gamma_scaled_rate"), 0.0) > 0.0)
    summary["pass_small_price_large_curvature"] = bool(finite(summary.get("high_score_small_price_large_curvature_rate"), 0.0) > 0.0)
    # Pointwise curvature ratios can be denominator-sensitive.
    summary["joint_gamma_signal"] = bool(
        summary["pass_sample_size"] and summary["pass_greek_rr"] and summary["pass_material_scaled"] and summary["pass_small_price_large_gamma"]
    )
    summary["scaled_hedge_signal"] = bool(
        summary["pass_sample_size"] and summary["pass_material_scaled"] and summary["pass_small_price_large_gamma_scaled"]
    )
    summary["curvature_signal"] = bool(
        summary["pass_sample_size"] and summary["pass_curvature_scaled"] and summary["pass_small_price_large_curvature"]
    )
    summary["primary_signal"] = summary["scaled_hedge_signal"]
    summary["analysis_signal"] = summary["primary_signal"]
    summary["any_signal"] = bool(
        summary["joint_gamma_signal"] or
        summary["scaled_hedge_signal"] or
        summary["curvature_signal"]
    )
    summary["analysis_note"] = (
        "scaled hedge criterion; curvature diagnostic"
    )
    return pd.DataFrame([summary])


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Repair-risk analysis for Deribit options")
    p.add_argument("--synthetic-call-panel", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--currency", choices=["BTC", "ETH"], default="BTC")
    p.add_argument("--snapshot-hour", type=int, default=12)
    p.add_argument("--month", nargs="*", default=[])
    p.add_argument("--dte-min", type=float, default=None, help="Minimum DTE for repair slices")
    p.add_argument("--dte-max", type=float, default=None, help="Maximum DTE for repair slices")
    p.add_argument("--dte-min-repair", type=float, default=2.0)
    p.add_argument("--dte-max-repair", type=float, default=30.0)
    p.add_argument("--dte-min-marking", type=float, default=0.0)
    p.add_argument("--dte-max-marking", type=float, default=31.0)
    p.add_argument("--marking-extra-days", type=int, default=1)
    p.add_argument("--include-nextday-repair-slices", action="store_true")
    p.add_argument("--price-floor", type=float, default=0.0005)
    p.add_argument("--strike-tol", type=float, default=0.02)
    p.add_argument("--min-strikes", type=int, default=7)
    p.add_argument("--arms", nargs="+", default=["L1", "L2", "SMOOTH"])
    p.add_argument("--add-arms", nargs="*", default=[])
    p.add_argument("--include-broader-arms", action="store_true")
    p.add_argument("--smooth-rho", type=float, default=0.01)
    p.add_argument("--smooth-rhos", nargs="*", type=float, default=[])
    p.add_argument("--budget-smoothing", nargs="*", type=float, default=[])
    p.add_argument("--spline-smoothing", type=float, default=1e-4)
    p.add_argument("--filter-spread-quantile", type=float, default=0.75)
    p.add_argument("--filter-bad-weight", type=float, default=0.01)
    p.add_argument("--svi-max-nfev", type=int, default=2000)
    p.add_argument("--elastic-penalty", type=float, default=1000.0)
    p.add_argument("--elastic-box", action="store_true")
    p.add_argument("--qp-backend", choices=["cvxpy", "scipy"], default="cvxpy")
    p.add_argument("--qp-solver", default="auto")
    p.add_argument("--solver-maxiter", type=int, default=800)
    p.add_argument("--exclude-optimal-inaccurate", action="store_true")
    p.add_argument("--ranking-score-method", choices=["relspread_h2", "abs_spread_h2"], default="relspread_h2")
    p.add_argument("--ranking-slice-agg", choices=["max", "p90", "median"], default="p90")
    p.add_argument("--selected-slices", type=int, default=None)
    p.add_argument("--slice-selection", choices=["top", "bottom", "random"], default="top",
                   help="Select top/bottom/random eligible slices by slice-level ranking score. Bottom and random selections use the same eligible pool.")
    p.add_argument("--max-slices", type=int, default=None)
    p.add_argument("--random-state", "--random-seed", type=int, default=7)
    p.add_argument("--hedge-tc-bps", type=float, default=0.0)
    p.add_argument("--allow-partial-arms", action="store_true")
    p.add_argument("--min-analysis-slices", type=int, default=30)
    p.add_argument("--min-strike-pairs", type=int, default=500)
    p.add_argument("--min-book-pairs", type=int, default=50)
    p.add_argument("--cxx-material-threshold", type=float, default=0.05)
    p.add_argument("--gamma-scaled-only-threshold", type=float, default=0.05)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dte_min_repair = args.dte_min if args.dte_min is not None else args.dte_min_repair
    dte_max_repair = args.dte_max if args.dte_max is not None else args.dte_max_repair
    effective_arms = expand_arms_from_args(args)
    cfg = RepairConfig(
        currency=args.currency.upper(), snapshot_hour=args.snapshot_hour,
        dte_min_repair=dte_min_repair, dte_max_repair=dte_max_repair,
        dte_min_marking=args.dte_min_marking, dte_max_marking=args.dte_max_marking,
        marking_extra_days=args.marking_extra_days,
        price_floor=args.price_floor, strike_tol=args.strike_tol, min_strikes=args.min_strikes,
        arms=effective_arms, smooth_rho=args.smooth_rho,
        smooth_rhos=tuple(args.smooth_rhos), budget_smoothing=tuple(args.budget_smoothing),
        spline_smoothing=args.spline_smoothing, filter_spread_quantile=args.filter_spread_quantile,
        filter_bad_weight=args.filter_bad_weight, svi_max_nfev=args.svi_max_nfev,
        elastic_penalty=args.elastic_penalty, elastic_box=args.elastic_box,
        qp_backend=args.qp_backend, qp_solver=args.qp_solver, solver_maxiter=args.solver_maxiter,
        ranking_score_method=args.ranking_score_method, ranking_slice_agg=args.ranking_slice_agg,
        selected_slices=args.selected_slices, max_slices=args.max_slices,
        random_state=args.random_state, hedge_tc_bps=args.hedge_tc_bps,
        require_all_arms_for_main=not args.allow_partial_arms,
    )
    cfg.include_nextday_repair_slices = args.include_nextday_repair_slices
    cfg.exclude_optimal_inaccurate = args.exclude_optimal_inaccurate
    cfg.min_analysis_slices = args.min_analysis_slices
    cfg.min_strike_pairs = args.min_strike_pairs
    cfg.min_book_pairs = args.min_book_pairs
    cfg.cxx_material_threshold = args.cxx_material_threshold
    cfg.gamma_scaled_only_threshold = args.gamma_scaled_only_threshold
    cfg.slice_selection = args.slice_selection

    out_dir = ensure_dir(args.output_dir)
    cfg.full_feasibility_audit_path = str(out_dir / "slice_feasibility_audit.csv")
    cfg.constraint_system = "full_normalized_call_strike_v1"
    cfg_json = asdict(cfg)
    for extra in ["include_nextday_repair_slices", "exclude_optimal_inaccurate", "min_analysis_slices", "min_strike_pairs", "min_book_pairs", "cxx_material_threshold", "gamma_scaled_only_threshold", "slice_selection", "full_feasibility_audit_path", "constraint_system"]:
        cfg_json[extra] = getattr(cfg, extra)
    (out_dir / "repair_config.json").write_text(json.dumps(cfg_json, indent=2), encoding="utf-8")
    print(f"[load] {args.synthetic_call_panel}", flush=True)
    print(f"[arms] {', '.join(cfg.arms)}", flush=True)
    print(f"[selection] {cfg.slice_selection}, N={cfg.selected_slices}, agg={cfg.ranking_slice_agg}", flush=True)

    panel_repair_raw = load_synthetic_panel(args.synthetic_call_panel, cfg, args.month,
                                            dte_min=cfg.dte_min_repair, dte_max=cfg.dte_max_repair,
                                            extra_days=0)
    panel_marking_raw = load_synthetic_panel(args.synthetic_call_panel, cfg, args.month,
                                             dte_min=cfg.dte_min_marking, dte_max=cfg.dte_max_marking,
                                             extra_days=cfg.marking_extra_days)
    print(f"[panel repair raw] rows={len(panel_repair_raw):,}", flush=True)
    print(f"[panel marking raw] rows={len(panel_marking_raw):,}", flush=True)
    if panel_repair_raw.empty:
        print("No repair-panel rows after filtering.", file=sys.stderr)
        return 2

    panel_repair_state = add_state_flags_base(panel_repair_raw, cfg)
    panel_marking = add_state_flags_base(panel_marking_raw, cfg) if len(panel_marking_raw) else panel_marking_raw
    panel_repair = select_repair_slices(panel_repair_state, cfg, panel_helper=panel_marking)
    panel_repair.to_parquet(out_dir / "repair_panel.parquet", index=False)
    panel_marking.to_parquet(out_dir / "marking_panel.parquet", index=False)
    print(f"[panel repair selected] rows={len(panel_repair):,}, slices={len(panel_repair.groupby(slice_key_cols())) if len(panel_repair) else 0}", flush=True)
    if panel_repair.empty:
        print("No selected repair slices.", file=sys.stderr)
        return 2

    surfaces, diag = run_repairs(panel_repair, cfg)
    surfaces.to_parquet(out_dir / "repaired_surfaces.parquet", index=False)
    diag.to_parquet(out_dir / "repair_diagnostics.parquet", index=False)
    diag.to_csv(out_dir / "repair_diagnostics.csv", index=False)
    print(f"[repairs] surfaces={len(surfaces):,}, diagnostics={len(diag):,}", flush=True)
    if surfaces.empty:
        print("No repaired surfaces. Check solver diagnostics.", file=sys.stderr)
        return 3

    if cfg.require_all_arms_for_main:
        core_diag = diag.copy()
        if "core_repair_slice" in core_diag:
            core_diag = core_diag[core_diag["core_repair_slice"].astype(bool)]
        ok_counts = core_diag[core_diag["success"]].groupby(slice_key_cols())["arm"].nunique().reset_index(name="n_success_arms") if len(core_diag) else pd.DataFrame()
        ok_keys = set(map(tuple, ok_counts[ok_counts["n_success_arms"] >= len(cfg.arms)][slice_key_cols()].to_numpy())) if len(ok_counts) else set()
        before = len(surfaces)
        surfaces = surfaces[surfaces[slice_key_cols()].apply(lambda r: (not bool(r.get("core_repair_slice", True))) or tuple(r[slice_key_cols()]) in ok_keys, axis=1)].copy()
        print(f"[filter] complete-core-arm surfaces {before:,} -> {len(surfaces):,}", flush=True)
        surfaces.to_parquet(out_dir / "repaired_surfaces_complete_arms.parquet", index=False)
        if surfaces.empty:
            print("No complete-arm core slices after filtering; use --allow-partial-arms to retain partial support.", file=sys.stderr)
            return 3

    greeks = compute_greeks(surfaces)
    greeks.to_parquet(out_dir / "greeks.parquet", index=False)
    print(f"[greeks] rows={len(greeks):,}", flush=True)

    strike_disp = compute_strike_pairwise_dispersion(greeks, cfg)
    strike_disp.to_parquet(out_dir / "strike_pairwise_dispersion.parquet", index=False)
    strike_disp.to_csv(out_dir / "strike_pairwise_dispersion.csv", index=False)
    print(f"[strike dispersion] rows={len(strike_disp):,}", flush=True)

    book_outputs = compute_book_outputs(greeks, panel_marking, cfg)
    book_outputs.to_parquet(out_dir / "book_outputs.parquet", index=False)
    print(f"[books] rows={len(book_outputs):,}", flush=True)

    book_disp = compute_book_pairwise_dispersion(book_outputs)
    book_disp.to_parquet(out_dir / "book_pairwise_dispersion.parquet", index=False)
    book_disp.to_csv(out_dir / "book_pairwise_dispersion.csv", index=False)
    print(f"[book dispersion] rows={len(book_disp):,}", flush=True)

    arm_pair_summary = make_arm_pair_summary(strike_disp, book_disp)
    arm_pair_summary.to_csv(out_dir / "arm_pair_summary.csv", index=False)
    arm_pair_support = make_arm_pair_common_support(diag)
    arm_pair_support.to_csv(out_dir / "arm_pair_common_support.csv", index=False)
    protocol_summary = make_protocol_comparison_summary(strike_disp)
    protocol_summary.to_csv(out_dir / "protocol_comparison_summary.csv", index=False)

    summary = summarize(panel_marking, panel_repair, surfaces, diag, greeks, strike_disp, book_outputs, book_disp, cfg)
    summary.to_csv(out_dir / "repair_summary.csv", index=False)
    print("[summary]")
    print(summary.T.to_string(header=False))

    if len(strike_disp):
        main_strike = strike_disp[strike_disp.get("core_repair_slice", pd.Series(True, index=strike_disp.index)).astype(bool)]
        state_summary = main_strike.groupby(["high_score_cell", "threshold_screen_cell", "dte_bucket"], dropna=False).agg(
            n=("gamma_rr", "size"),
            median_gamma_rr=("gamma_rr", "median"),
            p90_gamma_rr=("gamma_rr", lambda x: np.nanpercentile(x, 90)),
            median_gamma_diff_scaled=("gamma_diff_scaled", "median"),
            p90_gamma_diff_scaled=("gamma_diff_scaled", lambda x: np.nanpercentile(x, 90)),
            median_cxx_diff_scaled=("cxx_diff_scaled", "median"),
            p90_cxx_diff_scaled=("cxx_diff_scaled", lambda x: np.nanpercentile(x, 90)),
            median_rnd_diff_scaled=("rnd_diff_scaled", "median"),
            p90_rnd_diff_scaled=("rnd_diff_scaled", lambda x: np.nanpercentile(x, 90)),
            small_price_large_gamma_rate=("small_price_large_gamma", "mean"),
            small_price_large_gamma_scaled_rate=("small_price_large_gamma_scaled", "mean"),
            small_price_large_curvature_rate=("small_price_large_curvature", "mean"),
            median_price_diff_box=("price_diff_box", "median"),
            median_ranking_score=("ranking_score", "median"),
        ).reset_index()
        state_summary.to_csv(out_dir / "state_summary.csv", index=False)
    if len(diag):
        diag_summary = diag.groupby("arm").agg(
            attempts=("success", "size"), successes=("success", "sum"), success_rate=("success", "mean"),
            optimal_inaccurate_rate=("solver_status_inaccurate", "mean"),
            median_box_slack=("box_slack_sum", "median"), p90_box_slack=("box_slack_sum", lambda x: np.nanpercentile(x, 90)),
            median_mono_v=("max_mono_violation", "median"), median_conv_v=("max_conv_violation", "median"),
            bad_box_attempt_rate=("bad_box_count", lambda x: float(np.nanmean(pd.to_numeric(x, errors="coerce") > 0))),
            median_filtered_share=("filtered_quote_share", "median")
        ).reset_index()
        diag_summary.to_csv(out_dir / "repair_arm_summary.csv", index=False)

    if not args.no_plots:
        make_figures(strike_disp, book_disp, greeks, out_dir)
    print(f"[done] outputs in {out_dir}", flush=True)
    return 0


# =============================================================================
# Book-level hedge diagnostics
# =============================================================================


def compute_book_pairwise_dispersion(book_outputs: pd.DataFrame) -> pd.DataFrame:
    """Compute book-level hedge and threshold diagnostics."""
    if book_outputs.empty:
        return pd.DataFrame()
    rows = []
    trigger_thresholds = (0.5, 1.0, 2.0)       # Hedge notional / book scale.
    diff_thresholds = (0.05, 0.10, 0.25)       # Hedge difference / book scale.
    slippage_thresholds = (0.01, 0.05, 0.10)   # Next-step slippage / book scale.
    gamma_pnl_thresholds = (0.01, 0.05, 0.10)  # Gamma proxy / book scale.

    for _, g in book_outputs.groupby(["currency", "date", "expiration_date", "book"], sort=False):
        if g["arm"].nunique() < 2:
            continue
        rec = {r["arm"]: r for _, r in g.iterrows()}
        arms = sorted(rec)
        gamma_scale = max(np.nanmedian(np.abs(g["gamma_usd"].to_numpy(dtype=float))), EPS)
        delta_scale = max(np.nanmedian(np.abs(g["delta_usd"].to_numpy(dtype=float))), EPS)
        for a, b in combinations(arms, 2):
            ra, rb = rec[a], rec[b]
            gamma_diff = abs(float(ra["gamma_usd"]) - float(rb["gamma_usd"]))
            delta_diff = abs(float(ra["delta_usd"]) - float(rb["delta_usd"]))
            hedge_a = float(ra["hedge_underlying_units"])
            hedge_b = float(rb["hedge_underlying_units"])
            hedge_diff = abs(hedge_a - hedge_b)
            loss_common_diff = np.nan
            if np.isfinite(ra.get("loss_common")) and np.isfinite(rb.get("loss_common")):
                loss_common_diff = abs(float(ra["loss_common"]) - float(rb["loss_common"]))
            loss_pipeline_diff = np.nan
            if np.isfinite(ra.get("loss_pipeline")) and np.isfinite(rb.get("loss_pipeline")):
                loss_pipeline_diff = abs(float(ra["loss_pipeline"]) - float(rb["loss_pipeline"]))

            x_t = finite(ra.get("underlyer_spot"), np.nan)
            x_next = finite(ra.get("underlyer_spot_next"), np.nan)
            dS_abs = abs(x_next - x_t) if np.isfinite(x_t) and np.isfinite(x_next) else np.nan
            value_scale_candidates = [
                abs(finite(ra.get("value_common_usd"), np.nan)),
                abs(finite(rb.get("value_common_usd"), np.nan)),
                abs(finite(ra.get("value_repaired_usd"), np.nan)),
                abs(finite(rb.get("value_repaired_usd"), np.nan)),
                1.0,
            ]
            pnl_scale = max([v for v in value_scale_candidates if np.isfinite(v)] or [1.0])
            hedge_slippage_scaled = hedge_diff * dS_abs / (pnl_scale + EPS) if np.isfinite(dS_abs) else np.nan
            gamma_pnl_proxy = 0.5 * gamma_diff * (dS_abs ** 2) / (pnl_scale + EPS) if np.isfinite(dS_abs) else np.nan
            hedge_notional_scaled_a = abs(hedge_a) * x_t / (pnl_scale + EPS) if np.isfinite(x_t) else np.nan
            hedge_notional_scaled_b = abs(hedge_b) * x_t / (pnl_scale + EPS) if np.isfinite(x_t) else np.nan
            hedge_diff_notional_scaled = hedge_diff * x_t / (pnl_scale + EPS) if np.isfinite(x_t) else np.nan

            row = {
                "currency": ra["currency"], "date": int(ra["date"]), "expiration_date": int(ra["expiration_date"]),
                "book": ra["book"], "arm_a": a, "arm_b": b, "DTE": float(ra["DTE"]),
                "delta_diff": delta_diff,
                "delta_rr": delta_diff / (delta_scale + EPS),
                "gamma_diff": gamma_diff,
                "gamma_rr": gamma_diff / (gamma_scale + EPS),
                "hedge_diff": hedge_diff,
                "underlyer_move_abs": dS_abs,
                "pnl_scale_usd": pnl_scale,
                "hedge_slippage_scaled": hedge_slippage_scaled,
                "gamma_pnl_proxy": gamma_pnl_proxy,
                "hedge_notional_scaled_a": hedge_notional_scaled_a,
                "hedge_notional_scaled_b": hedge_notional_scaled_b,
                "hedge_diff_notional_scaled": hedge_diff_notional_scaled,
                "loss_common_diff": loss_common_diff,
                "loss_pipeline_diff": loss_pipeline_diff,
                "nextday_available": bool(ra["nextday_available"] and rb["nextday_available"]),
                "pipeline_nextday_available": bool(ra.get("pipeline_nextday_available", False) and rb.get("pipeline_nextday_available", False)),
            }
            for thr in trigger_thresholds:
                tag = label_float(thr)
                da = np.isfinite(hedge_notional_scaled_a) and hedge_notional_scaled_a > thr
                db = np.isfinite(hedge_notional_scaled_b) and hedge_notional_scaled_b > thr
                row[f"hedge_trigger_flip_{tag}"] = bool(da != db)
            row["hedge_trigger_flip_any"] = bool(any(row.get(f"hedge_trigger_flip_{label_float(th)}", False) for th in trigger_thresholds))
            for thr in diff_thresholds:
                tag = label_float(thr)
                row[f"hedge_diff_notional_gt_{tag}"] = bool(np.isfinite(hedge_diff_notional_scaled) and hedge_diff_notional_scaled > thr)
            for thr in slippage_thresholds:
                tag = label_float(thr)
                row[f"hedge_slippage_gt_{tag}"] = bool(np.isfinite(hedge_slippage_scaled) and hedge_slippage_scaled > thr)
            for thr in gamma_pnl_thresholds:
                tag = label_float(thr)
                row[f"gamma_pnl_proxy_gt_{tag}"] = bool(np.isfinite(gamma_pnl_proxy) and gamma_pnl_proxy > thr)
            rows.append(row)
    return pd.DataFrame(rows)


def make_arm_pair_summary(strike_disp: pd.DataFrame, book_disp: pd.DataFrame) -> pd.DataFrame:
    """Arm-pair attribution for continuous and threshold-based outcomes."""
    rows = []
    if len(strike_disp):
        for (a, b), g in strike_disp.groupby(["arm_a", "arm_b"], sort=False):
            rows.append({
                "level": "strike", "arm_a": a, "arm_b": b, "book": "",
                "n": len(g), "n_core": int(g.get("core_repair_slice", pd.Series(True, index=g.index)).astype(bool).sum()),
                "p50_gamma_rr": float(np.nanmedian(g["gamma_rr"])),
                "p90_gamma_rr": float(np.nanpercentile(g["gamma_rr"], 90)),
                "p50_gamma_diff_scaled": float(np.nanmedian(g["gamma_diff_scaled"])),
                "p90_gamma_diff_scaled": float(np.nanpercentile(g["gamma_diff_scaled"], 90)),
                "p50_cxx_diff_scaled": float(np.nanmedian(g["cxx_diff_scaled"])),
                "p90_cxx_diff_scaled": float(np.nanpercentile(g["cxx_diff_scaled"], 90)),
                "p50_rnd_diff_scaled": float(np.nanmedian(g["rnd_diff_scaled"])),
                "p90_rnd_diff_scaled": float(np.nanpercentile(g["rnd_diff_scaled"], 90)),
                "small_price_large_gamma_rate": float(np.nanmean(g["small_price_large_gamma"])),
                "small_price_large_gamma_scaled_rate": float(np.nanmean(g["small_price_large_gamma_scaled"])) if "small_price_large_gamma_scaled" in g else np.nan,
                "small_price_large_curvature_rate": float(np.nanmean(g["small_price_large_curvature"])),
            })
    if len(book_disp):
        for (a, b, book), g in book_disp.groupby(["arm_a", "arm_b", "book"], sort=False):
            row = {
                "level": "book", "arm_a": a, "arm_b": b, "book": book,
                "n": len(g), "n_core": len(g),
                "p50_gamma_rr": float(np.nanmedian(g["gamma_rr"])),
                "p90_gamma_rr": float(np.nanpercentile(g["gamma_rr"], 90)),
                "p50_delta_rr": float(np.nanmedian(g["delta_rr"])),
                "p90_delta_rr": float(np.nanpercentile(g["delta_rr"], 90)),
                "p90_hedge_slippage_scaled": float(np.nanpercentile(g["hedge_slippage_scaled"], 90)) if "hedge_slippage_scaled" in g else np.nan,
                "p90_gamma_pnl_proxy": float(np.nanpercentile(g["gamma_pnl_proxy"], 90)) if "gamma_pnl_proxy" in g else np.nan,
                "p90_loss_common_diff": float(np.nanpercentile(g["loss_common_diff"], 90)),
                "p90_loss_pipeline_diff": float(np.nanpercentile(g["loss_pipeline_diff"], 90)) if "loss_pipeline_diff" in g else np.nan,
                "common_nextday_rate": float(np.nanmean(g["nextday_available"])),
                "pipeline_nextday_rate": float(np.nanmean(g.get("pipeline_nextday_available", pd.Series(False, index=g.index)))),
            }
            for col in [c for c in g.columns if c.startswith("hedge_trigger_flip_") or c.startswith("hedge_diff_notional_gt_") or c.startswith("hedge_slippage_gt_") or c.startswith("gamma_pnl_proxy_gt_")]:
                row[f"{col}_rate"] = float(np.nanmean(g[col].astype(float)))
            rows.append(row)
    return pd.DataFrame(rows)





# =============================================================================
# VaR-style diagnostics
# =============================================================================
# Matched common-scenario and repair-specific VaR/CVaR summaries.

_parse_args_without_var = parse_args
_main_without_var = main


def _parse_var_args(argv: Optional[Sequence[str]] = None):
    raw = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    vg = ap.add_mutually_exclusive_group()
    vg.add_argument("--var-style", dest="var_style", action="store_true", help="Enable VaR-style diagnostics.")
    vg.add_argument("--no-var-style", dest="no_var_style", action="store_true", help="Disable VaR-style/VaR-style outputs.")
    ap.add_argument("--var-alphas", nargs="*", type=float, default=None, help="VaR quantile levels, e.g. 0.95 0.99.")
    ap.add_argument("--var-min-losses", type=int, default=None, help="Minimum finite losses per book/arm required for VaR-style.")
    ap.add_argument("--var-absolute-limits", nargs="*", type=float, default=None, help="Fixed normalized VaR/capital limits, e.g. 0.05 0.10 0.25.")
    ap.add_argument("--var-capital-quantiles", nargs="*", type=float, default=None, help="Pooled arm-VaR quantile capital limits, e.g. 0.50 0.75 0.90.")
    ap.add_argument("--var-baseline-arm", default=None, help="Baseline arm for baseline-multiple capital limits, default L2.")
    ap.add_argument("--var-baseline-multipliers", nargs="*", type=float, default=None, help="Baseline VaR multipliers, e.g. 0.90 1.00 1.10.")
    ns, rest = ap.parse_known_args(raw)
    var_style = True
    if ns.no_var_style:
        var_style = False
    elif ns.var_style:
        var_style = True
    extras = {
        "var_style": var_style,
        "var_alphas": tuple(ns.var_alphas if ns.var_alphas else [0.95, 0.99]),
        "var_min_losses": int(ns.var_min_losses if ns.var_min_losses is not None else 10),
        "var_absolute_limits": tuple(ns.var_absolute_limits if ns.var_absolute_limits else [0.05, 0.10, 0.25]),
        "var_capital_quantiles": tuple(ns.var_capital_quantiles if ns.var_capital_quantiles else [0.50, 0.75, 0.90]),
        "var_baseline_arm": str(ns.var_baseline_arm if ns.var_baseline_arm is not None else "L2").upper(),
        "var_baseline_multipliers": tuple(ns.var_baseline_multipliers if ns.var_baseline_multipliers else [0.90, 1.00, 1.10]),
    }
    return rest, extras


def parse_args(argv: Optional[Sequence[str]] = None):
    rest, extras = _parse_var_args(argv)
    args = _parse_args_without_var(rest)
    for k, v in extras.items():
        setattr(args, k, v)
    return args


def _alpha_tag(alpha: float) -> str:
    return "a" + label_float(alpha)


def _limit_tag(x: float, prefix: str = "") -> str:
    return prefix + label_float(x)


def _quantile(x, prob: float) -> float:
    xs = _finite_series(x)
    return float(np.nanpercentile(xs.to_numpy(float), 100 * prob)) if len(xs) else np.nan


def _finite_series(x) -> pd.Series:
    return pd.to_numeric(pd.Series(x), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _empirical_var_cvar(x, alpha: float) -> tuple[float, float]:
    xs = _finite_series(x)
    if len(xs) == 0:
        return np.nan, np.nan
    var = float(np.nanquantile(xs.to_numpy(float), alpha))
    tail = xs[xs >= var]
    cvar = float(tail.mean()) if len(tail) else np.nan
    return var, cvar


def compute_var_style_outputs(book_outputs: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute matched VaR/CVaR summaries under both loss specifications."""
    if book_outputs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    alphas = tuple(float(a) for a in getattr(args, "var_alphas", (0.95, 0.99)))
    min_losses = int(getattr(args, "var_min_losses", 10))
    abs_limits = tuple(float(x) for x in getattr(args, "var_absolute_limits", (0.05, 0.10, 0.25)))
    cap_qs = tuple(float(q) for q in getattr(args, "var_capital_quantiles", (0.50, 0.75, 0.90)))
    baseline_arm = str(getattr(args, "var_baseline_arm", "L2")).upper()
    baseline_mults = tuple(float(m) for m in getattr(args, "var_baseline_multipliers", (0.90, 1.00, 1.10)))

    loss_specs = [("common", "loss_common", "nextday_available")]
    if "loss_pipeline" in book_outputs.columns:
        loss_specs.append(("pipeline", "loss_pipeline", "pipeline_nextday_available"))

    arm_rows = []
    clean_by_loss = {}
    for loss_type, loss_col, avail_col in loss_specs:
        if loss_col not in book_outputs.columns:
            continue
        df = book_outputs.copy()
        if avail_col in df.columns:
            df = df[df[avail_col].astype(bool)].copy()
        df[loss_col] = pd.to_numeric(df[loss_col], errors="coerce")
        df = df[np.isfinite(df[loss_col])].copy()
        clean_by_loss[loss_type] = (df, loss_col)
        if df.empty:
            continue
        for (currency, book, arm), g in df.groupby(["currency", "book", "arm"], sort=False):
            losses = _finite_series(g[loss_col])
            row = {
                "currency": currency, "book": book, "arm": arm, "loss_type": loss_type,
                "loss_col": loss_col, "n_loss": int(len(losses)),
                "n_dates": int(g["date"].nunique()) if "date" in g else int(len(losses)),
                "mean_loss": float(losses.mean()) if len(losses) else np.nan,
                "median_loss": float(losses.median()) if len(losses) else np.nan,
                "p90_loss": float(np.nanquantile(losses, 0.90)) if len(losses) else np.nan,
                "max_loss": float(losses.max()) if len(losses) else np.nan,
                "var_style_valid": bool(len(losses) >= min_losses),
                "var_min_losses": min_losses,
            }
            for alpha in alphas:
                tag = _alpha_tag(alpha)
                vv, cc = _empirical_var_cvar(losses, alpha) if len(losses) >= min_losses else (np.nan, np.nan)
                row[f"var_{tag}"] = vv
                row[f"cvar_{tag}"] = cc
            arm_rows.append(row)
    arm_summary = pd.DataFrame(arm_rows)

    pair_rows = []
    if not arm_summary.empty:
        for (currency, book, loss_type), g in arm_summary.groupby(["currency", "book", "loss_type"], sort=False):
            gvalid = g[g["var_style_valid"].astype(bool)].copy()
            if gvalid["arm"].nunique() < 2:
                continue
            rec = {str(r["arm"]): r for _, r in gvalid.iterrows()}
            arms = sorted(rec)
            for a, b in combinations(arms, 2):
                ra, rb = rec[a], rec[b]
                row = {"currency": currency, "book": book, "loss_type": loss_type, "arm_a": a, "arm_b": b,
                       "n_loss_a": int(ra.get("n_loss", 0)), "n_loss_b": int(rb.get("n_loss", 0))}
                for alpha in alphas:
                    tag = _alpha_tag(alpha)
                    va = finite(ra.get(f"var_{tag}"), np.nan); vb = finite(rb.get(f"var_{tag}"), np.nan)
                    ca = finite(ra.get(f"cvar_{tag}"), np.nan); cb = finite(rb.get(f"cvar_{tag}"), np.nan)
                    vals = _finite_series(gvalid[f"var_{tag}"]) if f"var_{tag}" in gvalid else pd.Series(dtype=float)
                    row[f"var_{tag}_a"] = va; row[f"var_{tag}_b"] = vb
                    row[f"var_{tag}_diff"] = abs(va - vb) if np.isfinite(va) and np.isfinite(vb) else np.nan
                    row[f"var_{tag}_diff_scaled"] = abs(va - vb) / (max(abs(va), abs(vb), EPS)) if np.isfinite(va) and np.isfinite(vb) else np.nan
                    row[f"cvar_{tag}_diff"] = abs(ca - cb) if np.isfinite(ca) and np.isfinite(cb) else np.nan
                    row[f"cvar_{tag}_diff_scaled"] = abs(ca - cb) / (max(abs(ca), abs(cb), EPS)) if np.isfinite(ca) and np.isfinite(cb) else np.nan
                    for lim in abs_limits:
                        ltag = _limit_tag(lim, "abs_")
                        da = np.isfinite(va) and va > lim; db = np.isfinite(vb) and vb > lim
                        row[f"capital_flip_{tag}_{ltag}"] = bool(da != db)
                        row[f"capital_limit_{tag}_{ltag}"] = lim
                    for qv in cap_qs:
                        qtag = _limit_tag(qv, "q")
                        lim = float(np.nanquantile(vals, qv)) if len(vals) else np.nan
                        da = np.isfinite(va) and np.isfinite(lim) and va > lim
                        db = np.isfinite(vb) and np.isfinite(lim) and vb > lim
                        row[f"capital_flip_{tag}_{qtag}"] = bool(da != db)
                        row[f"capital_limit_{tag}_{qtag}"] = lim
                    if baseline_arm in rec:
                        base_v = finite(rec[baseline_arm].get(f"var_{tag}"), np.nan)
                        for mult in baseline_mults:
                            mtag = _limit_tag(mult, f"baseline_{baseline_arm}_x")
                            lim = base_v * mult if np.isfinite(base_v) else np.nan
                            da = np.isfinite(va) and np.isfinite(lim) and va > lim
                            db = np.isfinite(vb) and np.isfinite(lim) and vb > lim
                            row[f"capital_flip_{tag}_{mtag}"] = bool(da != db)
                            row[f"capital_limit_{tag}_{mtag}"] = lim
                pair_rows.append(row)
    pairwise = pd.DataFrame(pair_rows)

    breach_rows = []
    if not arm_summary.empty:
        for loss_type, (df, loss_col) in clean_by_loss.items():
            vars_long = arm_summary[(arm_summary["loss_type"] == loss_type) & (arm_summary["var_style_valid"].astype(bool))]
            if df.empty or vars_long.empty:
                continue
            var_cols = ["currency", "book", "arm"] + [f"var_{_alpha_tag(a)}" for a in alphas]
            merged = df.merge(vars_long[var_cols], on=["currency", "book", "arm"], how="inner")
            for (currency, date, expiration, book), g in merged.groupby(["currency", "date", "expiration_date", "book"], sort=False):
                if g["arm"].nunique() < 2:
                    continue
                rec = {str(r["arm"]): r for _, r in g.iterrows()}
                for a, b in combinations(sorted(rec), 2):
                    ra, rb = rec[a], rec[b]
                    row = {"currency": currency, "date": int(date), "expiration_date": int(expiration), "book": book,
                           "loss_type": loss_type, "arm_a": a, "arm_b": b}
                    la = finite(ra.get(loss_col), np.nan); lb = finite(rb.get(loss_col), np.nan)
                    for alpha in alphas:
                        tag = _alpha_tag(alpha)
                        va = finite(ra.get(f"var_{tag}"), np.nan); vb = finite(rb.get(f"var_{tag}"), np.nan)
                        ba = np.isfinite(la) and np.isfinite(va) and la > va
                        bb = np.isfinite(lb) and np.isfinite(vb) and lb > vb
                        row[f"breach_a_{tag}"] = bool(ba)
                        row[f"breach_b_{tag}"] = bool(bb)
                        row[f"breach_flip_{tag}"] = bool(ba != bb)
                    breach_rows.append(row)
    breach_pairwise = pd.DataFrame(breach_rows)

    sum_rows = []
    row = {"var_style_arm_rows": len(arm_summary), "var_style_pairwise_rows": len(pairwise), "var_style_breach_rows": len(breach_pairwise)}
    if not pairwise.empty:
        for alpha in alphas:
            tag = _alpha_tag(alpha)
            for loss_type in sorted(pairwise["loss_type"].dropna().unique()):
                gg = pairwise[pairwise["loss_type"] == loss_type]
                if f"var_{tag}_diff_scaled" in gg:
                    row[f"var_style_{loss_type}_{tag}_p90_var_diff_scaled"] = _quantile(gg[f"var_{tag}_diff_scaled"], 0.9)
                flip_cols = [c for c in gg.columns if c.startswith(f"capital_flip_{tag}_")]
                if flip_cols:
                    row[f"var_style_{loss_type}_{tag}_any_capital_flip_rate"] = float(gg[flip_cols].astype(bool).any(axis=1).mean())
    if not breach_pairwise.empty:
        for alpha in alphas:
            tag = _alpha_tag(alpha)
            col = f"breach_flip_{tag}"
            if col in breach_pairwise:
                row[f"var_style_breach_flip_{tag}_rate"] = float(breach_pairwise[col].astype(bool).mean())
    if len(row) > 3:
        sum_rows.append(row)
    run_summary = pd.DataFrame(sum_rows)
    return arm_summary, pairwise, breach_pairwise, run_summary


def _augment_repair_summary_with_var(out_dir: Path, var_run_summary: pd.DataFrame) -> None:
    f = out_dir / "repair_summary.csv"
    if not f.exists():
        return
    try:
        repair_summary = pd.read_csv(f)
    except Exception:
        return
    if not var_run_summary.empty:
        for c in var_run_summary.columns:
            repair_summary[c] = var_run_summary.iloc[0][c]
    repair_summary["var_style_enabled"] = True
    repair_summary["var_style_note"] = "within-run VaR/CVaR sensitivity from matched book losses"
    repair_summary.to_csv(f, index=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ret = _main_without_var(argv)
    if ret != 0:
        return ret
    if not getattr(args, "var_style", True):
        return ret
    out_dir = Path(args.output_dir)
    try:
        book_outputs = pd.read_parquet(out_dir / "book_outputs.parquet")
    except Exception:
        try:
            book_outputs = pd.read_csv(out_dir / "book_outputs.csv")
        except Exception as e:
            print(f"[var-style] could not load book_outputs: {e}", flush=True)
            return ret
    arm, pair, breach, summary = compute_var_style_outputs(book_outputs, args)
    if not arm.empty:
        arm.to_csv(out_dir / "var_style_arm_summary.csv", index=False)
        try: arm.to_parquet(out_dir / "var_style_arm_summary.parquet", index=False)
        except Exception: pass
    if not pair.empty:
        pair.to_csv(out_dir / "var_style_pairwise.csv", index=False)
        try: pair.to_parquet(out_dir / "var_style_pairwise.parquet", index=False)
        except Exception: pass
    if not breach.empty:
        breach.to_csv(out_dir / "var_style_breach_pairwise.csv", index=False)
        try: breach.to_parquet(out_dir / "var_style_breach_pairwise.parquet", index=False)
        except Exception: pass
    if not summary.empty:
        summary.to_csv(out_dir / "var_style_run_summary.csv", index=False)
    _augment_repair_summary_with_var(out_dir, summary)
    print(f"[var-style] arm={len(arm):,}, pairwise={len(pair):,}, breach={len(breach):,}", flush=True)
    return ret



# =============================================================================
# NNProj target model and projection
# =============================================================================
# Historical neural target projected into the common feasible set.

_NNPROJ_ARGS = {}
_NNPROJ_MODEL_INFO = None
_parse_base_args = parse_args
_run_repairs_without_nn = run_repairs
_solve_slice_repairs_without_nn = solve_slice_repairs


def _parse_nn_args(argv: Optional[Sequence[str]] = None):
    raw = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--include-ml-arm", action="store_true",
                    help="Include NNProj: a past-trained total-variance target followed by projection.")
    ap.add_argument("--nnproj-arm-name", default="NNPROJ")
    ap.add_argument("--nnproj-train-mode", choices=["past", "selected"], default="past",
                    help="past uses pre-evaluation rows; selected uses the supplied evaluation panel.")
    ap.add_argument("--nnproj-train-start", default=None, help="YYYY-MM or YYYY-MM-DD inclusive train start. Default uses lookback from train end.")
    ap.add_argument("--nnproj-train-end", default=None, help="YYYY-MM or YYYY-MM-DD exclusive train end. Default = eval month start minus embargo days.")
    ap.add_argument("--nnproj-train-lookback-months", type=int, default=12,
                    help="Past lookback months when train-start is omitted. 0 means all past rows before train-end.")
    ap.add_argument("--nnproj-embargo-days", type=int, default=7)
    ap.add_argument("--nnproj-hidden", default="64,64")
    ap.add_argument("--nnproj-alpha", type=float, default=1e-4)
    ap.add_argument("--nnproj-max-iter", type=int, default=500)
    ap.add_argument("--nnproj-max-train-rows", type=int, default=50000)
    ap.add_argument("--nnproj-min-train-rows", type=int, default=500)
    ap.add_argument("--nnproj-random-state", type=int, default=None)
    ns, rest = ap.parse_known_args(raw)
    extras = {
        "include_ml_arm": bool(ns.include_ml_arm),
        "nnproj_arm_name": str(ns.nnproj_arm_name).upper(),
        "nnproj_train_mode": str(ns.nnproj_train_mode),
        "nnproj_train_start": ns.nnproj_train_start,
        "nnproj_train_end": ns.nnproj_train_end,
        "nnproj_train_lookback_months": int(ns.nnproj_train_lookback_months),
        "nnproj_embargo_days": int(ns.nnproj_embargo_days),
        "nnproj_hidden": str(ns.nnproj_hidden),
        "nnproj_alpha": float(ns.nnproj_alpha),
        "nnproj_max_iter": int(ns.nnproj_max_iter),
        "nnproj_max_train_rows": int(ns.nnproj_max_train_rows),
        "nnproj_min_train_rows": int(ns.nnproj_min_train_rows),
        "nnproj_random_state": ns.nnproj_random_state,
    }
    return rest, extras


def parse_args(argv: Optional[Sequence[str]] = None):
    global _NNPROJ_ARGS
    rest, extras = _parse_nn_args(argv)
    args = _parse_base_args(rest)
    if extras["include_ml_arm"]:
        add = list(getattr(args, "add_arms", []) or [])
        if extras["nnproj_arm_name"] not in [str(a).upper() for a in add]:
            add.append(extras["nnproj_arm_name"])
        setattr(args, "add_arms", add)
    for k, v in extras.items():
        setattr(args, k, v)
    # OOS training metadata.
    extras["synthetic_call_panel"] = str(getattr(args, "synthetic_call_panel", ""))
    extras["month"] = list(getattr(args, "month", []) or [])
    extras["currency"] = str(getattr(args, "currency", "")).upper()
    extras["snapshot_hour"] = int(getattr(args, "snapshot_hour", 12))
    extras["dte_min_repair"] = float(getattr(args, "dte_min_repair", getattr(args, "dte_min", 2.0)) or 2.0)
    extras["dte_max_repair"] = float(getattr(args, "dte_max_repair", getattr(args, "dte_max", 30.0)) or 30.0)
    _NNPROJ_ARGS = dict(extras)
    return args


def _nn_enabled(cfg: RepairConfig) -> bool:
    return any(str(a).upper() in {"NNPROJ", "NN_PROJ", "MLPROJ", "ML_PROJ", "NEURAL_PROJ"} for a in getattr(cfg, "arms", ()))


def _parse_month_start(m: str) -> pd.Timestamp:
    s = str(m)
    if len(s) == 7:
        return pd.Timestamp(f"{s}-01", tz="UTC")
    return pd.Timestamp(s, tz="UTC")


def _infer_eval_start() -> pd.Timestamp:
    months = _NNPROJ_ARGS.get("month") or []
    if not months:
        raise ValueError("NNPROJ past training requires --month so eval start can be inferred")
    starts = [_parse_month_start(m) for m in months]
    return min(starts)


def _infer_train_window() -> Tuple[Optional[pd.Timestamp], pd.Timestamp]:
    if _NNPROJ_ARGS.get("nnproj_train_end"):
        train_end = _parse_month_start(_NNPROJ_ARGS["nnproj_train_end"])
    else:
        eval_start = _infer_eval_start()
        train_end = eval_start - pd.Timedelta(days=int(_NNPROJ_ARGS.get("nnproj_embargo_days", 7)))
    if _NNPROJ_ARGS.get("nnproj_train_start"):
        train_start = _parse_month_start(_NNPROJ_ARGS["nnproj_train_start"])
    else:
        lookback = int(_NNPROJ_ARGS.get("nnproj_train_lookback_months", 12))
        train_start = None if lookback <= 0 else train_end - pd.DateOffset(months=lookback)
    return train_start, train_end


def _load_nnproj_train_panel(cfg: RepairConfig) -> pd.DataFrame:
    """Load the dated training panel for NNProj."""
    mode = str(_NNPROJ_ARGS.get("nnproj_train_mode", "past"))
    if mode == "selected":
        return pd.DataFrame()
    path = Path(str(_NNPROJ_ARGS.get("synthetic_call_panel", "")))
    if not path.exists():
        raise FileNotFoundError(f"NNPROJ training panel not found: {path}")
    train_start, train_end = _infer_train_window()
    # Apply the training window after loading full history.
    train_cfg = replace(cfg, currency=str(_NNPROJ_ARGS.get("currency", cfg.currency)).upper(),
                        snapshot_hour=int(_NNPROJ_ARGS.get("snapshot_hour", cfg.snapshot_hour)))
    df = load_synthetic_panel(path, train_cfg, [],
                              dte_min=float(_NNPROJ_ARGS.get("dte_min_repair", cfg.dte_min_repair)),
                              dte_max=float(_NNPROJ_ARGS.get("dte_max_repair", cfg.dte_max_repair)),
                              extra_days=0)
    dt = pd.to_datetime(df["date"].astype("int64"), unit="s", utc=True)
    if train_start is not None:
        df = df[dt >= train_start].copy()
        dt = pd.to_datetime(df["date"].astype("int64"), unit="s", utc=True)
    df = df[dt < train_end].copy()
    return df.reset_index(drop=True)


def _nnproj_features_from_df(df: pd.DataFrame) -> np.ndarray:
    k = pd.to_numeric(df.get("k"), errors="coerce").to_numpy(dtype=float)
    tau = pd.to_numeric(df.get("tau"), errors="coerce").to_numpy(dtype=float)
    dte = pd.to_numeric(df.get("DTE"), errors="coerce").to_numpy(dtype=float)
    rel = pd.to_numeric(df.get("relspread_floor"), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(rel).any() and {"C_bid", "C_ask", "C_mid"}.issubset(df.columns):
        rel = (pd.to_numeric(df["C_ask"], errors="coerce") - pd.to_numeric(df["C_bid"], errors="coerce")).to_numpy(float) / np.maximum(np.abs(pd.to_numeric(df["C_mid"], errors="coerce").to_numpy(float)), 5e-4)
    # Exclude C_mid to prevent direct target copying.
    X = np.column_stack([
        k,
        np.abs(k),
        k ** 2,
        k ** 3,
        np.log(np.maximum(tau, 1e-8)),
        np.log1p(np.maximum(dte, 0.0)),
        np.log1p(np.clip(rel, 0.0, 100.0)),
    ])
    return np.where(np.isfinite(X), X, 0.0)


def _nnproj_features_from_arr(arr: Dict[str, np.ndarray], cfg: RepairConfig) -> np.ndarray:
    K = np.asarray(arr["K"], dtype=float)
    F = np.asarray(arr["F"], dtype=float)
    F0 = np.nanmedian(F[np.isfinite(F) & (F > 0)]) if np.isfinite(F).any() else np.nan
    if not np.isfinite(F0) or F0 <= 0:
        F0 = np.nanmedian(K[np.isfinite(K) & (K > 0)])
    k = np.log(np.maximum(K, EPS) / max(F0, EPS))
    tau = np.asarray(arr["tau"], dtype=float)
    rel = np.asarray(arr.get("scale", np.ones_like(K)), dtype=float) / np.maximum(np.abs(np.asarray(arr.get("mid", np.ones_like(K)), dtype=float)), float(getattr(cfg, "price_floor", 0.0005)))
    tmp = pd.DataFrame({"k": k, "tau": tau, "DTE": tau * 365.0, "relspread_floor": rel})
    return _nnproj_features_from_df(tmp)


def _nnproj_training_xy(df: pd.DataFrame, max_rows: int, random_state: int) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, str]]:
    cols = ["C_mid", "forward_price", "strike", "tau"]
    g = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
    g = g[(g["forward_price"] > 0) & (g["strike"] > 0) & (g["tau"] > 0) & (g["C_mid"].between(0, 1))]
    n_before = len(g)
    if len(g) > max_rows:
        g = g.sample(n=max_rows, random_state=random_state)
    sig = np.array([implied_vol_from_base_call_scalar(float(r.C_mid), float(r.forward_price), float(r.strike), float(r.tau)) for r in g.itertuples(index=False)], dtype=float)
    valid = np.isfinite(sig) & (sig > 0) & (sig < 10)
    g = g.iloc[np.where(valid)[0]].copy()
    sig = sig[valid]
    if len(g) == 0:
        return np.empty((0, 7)), np.array([]), n_before, {}
    tv = np.maximum(sig ** 2 * pd.to_numeric(g["tau"], errors="coerce").to_numpy(dtype=float), 1e-10)
    X = _nnproj_features_from_df(g)
    y = np.log(tv)
    valid2 = np.isfinite(X).all(axis=1) & np.isfinite(y)
    info = {}
    if len(g):
        dt = pd.to_datetime(g["date"].astype("int64"), unit="s", utc=True) if "date" in g else pd.Series([], dtype="datetime64[ns, UTC]")
        if len(dt):
            info["train_min_date"] = str(dt.min())
            info["train_max_date"] = str(dt.max())
    return X[valid2], y[valid2], n_before, info


def train_nnproj_model(selected_panel: pd.DataFrame, cfg: RepairConfig) -> Optional[Dict]:
    if not _nn_enabled(cfg):
        return None
    try:
        from sklearn.neural_network import MLPRegressor  # type: ignore
        from sklearn.pipeline import make_pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.metrics import r2_score  # type: ignore
    except Exception as e:
        return {"model": None, "status": "sklearn_not_installed", "message": str(e), "n_train": 0}
    mode = str(_NNPROJ_ARGS.get("nnproj_train_mode", "past"))
    if mode == "selected":
        train_df = selected_panel.copy()
        train_start = train_end = None
    else:
        train_start, train_end = _infer_train_window()
        train_df = _load_nnproj_train_panel(cfg)
    rs0 = _NNPROJ_ARGS.get("nnproj_random_state", None)
    rs = int(rs0 if rs0 is not None else getattr(cfg, "random_state", 7))
    max_rows = int(_NNPROJ_ARGS.get("nnproj_max_train_rows", 50000))
    min_rows = int(_NNPROJ_ARGS.get("nnproj_min_train_rows", 500))
    X, y, n_before, date_info = _nnproj_training_xy(train_df, max_rows=max_rows, random_state=rs)
    if len(y) < max(50, min_rows):
        return {"model": None, "status": "not_enough_oos_training_rows", "message": f"valid rows={len(y)}", "n_train": int(len(y)), "n_before_iv_filter": int(n_before), "train_mode": mode}
    hidden_txt = str(_NNPROJ_ARGS.get("nnproj_hidden", "64,64"))
    hidden = tuple(int(x) for x in hidden_txt.replace("x", ",").split(",") if str(x).strip()) or (64, 64)
    alpha = float(_NNPROJ_ARGS.get("nnproj_alpha", 1e-4))
    max_iter = int(_NNPROJ_ARGS.get("nnproj_max_iter", 500))
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=hidden, activation="tanh", solver="adam", alpha=alpha,
                     max_iter=max_iter, early_stopping=True, validation_fraction=0.15,
                     n_iter_no_change=20, random_state=rs)
    )
    try:
        model.fit(X, y)
        pred = model.predict(X)
        r2 = float(r2_score(y, pred)) if len(y) > 2 else np.nan
        ret = {"model": model, "status": "trained_oos" if mode != "selected" else "trained_selected",
               "n_train": int(len(y)), "n_before_iv_filter": int(n_before), "r2_train": r2,
               "target": "log_total_variance", "hidden": str(hidden), "alpha": alpha,
               "max_iter": max_iter, "train_mode": mode,
               "embargo_days": int(_NNPROJ_ARGS.get("nnproj_embargo_days", 7)),
               "lookback_months": int(_NNPROJ_ARGS.get("nnproj_train_lookback_months", 12))}
        if mode != "selected":
            ret["train_start"] = str(train_start) if train_start is not None else "all_past"
            ret["train_end_exclusive"] = str(train_end)
        ret.update(date_info)
        return ret
    except Exception as e:
        return {"model": None, "status": "train_failed", "message": str(e), "n_train": int(len(y)), "train_mode": mode}


def nnproj_projected_target(arr: Dict[str, np.ndarray], cfg: RepairConfig) -> Optional[np.ndarray]:
    info = _NNPROJ_MODEL_INFO
    if not info or info.get("model") is None:
        return None
    try:
        X = _nnproj_features_from_arr(arr, cfg)
        log_tv = np.asarray(info["model"].predict(X), dtype=float)
        log_tv = np.clip(log_tv, np.log(1e-10), np.log(25.0))
        tv = np.exp(log_tv)
        tau = np.asarray(arr["tau"], dtype=float)
        sig = np.sqrt(np.maximum(tv / np.maximum(tau, 1e-8), 1e-10))
        sig = np.clip(sig, 1e-5, 10.0)
        target = deribit_call_base_price_from_iv(arr["F"], arr["K"], tau, sig)
        target = np.asarray(target, dtype=float)
        target = np.where(np.isfinite(target), target, arr["mid"])
        return np.clip(target, 0.0, 1.0)
    except Exception:
        return None


def solve_slice_repairs(slc: pd.DataFrame, cfg: RepairConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    slc = slc.sort_values("strike").copy()
    arr = prepare_slice_arrays(slc, cfg)
    n = len(arr["K"])
    if n < cfg.min_strikes:
        return pd.DataFrame(), pd.DataFrame()
    arms = tuple(a.upper() for a in cfg.arms)
    solutions: Dict[str, Dict] = {}
    l1_c = None
    if "L1" in arms:
        sol = solve_l1(arr, cfg)
        solutions["L1"] = sol
        if sol.get("success") and sol.get("c") is not None:
            l1_c = sol["c"]
    for arm in arms:
        if arm == "L1":
            continue
        arm_norm = {"NN_PROJ": "NNPROJ", "MLPROJ": "NNPROJ", "ML_PROJ": "NNPROJ", "NEURAL_PROJ": "NNPROJ"}.get(arm, arm)
        if arm_norm == "NNPROJ":
            target = nnproj_projected_target(arr, cfg)
            if target is None:
                info = _NNPROJ_MODEL_INFO or {}
                solutions["NNPROJ"] = {"success": False, "status": info.get("status", "nnproj_no_oos_model"), "message": info.get("message", "NNPROJ OOS model unavailable"), "c": None,
                                        "nnproj_train_rows": info.get("n_train", 0), "nnproj_train_r2": info.get("r2_train", np.nan), "nnproj_status": info.get("status", None)}
            else:
                sol = solve_qp_like(arr, cfg, "NNPROJ", l1_c, target_mid=target, smooth_rho=0.0)
                info = _NNPROJ_MODEL_INFO or {}
                sol["nnproj_train_rows"] = info.get("n_train", np.nan)
                sol["nnproj_train_r2"] = info.get("r2_train", np.nan)
                sol["nnproj_status"] = info.get("status", None)
                sol["nnproj_train_mode"] = info.get("train_mode", None)
                solutions["NNPROJ"] = sol
        elif arm_norm == "L2":
            solutions[arm_norm] = solve_qp_like(arr, cfg, arm_norm, l1_c)
        elif arm_norm in {"SPREAD_WEIGHTED_L2", "FILTER_L2"}:
            w = filtered_weights_for_slice(slc, arr, cfg)
            solutions["SPREAD_WEIGHTED_L2" if arm_norm == "FILTER_L2" else arm_norm] = solve_qp_like(arr, cfg, arm_norm, l1_c, target_weights=w)
        elif arm_norm == "FILTER_RELAXED_L2":
            solutions[arm_norm] = solve_filter_relaxed_l2(arr, slc, cfg, l1_c)
        elif arm_norm == "SMOOTH" or arm_norm.startswith("SMOOTH_RHO_"):
            rho = parse_smooth_rho_from_arm(arm_norm, cfg)
            solutions[arm_norm] = solve_qp_like(arr, cfg, arm_norm, l1_c, smooth_rho=rho)
        elif arm_norm.startswith("BUDGET_"):
            b = parse_budget_from_arm(arm_norm)
            solutions[arm_norm] = solve_budget_smoothing(arr, cfg, b, l1_c) if b is not None else {"success": False, "status": "bad_budget_arm", "message": arm_norm, "c": None}
        elif arm_norm == "IVSPLINE":
            target = ivspline_projected_target(arr, cfg)
            solutions[arm_norm] = solve_qp_like(arr, cfg, arm_norm, l1_c, target_mid=target, smooth_rho=cfg.smooth_rho) if target is not None else {"success": False, "status": "ivspline_failed", "message": "could not build IV spline target", "c": None}
        elif arm_norm == "SVI":
            target = svi_projected_target(arr, cfg)
            solutions[arm_norm] = solve_qp_like(arr, cfg, arm_norm, l1_c, target_mid=target, smooth_rho=cfg.smooth_rho) if target is not None else {"success": False, "status": "svi_failed", "message": "could not fit SVI target", "c": None}
        elif arm_norm == "ELASTIC_L2":
            cfg_el = replace(cfg, elastic_box=True)
            solutions[arm_norm] = solve_qp_like(arr, cfg_el, arm_norm, l1_c, cfg_override=cfg_el)
        else:
            solutions[arm_norm] = {"success": False, "status": "unknown_arm", "message": f"Unknown arm {arm_norm}", "c": None}

    key_vals = {c: slc.iloc[0][c] for c in slice_key_cols()}
    diag_rows, surface_rows = [], []
    for arm, sol in solutions.items():
        if _cfg_get(cfg, "exclude_optimal_inaccurate", False) and str(sol.get("status", "")).lower() == "optimal_inaccurate":
            sol = dict(sol); sol["success"] = False; sol["message"] = str(sol.get("message", "")) + "; excluded optimal_inaccurate"
        diag = {**key_vals, "arm": arm, "n_strikes": n, "success": bool(sol.get("success", False)), "status": sol.get("status"),
                "solver_status_inaccurate": "inaccurate" in str(sol.get("status", "")).lower(), "message": sol.get("message"),
                "objective": finite(sol.get("objective")), "bad_box_count": int(arr.get("bad_box_count", 0)), "mid_clip_count": int(arr.get("mid_clip_count", 0)),
                "filtered_quote_share": finite(sol.get("filtered_quote_share"), np.nan),
                "nnproj_train_rows": sol.get("nnproj_train_rows", np.nan), "nnproj_train_r2": sol.get("nnproj_train_r2", np.nan),
                "nnproj_status": sol.get("nnproj_status", None), "nnproj_train_mode": sol.get("nnproj_train_mode", None)}
        for k in ["verified_feasible", "max_mono_violation", "max_conv_violation",
                  "max_vertical_spread_violation", "max_vertical_slope", "max_constraint_violation",
                  "max_intrinsic_violation", "max_upper_violation", "box_slack_sum",
                  "max_box_low_violation", "max_box_high_violation"]:
            diag[k] = sol.get(k, np.nan)
        diag_rows.append(diag)
        if not sol.get("success", False) or sol.get("c") is None:
            continue
        c_hat = np.asarray(sol["c"], dtype=float)
        for j, (_, row) in enumerate(slc.iterrows()):
            surface_rows.append({
                **{col: row[col] for col in ["currency", "date", "expiration_date", "strike", "forward_price", "underlyer_spot", "tau", "DTE", "dte_bucket", "k", "C_bid", "C_ask", "C_mid", "intrinsic", "relspread_floor", "h_abs", "h_norm", "ranking_score", "ranking_score_rel_h2", "ranking_score_abs_h2", "slice_ranking_score", "threshold_screen_obs", "short_dte", "wing", "wide_spread", "sparse_strike", "high_score_cell", "threshold_screen_cell", "edge_strike", "core_repair_slice", "full_feasible"] if col in row.index},
                "arm": arm,
                "C_repaired": float(c_hat[j]),
                "repair_abs_mid_diff": float(abs(c_hat[j] - arr["mid"][j])),
                "repair_norm_mid_diff": float(abs(c_hat[j] - arr["mid"][j]) / (arr["scale"][j] + EPS)),
            })
    return pd.DataFrame(surface_rows), pd.DataFrame(diag_rows)


def run_repairs(panel_repair: pd.DataFrame, cfg: RepairConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    global _NNPROJ_MODEL_INFO
    if _nn_enabled(cfg):
        print("[nnproj-oos] training embargoed neural total-variance smoother...", flush=True)
        _NNPROJ_MODEL_INFO = train_nnproj_model(panel_repair, cfg)
        print(f"[nnproj-oos] status={(_NNPROJ_MODEL_INFO or {}).get('status')}; n={(_NNPROJ_MODEL_INFO or {}).get('n_train')}; r2={(_NNPROJ_MODEL_INFO or {}).get('r2_train')}", flush=True)
    else:
        _NNPROJ_MODEL_INFO = None
    surfaces = []
    diags = []
    groups = list(panel_repair.groupby(slice_key_cols(), sort=False))
    for i, (_, slc) in enumerate(groups, 1):
        srf, d = solve_slice_repairs(slc, cfg)
        if len(srf):
            surfaces.append(srf)
        if len(d):
            diags.append(d)
        if i % 25 == 0:
            print(f"[repair] solved {i}/{len(groups)} slices", flush=True)
    return (pd.concat(surfaces, ignore_index=True) if surfaces else pd.DataFrame(),
            pd.concat(diags, ignore_index=True) if diags else pd.DataFrame())


_main_without_nn = main


def main(argv: Optional[Sequence[str]] = None) -> int:
    ret = _main_without_nn(argv)
    try:
        args = parse_args(argv)
        out_dir = Path(args.output_dir)
        if getattr(args, "include_ml_arm", False):
            ml_info = {
                "enabled": True,
                "arm_name": getattr(args, "nnproj_arm_name", "NNPROJ"),
                "train_mode": _NNPROJ_ARGS.get("nnproj_train_mode"),
                "train_start": _NNPROJ_ARGS.get("nnproj_train_start"),
                "train_end": _NNPROJ_ARGS.get("nnproj_train_end"),
                "train_lookback_months": _NNPROJ_ARGS.get("nnproj_train_lookback_months"),
                "embargo_days": _NNPROJ_ARGS.get("nnproj_embargo_days"),
                "args": {k: v for k, v in _NNPROJ_ARGS.items() if k not in {"synthetic_call_panel"}},
                "model_info": {k: v for k, v in (_NNPROJ_MODEL_INFO or {}).items() if k != "model"},
                "note": "NNProj uses a dated training panel and projects its target into the common feasible set."
            }
            (out_dir / "nnproj_model_summary.json").write_text(json.dumps(ml_info, indent=2, default=str), encoding="utf-8")
            cfg_file = out_dir / "repair_config.json"
            if cfg_file.exists():
                try:
                    data = json.loads(cfg_file.read_text(encoding="utf-8"))
                    data["nnproj"] = ml_info
                    cfg_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
                except Exception:
                    pass
    except Exception:
        pass
    return ret



# =============================================================================
# NNProj training-window checks
# =============================================================================
_NNPROJ_HELP = """
repair_analysis.py

Core usage:
  python repair_analysis.py \
    --synthetic-call-panel FULL_SYNTHETIC_CALL_PANEL.parquet \
    --output-dir results/BTC_2022_11_nnproj_top50 \
    --currency BTC --snapshot-hour 12 --month 2022-11 \
    --dte-min-repair 2 --dte-max-repair 30 \
    --dte-min-marking 0 --dte-max-marking 31 --marking-extra-days 1 \
    --arms L1 L2 --smooth-rhos 0.01 0.1 1 10 \
    --include-ml-arm --nnproj-train-mode past \
    --nnproj-train-lookback-months 12 --nnproj-embargo-days 7 \
    --ranking-score-method relspread_h2 --ranking-slice-agg p90 \
    --selected-slices 50 --slice-selection top --random-seed 20260605

NNPROJ OOS arguments:
  --include-ml-arm                     Add the learned repair NNPROJ.
  --nnproj-arm-name NAME               Default NNPROJ.
  --nnproj-train-mode {past,selected}
                                       past uses only pre-evaluation data; selected includes evaluation-period data.
  --nnproj-train-start YYYY-MM[-DD]    Inclusive training start. Optional.
  --nnproj-train-end YYYY-MM[-DD]      Exclusive training end. Must be <= eval_start - embargo.
  --nnproj-train-lookback-months N     Default 12. 0 means all past rows before train-end.
  --nnproj-embargo-days N              Default 7.
  --nnproj-hidden 64,64                MLP hidden layer sizes.
  --nnproj-alpha FLOAT                 MLP L2 regularization.
  --nnproj-max-iter N                  MLP max iterations.
  --nnproj-max-train-rows N            Cap training rows.
  --nnproj-min-train-rows N            Minimum post-IV-filter training rows. Default 500.
  --nnproj-min-train-months N          Minimum distinct OOS train months. Default 2.
  --nnproj-random-state N              Random state.
  --nnproj-allow-untrained             Continue when target training returns no model.
  --nnproj-require-oos-trained         Require a dated training panel and completed window checks.

Input requirements:
  * Provide a full historical synthetic_call_panel parquet.
  * The analysis enforces train_end <= eval_month_start - embargo_days.
  * The default requires a completed past-data fit and a valid training window.
"""

_parse_nn_args_initial = _parse_nn_args
_nnproj_training_xy_initial = _nnproj_training_xy
_train_nnproj_model_initial = train_nnproj_model


def _parse_nn_args(argv: Optional[Sequence[str]] = None):
    raw = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--include-ml-arm", action="store_true")
    ap.add_argument("--nnproj-arm-name", default="NNPROJ")
    ap.add_argument("--nnproj-train-mode", choices=["past", "selected"], default="past")
    ap.add_argument("--nnproj-train-start", default=None)
    ap.add_argument("--nnproj-train-end", default=None)
    ap.add_argument("--nnproj-train-lookback-months", type=int, default=12)
    ap.add_argument("--nnproj-embargo-days", type=int, default=7)
    ap.add_argument("--nnproj-hidden", default="64,64")
    ap.add_argument("--nnproj-alpha", type=float, default=1e-4)
    ap.add_argument("--nnproj-max-iter", type=int, default=500)
    ap.add_argument("--nnproj-max-train-rows", type=int, default=50000)
    ap.add_argument("--nnproj-min-train-rows", type=int, default=500)
    ap.add_argument("--nnproj-min-train-months", type=int, default=2)
    ap.add_argument("--nnproj-random-state", type=int, default=None)
    ap.add_argument("--nnproj-allow-untrained", action="store_true")
    ap.add_argument("--nnproj-require-oos-trained", action="store_true")
    ns, rest = ap.parse_known_args(raw)
    extras = {
        "include_ml_arm": bool(ns.include_ml_arm),
        "nnproj_arm_name": str(ns.nnproj_arm_name).upper(),
        "nnproj_train_mode": str(ns.nnproj_train_mode),
        "nnproj_train_start": ns.nnproj_train_start,
        "nnproj_train_end": ns.nnproj_train_end,
        "nnproj_train_lookback_months": int(ns.nnproj_train_lookback_months),
        "nnproj_embargo_days": int(ns.nnproj_embargo_days),
        "nnproj_hidden": str(ns.nnproj_hidden),
        "nnproj_alpha": float(ns.nnproj_alpha),
        "nnproj_max_iter": int(ns.nnproj_max_iter),
        "nnproj_max_train_rows": int(ns.nnproj_max_train_rows),
        "nnproj_min_train_rows": int(ns.nnproj_min_train_rows),
        "nnproj_min_train_months": int(ns.nnproj_min_train_months),
        "nnproj_random_state": ns.nnproj_random_state,
        "nnproj_allow_untrained": bool(ns.nnproj_allow_untrained),
        "nnproj_require_oos_trained": bool(ns.nnproj_require_oos_trained or not ns.nnproj_allow_untrained),
    }
    return rest, extras


def _infer_train_window() -> Tuple[Optional[pd.Timestamp], pd.Timestamp]:
    """Infer and validate the NNProj training window."""
    eval_start = _infer_eval_start()
    embargo_days = int(_NNPROJ_ARGS.get("nnproj_embargo_days", 7))
    latest_train_end = eval_start - pd.Timedelta(days=embargo_days)
    if _NNPROJ_ARGS.get("nnproj_train_end"):
        train_end = _parse_month_start(_NNPROJ_ARGS["nnproj_train_end"])
        if train_end > latest_train_end:
            raise ValueError(
                "NNProj training end is outside the permitted window: "
                f"{train_end} is after eval_start - embargo ({latest_train_end}). "
                "Use a strictly past train_end or omit --nnproj-train-end."
            )
    else:
        train_end = latest_train_end
    if _NNPROJ_ARGS.get("nnproj_train_start"):
        train_start = _parse_month_start(_NNPROJ_ARGS["nnproj_train_start"])
        if train_start >= train_end:
            raise ValueError(f"NNPROJ train_start {train_start} must be before train_end {train_end}")
    else:
        lookback = int(_NNPROJ_ARGS.get("nnproj_train_lookback_months", 12))
        train_start = None if lookback <= 0 else train_end - pd.DateOffset(months=lookback)
    return train_start, train_end


def _nnproj_training_xy(df: pd.DataFrame, max_rows: int, random_state: int) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, str]]:
    X, y, n_before, info = _nnproj_training_xy_initial(df, max_rows=max_rows, random_state=random_state)
    try:
        if "date" in df.columns and len(df):
            dt = pd.to_datetime(df["date"].astype("int64"), unit="s", utc=True, errors="coerce")
            dt = dt[pd.notna(dt)]
            if len(dt):
                info["train_min_date_raw"] = str(dt.min())
                info["train_max_date_raw"] = str(dt.max())
                info["n_train_months_raw"] = str(int(dt.dt.to_period("M").nunique()))
        g = df.copy()
        for c in ["C_mid", "forward_price", "strike", "tau", "date"]:
            if c not in g.columns:
                raise KeyError(c)
        g = g.dropna(subset=["C_mid", "forward_price", "strike", "tau", "date"])
        g = g[(g["forward_price"] > 0) & (g["strike"] > 0) & (g["tau"] > 0) & (g["C_mid"].between(0, 1))]
        dt2 = pd.to_datetime(g["date"].astype("int64"), unit="s", utc=True, errors="coerce")
        dt2 = dt2[pd.notna(dt2)]
        if len(dt2):
            info["train_min_date_valid_prefilter"] = str(dt2.min())
            info["train_max_date_valid_prefilter"] = str(dt2.max())
            info["n_train_months_valid_prefilter"] = str(int(dt2.dt.to_period("M").nunique()))
    except Exception as e:
        info["train_span_warning"] = str(e)
    return X, y, n_before, info


def train_nnproj_model(selected_panel: pd.DataFrame, cfg: RepairConfig) -> Optional[Dict]:
    ret = _train_nnproj_model_initial(selected_panel, cfg)
    if not ret or not _nn_enabled(cfg):
        return ret
    ret = dict(ret)
    mode = str(ret.get("train_mode", _NNPROJ_ARGS.get("nnproj_train_mode", "past")))
    if mode != "selected" and ret.get("status") == "trained_oos":
        min_months = int(_NNPROJ_ARGS.get("nnproj_min_train_months", 2))
        n_months = 0
        for key in ["n_train_months_valid_prefilter", "n_train_months_raw"]:
            try:
                n_months = max(n_months, int(ret.get(key, 0)))
            except Exception:
                pass
        ret["n_train_months"] = int(n_months)
        ret["min_train_months_required"] = int(min_months)
        try:
            eval_start = _infer_eval_start()
            embargo_days = int(_NNPROJ_ARGS.get("nnproj_embargo_days", 7))
            allowed_end = eval_start - pd.Timedelta(days=embargo_days)
            tmax = pd.Timestamp(ret.get("train_max_date", ret.get("train_max_date_raw", "NaT")))
            if pd.notna(tmax) and tmax.tzinfo is None:
                tmax = tmax.tz_localize("UTC")
            ret["evaluation_start"] = str(eval_start)
            ret["latest_training_date"] = str(allowed_end)
            ret["training_window_valid"] = bool(pd.isna(tmax) or tmax < allowed_end)
            if not ret["training_window_valid"]:
                ret["model"] = None
                ret["status"] = "invalid_training_window"
                ret["message"] = f"train_max_date {tmax} is not before allowed train end {allowed_end}"
                return ret
        except Exception as e:
            ret["training_window_note"] = str(e)
        ret["history_panel_ok"] = bool(ret.get("training_window_valid", True) and n_months >= min_months)
        if n_months < min_months:
            ret["model"] = None
            ret["status"] = "not_enough_oos_train_months"
            ret["history_panel_ok"] = False
            ret["message"] = f"distinct train months={n_months}, required={min_months}"
            return ret
    return ret


_run_repairs_with_base_checks = run_repairs


def run_repairs(panel_repair: pd.DataFrame, cfg: RepairConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    global _NNPROJ_MODEL_INFO
    if _nn_enabled(cfg):
        print("[nnproj] fitting the dated total-implied-variance target...", flush=True)
        _NNPROJ_MODEL_INFO = train_nnproj_model(panel_repair, cfg)
        status = (_NNPROJ_MODEL_INFO or {}).get("status")
        print(f"[nnproj] status={status}; n={(_NNPROJ_MODEL_INFO or {}).get('n_train')}; r2={(_NNPROJ_MODEL_INFO or {}).get('r2_train')}", flush=True)
        if str(_NNPROJ_ARGS.get("nnproj_train_mode", "past")) != "selected":
            allow = bool(_NNPROJ_ARGS.get("nnproj_allow_untrained", False))
            require = bool(_NNPROJ_ARGS.get("nnproj_require_oos_trained", True))
            ok = bool((_NNPROJ_MODEL_INFO or {}).get("status") == "trained_oos" and (_NNPROJ_MODEL_INFO or {}).get("training_window_valid", True))
            # Enforce month-count and date-window checks.
            if "n_train_months" in (_NNPROJ_MODEL_INFO or {}):
                ok = ok and int((_NNPROJ_MODEL_INFO or {}).get("n_train_months", 0)) >= int((_NNPROJ_MODEL_INFO or {}).get("min_train_months_required", 2))
            if require and not allow and not ok:
                model_info = {k: v for k, v in (_NNPROJ_MODEL_INFO or {}).items() if k != "model"}
                raise RuntimeError(
                    "NNProj did not complete the required past-data training-window checks. "
                    f"model_info={model_info}"
                )
    else:
        _NNPROJ_MODEL_INFO = None
    surfaces = []
    diags = []
    groups = list(panel_repair.groupby(slice_key_cols(), sort=False))
    for i, (_, slc) in enumerate(groups, 1):
        srf, d = solve_slice_repairs(slc, cfg)
        if len(srf):
            surfaces.append(srf)
        if len(d):
            diags.append(d)
        if i % 25 == 0:
            print(f"[repair] solved {i}/{len(groups)} slices", flush=True)
    return (pd.concat(surfaces, ignore_index=True) if surfaces else pd.DataFrame(),
            pd.concat(diags, ignore_index=True) if diags else pd.DataFrame())


_main_with_nn = main


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if any(x in raw for x in ["-h", "--help"]):
        print(_NNPROJ_HELP)
        return 0
    ret = _main_with_nn(raw)
    try:
        args = parse_args(raw)
        out_dir = Path(args.output_dir)
        if getattr(args, "include_ml_arm", False):
            path = out_dir / "nnproj_model_summary.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                data = {"enabled": True}
            data["training_checks"] = {k: v for k, v in (_NNPROJ_MODEL_INFO or {}).items() if k != "model"}
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
    return ret


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    raise SystemExit(main())
