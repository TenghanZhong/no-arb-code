"""Isolated calendar-certified robustness pilot for the repair-risk paper.

This script does not modify the main Phase 2 pipeline.  It implements a small,
joint two-maturity QP to compare strike-slice repair with an additional
forward-moneyness calendar screen:

    c_long(x_g) >= c_short(x_g)

at common normalized-moneyness grid points x_g.  The calendar condition is a
robustness screen for the paper's forward-normalized call convention; it is not
a replacement for a full exchange-calendar arbitrage system.

Outputs are written to an isolated diagnostics directory.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

import sys
_COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
from full_strike_constraints import build_full_constraint_matrix, effective_bounds  # noqa: E402

_WORKING_DIR = Path(__file__).resolve().parents[1]
if str(_WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKING_DIR))
from artifact_paths import PANEL as PANEL_DEFAULT, RESULTS  # noqa: E402


OUTPUT_DEFAULT = RESULTS / "calendar"

PRICE_FLOOR = 5e-4
EPS = 1e-12


@dataclass(frozen=True)
class SliceArrays:
    currency: str
    date: int
    expiration: int
    dte: float
    strike: np.ndarray
    x: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    mid: np.ndarray
    intrinsic: np.ndarray
    scale: np.ndarray
    w_l2: np.ndarray
    danger_p90: float


@dataclass(frozen=True)
class PairCandidate:
    currency: str
    month: str
    date: int
    short_expiration: int
    long_expiration: int
    pair_score: float
    rank: int


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def to_month(epoch_seconds: pd.Series) -> pd.Series:
    return pd.to_datetime(epoch_seconds.astype("int64"), unit="s", utc=True).dt.strftime("%Y-%m")


def second_diff_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    rows = []
    for i in range(1, len(x) - 1):
        hm = x[i] - x[i - 1]
        hp = x[i + 1] - x[i]
        if hm <= 0 or hp <= 0:
            continue
        row = np.zeros(len(x))
        row[i - 1] = 2.0 / (hm + hp) * (1.0 / hm)
        row[i] = 2.0 / (hm + hp) * (-1.0 / hm - 1.0 / hp)
        row[i + 1] = 2.0 / (hm + hp) * (1.0 / hp)
        rows.append(row)
    return np.asarray(rows) if rows else np.zeros((0, len(x)))


def convex_ub_rows(x: np.ndarray, n_total: int, offset: int) -> list[np.ndarray]:
    rows = []
    for i in range(1, len(x) - 1):
        hm = x[i] - x[i - 1]
        hp = x[i + 1] - x[i]
        if hm <= 0 or hp <= 0:
            continue
        row = np.zeros(n_total)
        row[offset + i - 1] = -1.0 / hm
        row[offset + i] = 1.0 / hm + 1.0 / hp
        row[offset + i + 1] = -1.0 / hp
        rows.append(row)
    return rows


def monotone_ub_rows(n: int, n_total: int, offset: int) -> list[np.ndarray]:
    rows = []
    for i in range(n - 1):
        row = np.zeros(n_total)
        row[offset + i + 1] = 1.0
        row[offset + i] = -1.0
        rows.append(row)
    return rows


def interpolation_row(x: np.ndarray, xg: float, n_total: int, offset: int) -> np.ndarray | None:
    if xg < x[0] - 1e-14 or xg > x[-1] + 1e-14:
        return None
    j = int(np.searchsorted(x, xg, side="right") - 1)
    j = max(0, min(j, len(x) - 2))
    x0, x1 = x[j], x[j + 1]
    if x1 <= x0:
        return None
    w1 = (xg - x0) / (x1 - x0)
    row = np.zeros(n_total)
    row[offset + j] = 1.0 - w1
    row[offset + j + 1] = w1
    return row


def calendar_grid(short: SliceArrays, long: SliceArrays, n_grid: int) -> np.ndarray:
    lo = max(float(short.x[0]), float(long.x[0]))
    hi = min(float(short.x[-1]), float(long.x[-1]))
    if hi <= lo:
        return np.zeros(0)
    candidates = []
    for x in [short.x, long.x]:
        candidates.extend([float(v) for v in x if lo + 1e-12 <= v <= hi - 1e-12])
    if len(candidates) >= n_grid:
        qs = np.linspace(0.05, 0.95, n_grid)
        grid = np.quantile(np.asarray(candidates), qs)
    else:
        grid = np.linspace(lo, hi, n_grid + 2)[1:-1]
    grid = np.unique(np.round(grid, 12))
    return grid


def prepare_slice(slc: pd.DataFrame, min_strikes: int, max_strikes: int | None) -> SliceArrays | None:
    required = [
        "currency",
        "date",
        "expiration_date",
        "strike",
        "forward_price",
        "DTE",
        "C_bid",
        "C_ask",
        "C_mid",
        "intrinsic",
        "relspread",
    ]
    require_columns(slc, required, "panel slice")
    s = slc.sort_values("strike").copy()
    finite = np.isfinite(s[["strike", "forward_price", "C_bid", "C_ask", "C_mid", "intrinsic"]]).all(axis=1)
    s = s.loc[finite].copy()
    if "synthetic_valid" in s.columns:
        s = s.loc[s["synthetic_valid"].astype(bool)].copy()
    if len(s) < min_strikes:
        return None
    if max_strikes is not None and len(s) > max_strikes:
        keep_idx = np.linspace(0, len(s) - 1, max_strikes).round().astype(int)
        s = s.iloc[np.unique(keep_idx)].copy()
    f0 = float(np.nanmedian(s["forward_price"]))
    if not math.isfinite(f0) or f0 <= 0:
        return None
    strike = s["strike"].to_numpy(float)
    x = strike / f0
    if np.any(np.diff(x) <= 0):
        return None
    bid = np.nan_to_num(s["C_bid"].to_numpy(float), nan=0.0, neginf=0.0, posinf=1.0)
    ask = np.nan_to_num(s["C_ask"].to_numpy(float), nan=1.0, neginf=0.0, posinf=1.0)
    mid = np.nan_to_num(s["C_mid"].to_numpy(float), nan=0.5 * (bid + ask))
    intrinsic = np.maximum(1.0 - x, 0.0)
    bad_box = ask < bid
    if np.any(bad_box):
        center = 0.5 * (ask[bad_box] + bid[bad_box])
        bid[bad_box] = center
        ask[bad_box] = center
    bid = np.clip(bid, 0.0, 1.0)
    ask = np.clip(ask, 0.0, 1.0)
    mid = np.clip(mid, 0.0, 1.0)
    intrinsic = np.clip(intrinsic, 0.0, 1.0)
    lower, upper = effective_bounds(x, bid, ask, intrinsic)
    if np.any(lower > upper + 1e-12):
        return None
    scale = np.maximum(ask - bid, PRICE_FLOOR)
    w_l2 = 1.0 / scale**2
    lo_q, hi_q = np.nanquantile(w_l2, [0.05, 0.95])
    w_l2 = np.clip(w_l2, lo_q, hi_q)
    h = np.full(len(x), np.nan)
    if len(x) >= 3:
        h[1:-1] = 0.5 * (x[2:] - x[:-2])
    rel = np.nan_to_num(s["relspread"].to_numpy(float), nan=np.nan)
    danger = rel / np.maximum(h, 1e-8) ** 2
    danger_p90 = float(np.nanquantile(danger[np.isfinite(danger)], 0.90)) if np.isfinite(danger).any() else 0.0
    return SliceArrays(
        currency=str(s.iloc[0]["currency"]),
        date=int(s.iloc[0]["date"]),
        expiration=int(s.iloc[0]["expiration_date"]),
        dte=float(np.nanmedian(s["DTE"])),
        strike=strike,
        x=x,
        bid=bid,
        ask=ask,
        mid=mid,
        intrinsic=intrinsic,
        scale=scale,
        w_l2=w_l2,
        danger_p90=danger_p90,
    )


def build_constraints(short: SliceArrays, long: SliceArrays, *, add_calendar: bool, n_grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n1, n2 = len(short.x), len(long.x)
    n_total = n1 + n2
    rows = []
    rhs = []
    for x, offset in ((short.x, 0), (long.x, n1)):
        block_A, block_b, _ = build_full_constraint_matrix(x)
        for base_row, bound in zip(block_A, block_b):
            row = np.zeros(n_total)
            row[offset : offset + len(x)] = base_row
            rows.append(row)
            rhs.append(float(bound))
    grid = calendar_grid(short, long, n_grid) if add_calendar else np.zeros(0)
    for xg in grid:
        r_short = interpolation_row(short.x, float(xg), n_total, 0)
        r_long = interpolation_row(long.x, float(xg), n_total, n1)
        if r_short is None or r_long is None:
            continue
        rows.append(r_short - r_long)
        rhs.append(0.0)
    A = np.vstack(rows) if rows else np.zeros((0, n_total))
    b = np.asarray(rhs, dtype=float)
    if len(A):
        row_scale = np.max(np.abs(A), axis=1)
        row_scale = np.where(row_scale > 0, row_scale, 1.0)
        A = A / row_scale[:, None]
        b = b / row_scale
    return A, b, grid


def feasibility_point(bounds: Bounds, A: np.ndarray, b: np.ndarray) -> tuple[bool, np.ndarray | None, str]:
    c = np.zeros(len(bounds.lb))
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(bounds.lb, bounds.ub)), method="highs")
    except Exception as exc:
        return False, None, f"linprog_exception:{exc}"
    if not res.success:
        return False, None, f"linprog_status:{res.status}:{res.message}"
    return True, np.asarray(res.x, dtype=float), "linprog_feasible"


def solve_pair_qp(
    short: SliceArrays,
    long: SliceArrays,
    *,
    rho: float,
    add_calendar: bool,
    n_grid: int,
    maxiter: int,
) -> dict:
    n1, n2 = len(short.x), len(long.x)
    short_lower, short_upper = effective_bounds(short.x, short.bid, short.ask, short.intrinsic)
    long_lower, long_upper = effective_bounds(long.x, long.bid, long.ask, long.intrinsic)
    lower = np.concatenate([short_lower, long_lower])
    upper = np.concatenate([short_upper, long_upper])
    bounds = Bounds(lower, upper)
    A, b, grid = build_constraints(short, long, add_calendar=add_calendar, n_grid=n_grid)
    feasible, x0, status = feasibility_point(bounds, A, b)
    if not feasible or x0 is None:
        return {"success": False, "status": status, "x": None, "calendar_grid_n": len(grid)}
    mid = np.concatenate([short.mid, long.mid])
    w = np.concatenate([short.w_l2, long.w_l2])
    D_short = second_diff_matrix(short.x)
    D_long = second_diff_matrix(long.x)

    hess = np.diag(2.0 * w)
    if rho > 0:
        if D_short.size:
            hess[:n1, :n1] += 2.0 * rho * (D_short.T @ D_short)
        if D_long.size:
            hess[n1:, n1:] += 2.0 * rho * (D_long.T @ D_long)
    obj_scale = max(1.0, float(np.nanmedian(np.abs(np.diag(hess)))))

    def objective_raw(xv: np.ndarray) -> float:
        fit = float(np.sum(w * (xv - mid) ** 2))
        if rho <= 0:
            return fit
        c1 = xv[:n1]
        c2 = xv[n1:]
        smooth = 0.0
        if D_short.size:
            smooth += float(np.sum((D_short @ c1) ** 2))
        if D_long.size:
            smooth += float(np.sum((D_long @ c2) ** 2))
        return fit + rho * smooth

    def objective(xv: np.ndarray) -> float:
        return objective_raw(xv) / obj_scale

    def gradient(xv: np.ndarray) -> np.ndarray:
        grad = 2.0 * w * (xv - mid)
        if rho > 0:
            if D_short.size:
                grad[:n1] += 2.0 * rho * (D_short.T @ (D_short @ xv[:n1]))
            if D_long.size:
                grad[n1:] += 2.0 * rho * (D_long.T @ (D_long @ xv[n1:]))
        return grad / obj_scale

    def hessian(_: np.ndarray) -> np.ndarray:
        return hess / obj_scale

    constraints = [LinearConstraint(A, -np.inf * np.ones(len(b)), b)] if len(b) else []
    attempts = []
    try:
        res = minimize(
            objective,
            x0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": 1e-10, "disp": False},
        )
        attempts.append(("slsqp", res))
    except Exception as exc:
        attempts.append(("slsqp", None, exc))
    if not attempts[-1][1] or not attempts[-1][1].success:
        try:
            res = minimize(
                objective,
                x0,
                jac=gradient,
                hess=hessian,
                method="trust-constr",
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": maxiter,
                    "gtol": 1e-9,
                    "xtol": 1e-10,
                    "barrier_tol": 1e-10,
                    "verbose": 0,
                },
            )
            attempts.append(("trust-constr", res))
        except Exception as exc:
            attempts.append(("trust-constr", None, exc))
    method, res = next(((m, r) for item in reversed(attempts) for m, r, *rest in [item] if r is not None), ("none", None))
    if res is None or not res.success:
        messages = []
        for item in attempts:
            if len(item) == 3:
                messages.append(f"{item[0]}_exception:{item[2]}")
            else:
                messages.append(f"{item[0]}_status:{item[1].status}:{item[1].message}")
        return {"success": False, "status": " | ".join(messages), "x": None, "calendar_grid_n": len(grid)}
    xv = np.asarray(res.x, dtype=float)
    max_ub = float(np.max(A @ xv - b)) if len(b) else 0.0
    max_lower = float(np.max(bounds.lb - xv))
    max_upper = float(np.max(xv - bounds.ub))
    ok = max(max_ub, max_lower, max_upper) <= 5e-6
    return {
        "success": bool(ok),
        "status": "optimal" if ok else "postcheck_failed",
        "objective": float(objective_raw(xv)),
        "method": method,
        "x": xv,
        "max_linear_violation": max_ub,
        "max_lower_violation": max_lower,
        "max_upper_violation": max_upper,
        "calendar_grid_n": len(grid),
    }


def max_calendar_violation(short: SliceArrays, long: SliceArrays, c_short: np.ndarray, c_long: np.ndarray, n_grid: int) -> float:
    grid = calendar_grid(short, long, n_grid)
    if len(grid) == 0:
        return float("nan")
    vals = []
    for xg in grid:
        s = np.interp(xg, short.x, c_short)
        l = np.interp(xg, long.x, c_long)
        vals.append(max(0.0, s - l))
    return float(max(vals)) if vals else float("nan")


def curvature_summary(short: SliceArrays, long: SliceArrays, x_l2: np.ndarray, x_sm: np.ndarray) -> dict:
    n1 = len(short.x)
    diffs = []
    scaled = []
    for slc, a, b in [(short, x_l2[:n1], x_sm[:n1]), (long, x_l2[n1:], x_sm[n1:])]:
        D = second_diff_matrix(slc.x)
        if not D.size:
            continue
        ca = D @ a
        cb = D @ b
        d = np.abs(cb - ca)
        scale = np.nanmedian(np.abs(ca)) + 1e-8
        diffs.extend(d.tolist())
        scaled.extend((d / scale).tolist())
    if not diffs:
        return {"p90_abs_curvature_diff": np.nan, "p90_scaled_curvature_diff": np.nan, "n_curvature_rows": 0}
    return {
        "p90_abs_curvature_diff": float(np.nanquantile(diffs, 0.90)),
        "p90_scaled_curvature_diff": float(np.nanquantile(scaled, 0.90)),
        "n_curvature_rows": int(len(diffs)),
    }


def load_panel(path: Path, currencies: list[str], months: list[str], dte_min: float, dte_max: float) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    required = [
        "currency",
        "date",
        "expiration_date",
        "strike",
        "forward_price",
        "DTE",
        "C_bid",
        "C_ask",
        "C_mid",
        "intrinsic",
        "relspread",
    ]
    require_columns(df, required, "synthetic_call_panel")
    df = df[df["currency"].isin(currencies)].copy()
    df["month"] = to_month(df["date"])
    df = df[df["month"].isin(months)].copy()
    df = df[(df["DTE"] > dte_min) & (df["DTE"] <= dte_max)].copy()
    return df


def candidate_pairs(df: pd.DataFrame, min_strikes: int, max_strikes: int | None, n_grid: int) -> list[tuple[PairCandidate, SliceArrays, SliceArrays]]:
    slices: dict[tuple[str, int, int], SliceArrays] = {}
    for key, slc in df.groupby(["currency", "date", "expiration_date"], sort=False):
        arr = prepare_slice(slc, min_strikes=min_strikes, max_strikes=max_strikes)
        if arr is not None:
            slices[(str(key[0]), int(key[1]), int(key[2]))] = arr
    out = []
    by_date: dict[tuple[str, str, int], list[SliceArrays]] = {}
    for arr in slices.values():
        month = pd.to_datetime(arr.date, unit="s", utc=True).strftime("%Y-%m")
        by_date.setdefault((arr.currency, month, arr.date), []).append(arr)
    temp = []
    for (currency, month, date), arrs in by_date.items():
        arrs = sorted(arrs, key=lambda z: z.expiration)
        for short, long in zip(arrs[:-1], arrs[1:]):
            grid = calendar_grid(short, long, n_grid)
            if len(grid) < max(2, min(n_grid, 3)):
                continue
            score = max(short.danger_p90, long.danger_p90)
            temp.append((currency, month, date, score, short.expiration, long.expiration, short, long))
    temp = sorted(temp, key=lambda z: (z[0], z[1], -z[3], z[2], z[4]))
    ranks: dict[tuple[str, str], int] = {}
    for currency, month, date, score, short_exp, long_exp, short, long in temp:
        cm = (currency, month)
        ranks[cm] = ranks.get(cm, 0) + 1
        cand = PairCandidate(currency, month, date, short_exp, long_exp, float(score), ranks[cm])
        out.append((cand, short, long))
    return out


def run_calendar_pilot(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = load_panel(args.panel, args.currencies, args.months, args.dte_min, args.dte_max)
    cands = candidate_pairs(panel, args.min_strikes, args.max_strikes, args.calendar_grid)
    rows = []
    curv_rows = []
    used: dict[tuple[str, str], int] = {}
    for cand, short, long in cands:
        cm = (cand.currency, cand.month)
        if used.get(cm, 0) >= args.pairs_per_month:
            continue
        solved = {}
        for add_calendar in [False, True]:
            label = "calendar" if add_calendar else "strike_only"
            solved[(label, "L2")] = solve_pair_qp(short, long, rho=0.0, add_calendar=add_calendar, n_grid=args.calendar_grid, maxiter=args.maxiter)
            solved[(label, "SMOOTH10")] = solve_pair_qp(short, long, rho=args.rho, add_calendar=add_calendar, n_grid=args.calendar_grid, maxiter=args.maxiter)
        if not (solved[("calendar", "L2")]["success"] and solved[("calendar", "SMOOTH10")]["success"]):
            rows.append(
                {
                    "currency": cand.currency,
                    "month": cand.month,
                    "date": cand.date,
                    "candidate_rank": cand.rank,
                    "selected": False,
                    "calendar_l2_status": solved[("calendar", "L2")]["status"],
                    "calendar_smooth_status": solved[("calendar", "SMOOTH10")]["status"],
                }
            )
            continue
        used[cm] = used.get(cm, 0) + 1
        n1 = len(short.x)
        for label in ["strike_only", "calendar"]:
            l2 = solved[(label, "L2")]
            sm = solved[(label, "SMOOTH10")]
            if not (l2["success"] and sm["success"]):
                continue
            x_l2 = l2["x"]
            x_sm = sm["x"]
            curv = curvature_summary(short, long, x_l2, x_sm)
            l2_cal_v = max_calendar_violation(short, long, x_l2[:n1], x_l2[n1:], args.calendar_grid)
            sm_cal_v = max_calendar_violation(short, long, x_sm[:n1], x_sm[n1:], args.calendar_grid)
            rows.append(
                {
                    "currency": cand.currency,
                    "month": cand.month,
                    "date": cand.date,
                    "short_expiration": cand.short_expiration,
                    "long_expiration": cand.long_expiration,
                    "short_dte": short.dte,
                    "long_dte": long.dte,
                    "candidate_rank": cand.rank,
                    "selected": True,
                    "constraint_set": label,
                    "pair_score": cand.pair_score,
                    "n_short": len(short.x),
                    "n_long": len(long.x),
                    "calendar_grid_n": int(l2["calendar_grid_n"]),
                    "l2_status": l2["status"],
                    "smooth_status": sm["status"],
                    "l2_method": l2.get("method"),
                    "smooth_method": sm.get("method"),
                    "l2_calendar_violation": l2_cal_v,
                    "smooth_calendar_violation": sm_cal_v,
                    "l2_objective": l2.get("objective", np.nan),
                    "smooth_objective": sm.get("objective", np.nan),
                    **curv,
                }
            )
            curv_rows.append(
                {
                    "currency": cand.currency,
                    "month": cand.month,
                    "date": cand.date,
                    "constraint_set": label,
                    "p90_scaled_curvature_diff": curv["p90_scaled_curvature_diff"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(curv_rows)


def write_summary(outdir: Path, results: pd.DataFrame) -> None:
    lines = [
        "# Calendar-certified robustness pilot",
        "",
        "This isolated pilot adds forward-moneyness calendar inequalities to a joint two-maturity repair.",
        "It is a robustness screen for the paper's normalized-call convention, not a full production calendar-arbitrage engine.",
        "",
    ]
    selected = results[results.get("selected", False) == True].copy() if len(results) else pd.DataFrame()
    if selected.empty:
        lines.append("No selected calendar-feasible pairs were solved.")
    else:
        piv = selected.pivot_table(
            index=["currency", "month", "date", "candidate_rank"],
            columns="constraint_set",
            values=["p90_scaled_curvature_diff", "p90_abs_curvature_diff"],
            aggfunc="first",
        ).reset_index()
        piv.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip("_")
            if isinstance(col, tuple)
            else str(col)
            for col in piv.columns
        ]
        if {"p90_scaled_curvature_diff_strike_only", "p90_scaled_curvature_diff_calendar"}.issubset(piv.columns):
            piv["calendar_to_strike_ratio"] = (
                piv["p90_scaled_curvature_diff_calendar"]
                / piv["p90_scaled_curvature_diff_strike_only"]
            )
        lines.append(f"Selected calendar-feasible pairs: {len(piv)}")
        lines.append("")
        lines.append("| Currency | Month | Rank | Strike-only p90 scaled curvature | Calendar p90 scaled curvature | Ratio |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, row in piv.iterrows():
            ratio = row.get("calendar_to_strike_ratio", np.nan)
            lines.append(
                f"| {row['currency']} | {row['month']} | {int(row['candidate_rank'])} | "
                f"{row.get('p90_scaled_curvature_diff_strike_only', np.nan):.4g} | "
                f"{row.get('p90_scaled_curvature_diff_calendar', np.nan):.4g} | {ratio:.3f} |"
            )
        lines.append("")
        if "calendar_to_strike_ratio" in piv.columns:
            lines.append(
                f"Median calendar/strike ratio: {float(np.nanmedian(piv['calendar_to_strike_ratio'])):.3f}."
            )
            q10, q90 = np.nanquantile(piv["calendar_to_strike_ratio"], [0.10, 0.90])
            lines.append(f"10--90% ratio range: {q10:.3f} to {q90:.3f}.")
        if {"p90_abs_curvature_diff_strike_only", "p90_abs_curvature_diff_calendar"}.issubset(piv.columns):
            abs_ratio = piv["p90_abs_curvature_diff_calendar"] / piv["p90_abs_curvature_diff_strike_only"]
            lines.append(f"Median absolute-curvature ratio: {float(np.nanmedian(abs_ratio)):.3f}.")
    (outdir / "CALENDAR_CERTIFIED_ROBUSTNESS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_self_checks() -> None:
    x = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    base = pd.DataFrame(
        {
            "currency": "TST",
            "date": 1700000000,
            "strike": x * 100.0,
            "forward_price": 100.0,
            "DTE": 7.0,
            "C_bid": [0.205, 0.125, 0.060, 0.025, 0.010],
            "C_ask": [0.245, 0.165, 0.100, 0.065, 0.050],
            "C_mid": [0.225, 0.145, 0.080, 0.045, 0.030],
            "intrinsic": np.maximum(1.0 - x, 0.0),
            "relspread": 0.02,
            "synthetic_valid": True,
        }
    )
    short_df = base.assign(expiration_date=1700600000)
    long_df = base.assign(
        expiration_date=1701200000,
        DTE=14.0,
        C_bid=[0.210, 0.130, 0.065, 0.030, 0.015],
        C_ask=[0.250, 0.170, 0.105, 0.070, 0.055],
        C_mid=[0.230, 0.150, 0.085, 0.050, 0.035],
    )
    short = prepare_slice(short_df, min_strikes=5, max_strikes=None)
    long = prepare_slice(long_df, min_strikes=5, max_strikes=None)
    if short is None or long is None:
        raise AssertionError("Synthetic slices did not prepare.")
    strike = solve_pair_qp(short, long, rho=0.0, add_calendar=False, n_grid=5, maxiter=500)
    cal = solve_pair_qp(short, long, rho=0.0, add_calendar=True, n_grid=5, maxiter=500)
    if not (strike["success"] and cal["success"]):
        raise AssertionError(f"Synthetic QPs failed: {strike['status']} / {cal['status']}")
    n1 = len(short.x)
    viol = max_calendar_violation(short, long, cal["x"][:n1], cal["x"][n1:], 5)
    if viol > 1e-6:
        raise AssertionError(f"Calendar self-check violation too large: {viol}")
    curv = curvature_summary(short, long, strike["x"], cal["x"])
    if curv["n_curvature_rows"] <= 0:
        raise AssertionError("Curvature self-check produced no rows.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--currencies", nargs="+", default=["BTC", "ETH"])
    parser.add_argument("--months", nargs="+", default=["2022-11", "2023-03", "2023-06"])
    parser.add_argument("--dte-min", type=float, default=2.0)
    parser.add_argument("--dte-max", type=float, default=30.0)
    parser.add_argument("--min-strikes", type=int, default=8)
    parser.add_argument("--max-strikes", type=int, default=45)
    parser.add_argument("--calendar-grid", type=int, default=9)
    parser.add_argument("--rho", type=float, default=10.0)
    parser.add_argument("--pairs-per-month", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--self-check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_self_checks()
    if args.self_check_only:
        print("Self-checks passed.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, curv_rows = run_calendar_pilot(args)
    results.to_csv(args.output_dir / "calendar_certified_pair_results.csv", index=False)
    curv_rows.to_csv(args.output_dir / "calendar_curvature_rows.csv", index=False)
    write_summary(args.output_dir, results)
    print(f"Wrote calendar robustness outputs to {args.output_dir}")
    print(f"Rows: {len(results)}; selected rows: {int(results.get('selected', pd.Series(dtype=bool)).sum()) if len(results) else 0}")


if __name__ == "__main__":
    main()
