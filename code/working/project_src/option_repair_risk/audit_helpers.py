from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


EPS = 1e-12


@dataclass(frozen=True)
class FeasibilityResult:
    strict_feasible: bool
    elastic_feasible: bool
    elastic_slack: float
    solver_status: str


def classify_dte_bucket(dte: float) -> str:
    if dte < 2:
        return "0-1"
    if dte <= 7:
        return "2-7"
    if dte <= 14:
        return "8-14"
    if dte <= 30:
        return "15-30"
    if dte <= 45:
        return "31-45"
    return "46+"


def monotonicity_violations(call_prices: np.ndarray) -> np.ndarray:
    calls = np.asarray(call_prices, dtype=float)
    if calls.size < 2:
        return np.empty(0, dtype=float)
    return np.maximum(0.0, calls[1:] - calls[:-1])


def convexity_violations(strikes: np.ndarray, call_prices: np.ndarray) -> np.ndarray:
    k = np.asarray(strikes, dtype=float)
    c = np.asarray(call_prices, dtype=float)
    if c.size < 3:
        return np.empty(0, dtype=float)
    left_width = k[1:-1] - k[:-2]
    right_width = k[2:] - k[1:-1]
    valid = (left_width > 0) & (right_width > 0)
    out = np.zeros(c.size - 2, dtype=float)
    left_slope = (c[1:-1] - c[:-2]) / np.maximum(left_width, EPS)
    right_slope = (c[2:] - c[1:-1]) / np.maximum(right_width, EPS)
    out[valid] = np.maximum(0.0, left_slope[valid] - right_slope[valid])
    return out


def _shape_constraints(strikes: np.ndarray, n_vars: int, offset: int = 0):
    k = np.asarray(strikes, dtype=float)
    a_rows: list[np.ndarray] = []
    b_rows: list[float] = []

    # Monotonicity: C_i >= C_{i+1}; encoded as -C_i + C_{i+1} <= 0.
    for i in range(len(k) - 1):
        row = np.zeros(n_vars)
        row[offset + i] = -1.0
        row[offset + i + 1] = 1.0
        a_rows.append(row)
        b_rows.append(0.0)

    # Convexity on irregular grid: left slope <= right slope.
    for i in range(1, len(k) - 1):
        left = k[i] - k[i - 1]
        right = k[i + 1] - k[i]
        if left <= 0 or right <= 0:
            continue
        row = np.zeros(n_vars)
        row[offset + i - 1] = -1.0 / left
        row[offset + i] = 1.0 / left + 1.0 / right
        row[offset + i + 1] = -1.0 / right
        a_rows.append(row)
        b_rows.append(0.0)

    return a_rows, b_rows


def _strict_feasible(strikes: np.ndarray, bid: np.ndarray, ask: np.ndarray, intrinsic: np.ndarray):
    lower = np.maximum(bid, intrinsic)
    upper = np.minimum(ask, 1.0)
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower > upper):
        return False, "invalid_bounds"
    n = len(strikes)
    if n == 0:
        return False, "empty_slice"
    a_ub, b_ub = _shape_constraints(strikes, n)
    res = linprog(
        c=np.zeros(n),
        A_ub=np.vstack(a_ub) if a_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        bounds=list(zip(lower, upper)),
        method="highs",
    )
    return bool(res.success), str(res.message)


def elastic_monotone_convex_slack(
    strikes: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    intrinsic: np.ndarray,
    *,
    eps: float = 1e-10,
) -> FeasibilityResult:
    k = np.asarray(strikes, dtype=float)
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    intrinsic = np.asarray(intrinsic, dtype=float)
    order = np.argsort(k)
    k, bid, ask, intrinsic = k[order], bid[order], ask[order], intrinsic[order]
    valid = np.isfinite(k) & np.isfinite(bid) & np.isfinite(ask) & np.isfinite(intrinsic)
    k, bid, ask, intrinsic = k[valid], bid[valid], ask[valid], intrinsic[valid]

    strict, strict_status = _strict_feasible(k, bid, ask, intrinsic)
    if strict:
        return FeasibilityResult(True, True, 0.0, strict_status)

    n = len(k)
    if n == 0:
        return FeasibilityResult(False, False, float("nan"), "empty_slice")

    # Variables are [C_0..C_n-1, lower_slack_0.., upper_slack_0..].
    n_vars = 3 * n
    spreads = np.maximum(ask - bid, eps)
    c = np.zeros(n_vars)
    c[n : 2 * n] = 1.0 / spreads
    c[2 * n : 3 * n] = 1.0 / spreads

    a_rows, b_rows = _shape_constraints(k, n_vars)

    for i in range(n):
        # bid_i - lower_slack_i <= C_i -> -C_i - lower_slack_i <= -bid_i
        row = np.zeros(n_vars)
        row[i] = -1.0
        row[n + i] = -1.0
        a_rows.append(row)
        b_rows.append(-bid[i])

        # C_i <= ask_i + upper_slack_i -> C_i - upper_slack_i <= ask_i
        row = np.zeros(n_vars)
        row[i] = 1.0
        row[2 * n + i] = -1.0
        a_rows.append(row)
        b_rows.append(ask[i])

    bounds = [(float(max(intrinsic[i], 0.0)), 1.0) for i in range(n)]
    bounds.extend([(0.0, None)] * (2 * n))
    res = linprog(
        c=c,
        A_ub=np.vstack(a_rows) if a_rows else None,
        b_ub=np.array(b_rows) if b_rows else None,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return FeasibilityResult(False, False, float("nan"), str(res.message))
    slack = float(np.dot(c[n:], res.x[n:]))
    return FeasibilityResult(False, True, slack, str(res.message))


def self_check_audit_helpers() -> None:
    assert classify_dte_bucket(2.0) == "2-7"
    assert monotonicity_violations(np.array([0.2, 0.1])).sum() == 0
    assert monotonicity_violations(np.array([0.2, 0.3])).sum() > 0
    result = elastic_monotone_convex_slack(
        np.array([90.0, 100.0, 110.0]),
        np.array([0.29, 0.19, 0.14]),
        np.array([0.31, 0.21, 0.16]),
        np.array([0.2, 0.1, 0.0]),
    )
    assert result.strict_feasible and result.elastic_slack == 0
