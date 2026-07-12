"""Shared single-maturity normalized-call strike constraints.

All screening, repair, verification, and LP-range code in this experiment uses
this module. Rows are expressed as ``A_ub @ c <= b_ub`` for a call-price vector
ordered by strictly increasing normalized strike ``x = K / F``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, linprog


def _as_vector(name: str, values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).reshape(-1)
    if out.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains nonfinite values")
    return out


def validate_normalized_strikes(x: np.ndarray) -> np.ndarray:
    x = _as_vector("x", x)
    if x.size < 3:
        raise ValueError("at least three ordered strikes are required")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("normalized strikes must be strictly increasing")
    return x


def effective_bounds(
    x: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    intrinsic: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect quotes with normalized-call intrinsic and unit upper bounds."""
    x = _as_vector("x", x)
    bid = _as_vector("bid", bid)
    ask = _as_vector("ask", ask)
    if not (x.size == bid.size == ask.size):
        raise ValueError("x, bid, and ask must have the same length")
    if intrinsic is None:
        intrinsic_arr = np.maximum(1.0 - x, 0.0)
    else:
        intrinsic_arr = _as_vector("intrinsic", intrinsic)
        if intrinsic_arr.size != x.size:
            raise ValueError("intrinsic must have the same length as x")
        intrinsic_arr = np.maximum(intrinsic_arr, np.maximum(1.0 - x, 0.0))
    lower = np.maximum(bid, intrinsic_arr)
    upper = np.minimum(ask, 1.0)
    return lower, upper


def build_full_constraint_matrix(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return monotonicity, vertical-spread, and convexity inequality rows."""
    x = validate_normalized_strikes(x)
    n = x.size
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    families: list[str] = []

    for i in range(n - 1):
        row = np.zeros(n)
        row[i] = -1.0
        row[i + 1] = 1.0
        rows.append(row)
        rhs.append(0.0)
        families.append("monotonicity")

    for i, dx in enumerate(np.diff(x)):
        row = np.zeros(n)
        row[i] = 1.0
        row[i + 1] = -1.0
        rows.append(row)
        rhs.append(float(dx))
        families.append("vertical_spread")

    for i in range(1, n - 1):
        h_minus = x[i] - x[i - 1]
        h_plus = x[i + 1] - x[i]
        row = np.zeros(n)
        row[i - 1] = -1.0 / h_minus
        row[i] = 1.0 / h_minus + 1.0 / h_plus
        row[i + 1] = -1.0 / h_plus
        rows.append(row)
        rhs.append(0.0)
        families.append("convexity")

    return np.vstack(rows), np.asarray(rhs, dtype=float), families


def scipy_linear_constraint(x: np.ndarray) -> LinearConstraint:
    A_ub, b_ub, _ = build_full_constraint_matrix(x)
    return LinearConstraint(A_ub, -np.inf, b_ub)


def certify_call_curve(
    c: np.ndarray,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float = 2e-7,
) -> dict[str, Any]:
    """Report maximum violations for every constraint family."""
    c = _as_vector("c", c)
    x = validate_normalized_strikes(x)
    lower = _as_vector("lower", lower)
    upper = _as_vector("upper", upper)
    if not (c.size == x.size == lower.size == upper.size):
        raise ValueError("c, x, lower, and upper must have the same length")

    dx = np.diff(x)
    slopes = (c[:-1] - c[1:]) / dx
    derivative = np.diff(c) / dx
    box_lower = float(np.max(np.maximum(lower - c, 0.0)))
    box_upper = float(np.max(np.maximum(c - upper, 0.0)))
    monotonicity = float(np.max(np.maximum(c[1:] - c[:-1], 0.0)))
    vertical = float(np.max(np.maximum(slopes - 1.0, 0.0)))
    convexity = float(np.max(np.maximum(derivative[:-1] - derivative[1:], 0.0)))
    max_violation = max(box_lower, box_upper, monotonicity, vertical, convexity)

    return {
        "certified_feasible": bool(max_violation <= tolerance),
        "max_constraint_violation": float(max_violation),
        "max_box_lower_violation": box_lower,
        "max_box_upper_violation": box_upper,
        "max_monotonicity_violation": monotonicity,
        "max_vertical_spread_violation": vertical,
        "max_convexity_violation": convexity,
        "max_vertical_slope": float(np.max(slopes)),
        "min_vertical_slope": float(np.min(slopes)),
        "certification_tolerance": float(tolerance),
    }


def feasibility_certificate(
    x: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    intrinsic: np.ndarray | None = None,
    solver_tolerance: float = 1e-9,
    certification_tolerance: float = 2e-7,
) -> dict[str, Any]:
    """Solve a zero-objective LP and certify the returned feasible curve."""
    try:
        x_arr = validate_normalized_strikes(x)
        lower, upper = effective_bounds(x_arr, bid, ask, intrinsic)
    except ValueError as exc:
        return {
            "feasible": False,
            "solver_status": "not_run",
            "failure_type": "invalid_input",
            "message": str(exc),
            "max_constraint_violation": float("inf"),
            "solution": None,
        }

    empty = lower > upper + certification_tolerance
    if np.any(empty):
        return {
            "feasible": False,
            "solver_status": "not_run",
            "failure_type": "empty_effective_box",
            "message": f"{int(np.sum(empty))} lower bounds exceed upper bounds",
            "max_constraint_violation": float(np.max(lower - upper)),
            "solution": None,
        }

    A_ub, b_ub, _ = build_full_constraint_matrix(x_arr)
    result = linprog(
        c=np.zeros(x_arr.size),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=list(zip(lower, upper)),
        method="highs",
        options={
            "primal_feasibility_tolerance": solver_tolerance,
            "dual_feasibility_tolerance": solver_tolerance,
        },
    )
    if not result.success or result.x is None:
        return {
            "feasible": False,
            "solver_status": str(result.message),
            "failure_type": "lp_infeasible" if result.status == 2 else "solver_failure",
            "message": str(result.message),
            "max_constraint_violation": float("inf"),
            "solution": None,
        }

    report = certify_call_curve(
        c=result.x,
        x=x_arr,
        lower=lower,
        upper=upper,
        tolerance=certification_tolerance,
    )
    return {
        **report,
        "feasible": bool(report["certified_feasible"]),
        "solver_status": "optimal",
        "failure_type": "" if report["certified_feasible"] else "numerical_certification",
        "message": str(result.message),
        "solution": np.asarray(result.x, dtype=float),
    }
