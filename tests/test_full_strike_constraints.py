from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


ANALYSIS = Path(__file__).resolve().parents[1] / "code" / "analysis"
COMMON = ANALYSIS / "common"
sys.path.insert(0, str(COMMON))

from full_strike_constraints import (  # noqa: E402
    build_full_constraint_matrix,
    certify_call_curve,
    effective_bounds,
    feasibility_certificate,
)


def test_effective_bounds_intersect_quotes_with_normalized_call_bounds() -> None:
    x = np.array([0.80, 1.00, 1.20])
    bid = np.array([0.10, -0.01, 0.02])
    ask = np.array([1.20, 0.50, 0.40])

    lower, upper = effective_bounds(x=x, bid=bid, ask=ask)

    np.testing.assert_allclose(lower, [0.20, 0.00, 0.02])
    np.testing.assert_allclose(upper, [1.00, 0.50, 0.40])


def test_certifier_rejects_old_feasible_but_vertical_spread_invalid_curve() -> None:
    x = np.array([0.90, 1.00, 1.10])
    c = np.array([0.50, 0.10, 0.00])
    lower = np.zeros(3)
    upper = np.ones(3)

    report = certify_call_curve(c=c, x=x, lower=lower, upper=upper, tolerance=1e-10)

    assert report["max_monotonicity_violation"] == 0.0
    assert report["max_convexity_violation"] == 0.0
    assert report["max_vertical_spread_violation"] > 2.9
    assert report["certified_feasible"] is False


def test_shared_matrix_enforces_nonuniform_vertical_spread_and_convexity() -> None:
    x = np.array([0.80, 0.90, 1.15, 1.40])
    A_ub, b_ub, families = build_full_constraint_matrix(x)

    assert A_ub.shape == (3 * (len(x) - 1) - 1, len(x))
    assert families.count("monotonicity") == len(x) - 1
    assert families.count("vertical_spread") == len(x) - 1
    assert families.count("convexity") == len(x) - 2

    valid = np.array([0.30, 0.22, 0.08, 0.00])
    assert np.max(A_ub @ valid - b_ub) <= 1e-12

    invalid = np.array([0.50, 0.10, 0.00, 0.00])
    residuals = A_ub @ invalid - b_ub
    vertical_rows = np.array(families) == "vertical_spread"
    assert np.max(residuals[vertical_rows]) > 0.0


def test_feasibility_certificate_matches_direct_linear_program() -> None:
    x = np.array([0.75, 0.90, 1.05, 1.25])
    bid = np.array([0.26, 0.17, 0.08, 0.01])
    ask = np.array([0.31, 0.22, 0.12, 0.04])
    lower, upper = effective_bounds(x=x, bid=bid, ask=ask)
    A_ub, b_ub, _ = build_full_constraint_matrix(x)

    direct = linprog(
        c=np.zeros(len(x)),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=list(zip(lower, upper)),
        method="highs",
    )
    certificate = feasibility_certificate(x=x, bid=bid, ask=ask)

    assert direct.success
    assert certificate["feasible"] is True
    assert certificate["solver_status"] == "optimal"
    assert certificate["max_constraint_violation"] <= 1e-9


def test_infeasible_effective_box_is_reported_before_solver() -> None:
    x = np.array([0.80, 1.00, 1.20])
    bid = np.array([0.50, 0.20, 0.00])
    ask = np.array([0.10, 0.30, 0.10])

    certificate = feasibility_certificate(x=x, bid=bid, ask=ask)

    assert certificate["feasible"] is False
    assert certificate["failure_type"] == "empty_effective_box"
