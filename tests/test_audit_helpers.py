import numpy as np
import pytest

from option_repair_risk.audit_helpers import (
    classify_dte_bucket,
    convexity_violations,
    elastic_monotone_convex_slack,
    monotonicity_violations,
)


def test_classify_dte_bucket_matches_phase1_bins():
    assert classify_dte_bucket(1.0) == "0-1"
    assert classify_dte_bucket(2.0) == "2-7"
    assert classify_dte_bucket(14.0) == "8-14"
    assert classify_dte_bucket(30.0) == "15-30"
    assert classify_dte_bucket(45.0) == "31-45"
    assert classify_dte_bucket(60.0) == "46+"


def test_monotonicity_violations_are_zero_for_decreasing_calls():
    calls = np.array([0.4, 0.3, 0.1])
    assert monotonicity_violations(calls).sum() == pytest.approx(0.0)


def test_monotonicity_violations_detect_increasing_call_step():
    calls = np.array([0.4, 0.45, 0.1])
    assert monotonicity_violations(calls).tolist() == pytest.approx([0.05, 0.0])


def test_convexity_violations_are_zero_for_convex_curve():
    strikes = np.array([90.0, 100.0, 110.0])
    calls = np.array([0.3, 0.2, 0.15])
    assert convexity_violations(strikes, calls).sum() == pytest.approx(0.0)


def test_convexity_violations_detect_slope_decrease():
    strikes = np.array([90.0, 100.0, 110.0])
    calls = np.array([0.3, 0.25, 0.05])
    assert convexity_violations(strikes, calls).sum() > 0


def test_elastic_slack_zero_for_feasible_box():
    strikes = np.array([90.0, 100.0, 110.0])
    bid = np.array([0.29, 0.19, 0.14])
    ask = np.array([0.31, 0.21, 0.16])
    intrinsic = np.array([0.2, 0.1, 0.0])
    result = elastic_monotone_convex_slack(strikes, bid, ask, intrinsic)

    assert result.strict_feasible
    assert result.elastic_slack == pytest.approx(0.0, abs=1e-9)


def test_elastic_slack_positive_for_infeasible_monotone_box():
    strikes = np.array([90.0, 100.0, 110.0])
    bid = np.array([0.10, 0.30, 0.05])
    ask = np.array([0.11, 0.31, 0.06])
    intrinsic = np.array([0.0, 0.0, 0.0])
    result = elastic_monotone_convex_slack(strikes, bid, ask, intrinsic)

    assert not result.strict_feasible
    assert result.elastic_slack > 0
