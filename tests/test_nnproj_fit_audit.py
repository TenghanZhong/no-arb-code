import numpy as np
import pytest

from option_repair_risk.nnproj_fit_audit import (
    error_metrics,
    outside_box_metrics,
    regression_metrics,
    safe_qlike,
)


def test_regression_metrics_match_manual_values():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 4.0])

    metrics = regression_metrics(y, pred)

    assert metrics["n"] == 3
    assert metrics["rmse"] == pytest.approx(np.sqrt(1.0 / 3.0))
    assert metrics["mae"] == pytest.approx(1.0 / 3.0)
    assert metrics["r2"] == pytest.approx(0.5)


def test_error_metrics_scale_by_floor():
    actual = np.array([0.10, 0.20])
    pred = np.array([0.11, 0.18])
    scale = np.array([0.01, 0.00])

    metrics = error_metrics(actual, pred, scale, floor=0.01)

    assert metrics["rmse"] == pytest.approx(np.sqrt((0.01**2 + 0.02**2) / 2.0))
    assert metrics["mae_scaled"] == pytest.approx((1.0 + 2.0) / 2.0)
    assert metrics["p90_abs_scaled"] == pytest.approx(np.percentile([1.0, 2.0], 90))


def test_outside_box_metrics_count_exact_bid_ask_violations():
    pred = np.array([0.09, 0.11, 0.13, np.nan, 0.14])
    bid = np.array([0.10, 0.10, 0.10, 0.10, np.nan])
    ask = np.array([0.12, 0.12, 0.12, 0.12, 0.15])

    metrics = outside_box_metrics(pred, bid, ask)

    assert metrics["n"] == 3
    assert metrics["outside_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["below_bid_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["above_ask_rate"] == pytest.approx(1.0 / 3.0)


def test_safe_qlike_is_finite_for_positive_variances():
    actual = np.array([0.02, 0.03])
    pred = np.array([0.02, 0.04])

    assert np.isfinite(safe_qlike(actual, pred))
