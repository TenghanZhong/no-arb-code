from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = ROOT / "code" / "analysis" / "summarize_full_rerun.py"
PLOT_SCRIPT = ROOT / "code" / "analysis" / "diagnostics" / "plot_artifact_figures.py"


def load_summary_module():
    name = "test_summarize_full_rerun"
    spec = importlib.util.spec_from_file_location(name, SUMMARY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_plot_module():
    name = "test_plot_artifact_figures"
    spec = importlib.util.spec_from_file_location(name, PLOT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_relative_difference_figure_uses_only_the_classical_source() -> None:
    frame = pd.DataFrame(
        [
            {
                "source": source,
                "comparison": "L2_vs_SMOOTH_RHO_10",
                "loss_type": "common",
                "selection_regime": "top",
                "book": "ATM_STRADDLE",
                "p90_omega": value,
            }
            for source, value in (
                ("classical_additional_clusters", 0.1),
                ("nn_targeted", 0.2),
            )
        ]
    )

    selected = load_plot_module().var_relative_difference_plot_data(frame)

    assert len(selected) == 1
    assert selected.iloc[0]["source"] == "classical_additional_clusters"


def test_projection_success_is_scoped_to_requested_arm(tmp_path: Path) -> None:
    run = tmp_path / "run_1"
    run.mkdir()
    pd.DataFrame(
        [
            {
                "arm": "L1",
                "success": True,
                "status": "optimal",
                "message": "highs",
                "verified_feasible": True,
                "max_constraint_violation": 0.0,
                "max_vertical_slope": 0.9,
            },
            {
                "arm": "SVI",
                "success": True,
                "status": "optimal",
                "message": "clarabel",
                "verified_feasible": True,
                "max_constraint_violation": 1e-8,
                "max_vertical_slope": 0.95,
            },
            {
                "arm": "SVI",
                "success": False,
                "status": "cvxpy_failed",
                "message": "numerical certification",
                "verified_feasible": False,
                "max_constraint_violation": 3e-6,
                "max_vertical_slope": 1.000003,
            },
            {
                "arm": "SVI",
                "success": False,
                "status": "optimal",
                "message": "clarabel",
                "verified_feasible": False,
                "max_constraint_violation": 4e-6,
                "max_vertical_slope": 0.99,
            },
        ]
    ).to_csv(run / "repair_diagnostics.csv", index=False)

    summary = load_summary_module().projection_success(tmp_path, "SVI")

    assert summary["arm"] == "SVI"
    assert summary["attempts"] == 3
    assert summary["successes"] == 1
    assert summary["success_rate"] == 1 / 3
    assert summary["failure_status_counts"] == {"cvxpy_failed": 1, "optimal": 1}
    assert summary["failure_message_counts"] == {"numerical certification": 1, "clarabel": 1}
    assert summary["failure_reason_counts"] == {
        "solver_or_other_failure": 1,
        "postsolve_constraint_certification": 1,
    }
    assert summary["max_constraint_violation"] == 4e-6


def test_targeted_heldout_summary_uses_successful_audits(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "run": "run_a",
                "audit_status": "ok",
                "oos_log_tv_r2": 0.8,
                "oos_price_box_outside_rate": 0.4,
                "oos_price_mae_scaled": 1.5,
            },
            {
                "run": "run_b",
                "audit_status": "ok",
                "oos_log_tv_r2": 0.9,
                "oos_price_box_outside_rate": 0.5,
                "oos_price_mae_scaled": 2.5,
            },
            {
                "run": "run_c",
                "audit_status": "error",
                "oos_log_tv_r2": 99.0,
                "oos_price_box_outside_rate": 99.0,
                "oos_price_mae_scaled": 99.0,
            },
        ]
    ).to_csv(tmp_path / "nnproj_heldout_fit_by_run.csv", index=False)

    summary = load_summary_module().targeted_heldout_summary(tmp_path)

    assert summary["attempts"] == 3
    assert summary["successes"] == 2
    assert summary["status_counts"] == {"ok": 2, "error": 1}
    assert summary["median_oos_log_total_variance_r2"] == pytest.approx(0.85)
    assert summary["median_raw_price_outside_box_share"] == pytest.approx(0.45)
    assert summary["median_spread_scaled_price_mae"] == pytest.approx(2.0)


def test_lp_range_audit_reports_residuals_statuses_and_exceedance(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "range_status": "ok",
                "cxx_diff": 1.0,
                "feasible_cxx_range": 1.0,
                "lp_min_max_constraint_violation": 1e-10,
                "lp_max_max_constraint_violation": 2e-10,
            },
            {
                "range_status": "ok",
                "cxx_diff": 1.00002,
                "feasible_cxx_range": 1.0,
                "lp_min_max_constraint_violation": 3e-10,
                "lp_max_max_constraint_violation": 4e-10,
            },
            {
                "range_status": "solver_failure",
                "cxx_diff": 0.5,
                "feasible_cxx_range": float("nan"),
                "lp_min_max_constraint_violation": float("nan"),
                "lp_max_max_constraint_violation": float("nan"),
            },
        ]
    ).to_csv(tmp_path / "feasible_curvature_range_merged_pairwise.csv", index=False)
    tolerance_audit = {
        "audit_pass": True,
        "initial_exceedances": 1,
        "strict_tolerance_exceedances_after_refinement": 0,
        "audit_tolerance_exceedances": 0,
    }
    (tmp_path / "lp_range_tolerance_audit.json").write_text(
        json.dumps(tolerance_audit), encoding="utf-8"
    )

    summary = load_summary_module().lp_range_audit(tmp_path)

    assert summary["rows"] == 3
    assert summary["successful_rows"] == 2
    assert summary["status_counts"] == {"ok": 2, "solver_failure": 1}
    assert summary["range_exceedances"] == 1
    assert summary["max_raw_exceedance"] == pytest.approx(2e-5)
    assert summary["max_lp_constraint_violation"] == pytest.approx(4e-10)
    assert summary["lp_tolerance_audit"] == tolerance_audit


def test_hedge_notional_anchor_reads_persistence_units_without_rescaling(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "n_runs_total": 108,
                "n_runs_coverage_pass": 45,
                "n_pairs_coverage_pass": 1820,
                "median_run_p90_abs_gap_bps": 18.92,
                "median_run_p90_abs_gap_usd_per_1mm": 1892.0,
            }
        ]
    ).to_csv(tmp_path / "risk_reversal_hedge_persistence_summary.csv", index=False)

    summary = load_summary_module().hedge_notional_anchor(tmp_path)

    assert summary == {
        "runs_total": 108,
        "coverage_qualified_runs": 45,
        "coverage_qualified_pairs": 1820,
        "median_run_p90_hedge_notional_bps": pytest.approx(18.92),
        "usd_per_1m_gross_notional": pytest.approx(1892.0),
    }


def test_surface_mechanism_summary_preserves_pair_level_diagnostics(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "source": "classical",
                "pair": "L1__L2",
                "median_price_gap_abs": 0.001,
                "p90_price_gap_abs": 0.004,
                "median_d2_amplification_ratio": 0.2,
                "p90_d2_amplification_ratio": 0.7,
                "median_model_free_curvature_gamma_gap": 3e-6,
                "spearman_d2_vs_black_gamma": 0.8,
            },
            {
                "source": "classical",
                "pair": "L2__SMOOTH_RHO_10",
                "median_price_gap_abs": 0.002,
                "p90_price_gap_abs": 0.008,
                "median_d2_amplification_ratio": 0.3,
                "p90_d2_amplification_ratio": 0.9,
                "median_model_free_curvature_gamma_gap": 7e-6,
                "spearman_d2_vs_black_gamma": 0.9,
            },
        ]
    ).to_csv(tmp_path / "surface_mechanism_summary.csv", index=False)
    (tmp_path / "surface_mechanism_audit.json").write_text(
        '{"rows": 100, "evaluation_pair_rows": 20, "bound_exceedances": 0, "max_amplification_ratio": 0.99}',
        encoding="utf-8",
    )

    summary = load_summary_module().surface_mechanism_summary(tmp_path)

    assert summary["audit"]["bound_exceedances"] == 0
    assert summary["audit"]["max_amplification_ratio"] == pytest.approx(0.99)
    assert summary["pairs"]["classical:L2__SMOOTH_RHO_10"]["p90_price_gap_abs"] == pytest.approx(0.008)
    assert summary["pairs"]["classical:L2__SMOOTH_RHO_10"]["spearman_d2_vs_black_gamma"] == pytest.approx(0.9)


def test_json_sanitize_converts_nonfinite_values_to_null() -> None:
    sanitized = load_summary_module().json_sanitize(
        {"finite": 1.2, "nan": float("nan"), "inf": float("inf"), "nested": [float("-inf"), 3]}
    )

    assert sanitized == {"finite": 1.2, "nan": None, "inf": None, "nested": [None, 3]}


def test_calendar_screen_summary_matches_pre_and_post_rows(tmp_path: Path) -> None:
    rows = []
    for currency, before, after in [("BTC", 2.0, 2.2), ("ETH", 4.0, 3.6)]:
        for constraint_set, value in [("strike_only", before), ("calendar", after)]:
            rows.append(
                {
                    "currency": currency,
                    "month": "2024-01",
                    "date": 1704067200,
                    "short_expiration": 1704672000,
                    "long_expiration": 1705276800,
                    "selected": True,
                    "constraint_set": constraint_set,
                    "l2_status": "optimal",
                    "smooth_status": "optimal",
                    "l2_calendar_violation": 1e-9 if constraint_set == "calendar" else 1e-3,
                    "smooth_calendar_violation": 2e-9 if constraint_set == "calendar" else 2e-3,
                    "p90_scaled_curvature_diff": value,
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "calendar_spread_pair_results.csv", index=False)

    summary = load_summary_module().calendar_screen_summary(tmp_path)

    assert summary["matched_maturity_pairs"] == 2
    assert summary["optimization_rows"] == 4
    assert summary["all_rows_optimal"] is True
    assert summary["median_post_pre_dispersion_ratio"] == pytest.approx(1.0)
    assert summary["max_calendar_constraint_violation"] == pytest.approx(2e-9)
