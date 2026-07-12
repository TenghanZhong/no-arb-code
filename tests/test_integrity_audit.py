from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "analysis" / "verify_full_rerun.py"


def load_module():
    name = "test_verify_full_rerun"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "arm": "L2",
                "success": True,
                "verified_feasible": True,
                "max_constraint_violation": 1e-8,
                "max_vertical_slope": 0.99,
            },
            {
                "arm": "SVI",
                "success": False,
                "verified_feasible": False,
                "max_constraint_violation": 8e-6,
                "max_vertical_slope": 1.000008,
            },
        ]
    )


def test_failed_target_is_accounted_without_invalidating_certified_outputs() -> None:
    summary = load_module().diagnostic_certification_summary("svi", diagnostics())

    assert summary["repair_failures"] == 1
    assert summary["allowed_target_failures"] == 1
    assert summary["unexpected_failures"] == 0
    assert summary["max_all_constraint_violation"] == 8e-6
    assert summary["max_success_constraint_violation"] == 1e-8
    assert summary["successful_repairs_certified"] is True
    assert summary["diagnostic_audit_pass"] is True


def test_failed_base_repair_fails_integrity_audit() -> None:
    frame = diagnostics()
    frame.loc[1, "arm"] = "L2"

    summary = load_module().diagnostic_certification_summary("svi", frame)

    assert summary["unexpected_failures"] == 1
    assert summary["diagnostic_audit_pass"] is False


def write_seed_run(path: Path, l2_value: float) -> None:
    path.mkdir()
    pd.DataFrame(
        [
            {"currency": "BTC", "date": 1, "expiration_date": 2, "core_repair_slice": True},
            {"currency": "BTC", "date": 3, "expiration_date": 4, "core_repair_slice": True},
        ]
    ).to_parquet(path / "repair_panel.parquet", index=False)
    values = {
        "L1": 0.1,
        "L2": l2_value,
        "SMOOTH_RHO_0P01": 0.11,
        "SMOOTH_RHO_0P1": 0.12,
        "SMOOTH_RHO_1": 0.13,
        "SMOOTH_RHO_10": 0.14,
        "NNPROJ": 0.3,
    }
    pd.DataFrame(
        [
            {
                "currency": "BTC",
                "date": 1,
                "expiration_date": 2,
                "strike": 100.0,
                "arm": arm,
                "C_repaired": value,
                "core_repair_slice": True,
            }
            for arm, value in values.items()
        ]
    ).to_parquet(path / "repaired_surfaces.parquet", index=False)


def test_seed_baseline_consistency_requires_identical_deterministic_outputs(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    write_seed_run(run_a, 0.2)
    write_seed_run(run_b, 0.2)
    manifest = pd.DataFrame(
        [
            {"currency": "BTC", "month": "2024-01", "seed": 1, "output_dir": str(run_a)},
            {"currency": "BTC", "month": "2024-01", "seed": 2, "output_dir": str(run_b)},
        ]
    )

    passed = load_module().audit_seed_baseline_consistency(manifest)
    assert passed["audit_pass"] is True
    assert passed["max_baseline_abs_difference"] == 0.0

    surfaces = pd.read_parquet(run_b / "repaired_surfaces.parquet")
    surfaces.loc[surfaces["arm"].eq("L2"), "C_repaired"] = 0.21
    surfaces.to_parquet(run_b / "repaired_surfaces.parquet", index=False)

    failed = load_module().audit_seed_baseline_consistency(manifest)
    assert failed["audit_pass"] is False
    assert failed["max_baseline_abs_difference"] == pytest.approx(0.01)


def test_calendar_audit_requires_matched_optimal_pairs() -> None:
    frame = pd.DataFrame(
        [
            {
                "currency": "BTC",
                "month": "2024-01",
                "date": 1,
                "candidate_rank": 1,
                "selected": True,
                "constraint_set": constraint_set,
                "l2_status": "optimal",
                "smooth_status": "optimal",
                "l2_calendar_violation": 0.0,
                "smooth_calendar_violation": 0.0,
            }
            for constraint_set in ("strike_only", "calendar")
        ]
    )

    passed = load_module().audit_calendar_results(frame, expected_pairs=1)
    assert passed["audit_pass"] is True
    assert passed["matched_pairs"] == 1

    frame.loc[frame["constraint_set"].eq("calendar"), "smooth_calendar_violation"] = 1e-4
    failed = load_module().audit_calendar_results(frame, expected_pairs=1)
    assert failed["audit_pass"] is False


def test_lp_configuration_matches_repair_certification_policy() -> None:
    config = {
        "lp_constraints": "effective normalized-call bounds + monotonicity + vertical-spread upper bound + convexity",
        "highs_solver_tolerance": 1e-9,
        "repair_matched_certification_tolerance": 2e-6,
        "empty_box_tolerance": 1e-12,
    }

    passed = load_module().audit_lp_configuration(config)
    assert passed["audit_pass"] is True

    config["repair_matched_certification_tolerance"] = 2e-7
    failed = load_module().audit_lp_configuration(config)
    assert failed["audit_pass"] is False


def test_lp_range_tolerance_audit_requires_zero_audit_exceedances() -> None:
    report = {
        "audit_pass": True,
        "initial_exceedances": 129,
        "projection_failures": 0,
        "strict_tolerance_exceedances_after_refinement": 2,
        "audit_tolerance_exceedances": 0,
        "exceedances_within_repair_residual_scale": 129,
        "max_refined_constraint_violation": 1.1e-11,
        "refinement_solver_tolerance": 2e-11,
    }

    passed = load_module().audit_lp_range_tolerance(report)
    assert passed["audit_pass"] is True
    assert passed["strict_tolerance_exceedances_after_refinement"] == 2

    report["audit_tolerance_exceedances"] = 1
    assert load_module().audit_lp_range_tolerance(report)["audit_pass"] is False
