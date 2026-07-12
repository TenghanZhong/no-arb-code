from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "manifests"

MODULES = {
    "classical": (ROOT / "results" / "classical_18m", 108, 50),
    "svi": (ROOT / "results" / "svi_18m", 108, 50),
    "targeted_nn": (ROOT / "results" / "nn_targeted", 10, 50),
}
SEED_MANIFEST = ROOT / "results" / "nn_seed_18m" / "gate_run_manifest.csv"

REQUIRED = (
    "phase2_config.json",
    "phase2_summary.csv",
    "phase2_input_panel_repair.parquet",
    "repaired_surfaces.parquet",
    "repair_diagnostics.csv",
    "strike_pairwise_dispersion.csv",
    "slice_feasibility_audit.csv",
    "pair_scale_recompute.json",
)

ALLOWED_TARGET_FAILURE_ARMS = {
    "classical": set(),
    "svi": {"SVI"},
    "targeted_nn": {"NNPROJ"},
    "seed_nn": {"NNPROJ"},
}
DETERMINISTIC_SEED_ARMS = {
    "L1",
    "L2",
    "SMOOTH_RHO_0P01",
    "SMOOTH_RHO_0P1",
    "SMOOTH_RHO_1",
    "SMOOTH_RHO_10",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diagnostic_certification_summary(module: str, diagnostics: pd.DataFrame) -> dict[str, object]:
    success = diagnostics["success"].astype(str).str.lower().eq("true")
    verified = diagnostics.get(
        "verified_feasible", pd.Series(False, index=diagnostics.index)
    ).astype(str).str.lower().eq("true")
    residual = pd.to_numeric(diagnostics["max_constraint_violation"], errors="coerce")
    slope = pd.to_numeric(diagnostics["max_vertical_slope"], errors="coerce")
    arms = diagnostics["arm"].astype(str).str.upper()
    failures = ~success
    allowed_arms = ALLOWED_TARGET_FAILURE_ARMS[module]
    allowed_failures = failures & arms.isin(allowed_arms)
    unexpected_failures = failures & ~arms.isin(allowed_arms)
    success_certified = success & verified & residual.le(2e-6) & slope.le(1.0 + 2e-6)
    successful_repairs_certified = bool(success.any() and success_certified.loc[success].all())

    def finite_max(values: pd.Series) -> float:
        values = pd.to_numeric(values, errors="coerce").dropna()
        return float(values.max()) if len(values) else np.nan

    return {
        "repair_attempts": int(len(diagnostics)),
        "repair_successes": int(success.sum()),
        "repair_failures": int(failures.sum()),
        "repair_success_rate": float(success.mean()) if len(success) else np.nan,
        "allowed_target_failures": int(allowed_failures.sum()),
        "unexpected_failures": int(unexpected_failures.sum()),
        "failed_repair_arms": ";".join(sorted(arms.loc[failures].unique())),
        "max_all_constraint_violation": finite_max(residual),
        "max_success_constraint_violation": finite_max(residual.loc[success]),
        "max_failed_constraint_violation": finite_max(residual.loc[failures]),
        "max_success_vertical_slope": finite_max(slope.loc[success]),
        "successful_repairs_certified": successful_repairs_certified,
        "diagnostic_audit_pass": bool(successful_repairs_certified and not unexpected_failures.any()),
    }


def audit_run(
    module: str,
    run_dir: Path,
    required_slices: int,
    analysis_slice_cap: int | None = None,
) -> dict[str, object]:
    missing = [name for name in REQUIRED if not (run_dir / name).exists()]
    row: dict[str, object] = {
        "module": module,
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "missing_files": ";".join(missing),
        "files_complete": not missing,
        "required_slices": required_slices,
    }
    if missing:
        row["audit_pass"] = False
        return row

    config = json.loads((run_dir / "phase2_config.json").read_text(encoding="utf-8"))
    pair_scale = json.loads((run_dir / "pair_scale_recompute.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(run_dir / "phase2_input_panel_repair.parquet")
    core = panel[panel.get("core_repair_slice", pd.Series(True, index=panel.index)).astype(bool)]
    selected_slices = core[["currency", "date", "expiration_date"]].drop_duplicates().shape[0]
    cap = int(required_slices if analysis_slice_cap is None else analysis_slice_cap)
    analysis_core = core.copy()
    if analysis_slice_cap is not None:
        if "slice_rank_desc" not in analysis_core.columns:
            raise ValueError(f"Missing slice_rank_desc in {run_dir}")
        ranks = pd.to_numeric(analysis_core["slice_rank_desc"], errors="coerce")
        analysis_core = analysis_core[ranks <= cap].copy()
    analysis_slices = analysis_core[["currency", "date", "expiration_date"]].drop_duplicates().shape[0]
    feasibility = pd.read_csv(run_dir / "slice_feasibility_audit.csv")
    diagnostics = pd.read_csv(run_dir / "repair_diagnostics.csv")

    diagnostic_summary = diagnostic_certification_summary(module, diagnostics)
    full_selected = bool(core.get("full_feasible", pd.Series(False, index=core.index)).astype(bool).all())
    config_ok = config.get("constraint_system") == "full_normalized_call_strike_v1"
    pair_scale_ok = pair_scale.get("metric_version") == "pair_specific_gamma_scale_v1"
    row.update(
        {
            "constraint_system": config.get("constraint_system", ""),
            "constraint_system_ok": config_ok,
            "pair_scale_metric_ok": pair_scale_ok,
            "selected_slices": selected_slices,
            "selected_count_ok": selected_slices == required_slices,
            "analysis_slice_cap": cap,
            "analysis_slices": analysis_slices,
            "analysis_count_ok": analysis_slices == cap,
            "selected_rows_full_feasible": full_selected,
            "candidate_slices": int(len(feasibility)),
            "feasible_candidate_slices": int(feasibility["full_feasible"].astype(str).str.lower().eq("true").sum()),
            **diagnostic_summary,
            "max_constraint_violation": diagnostic_summary["max_success_constraint_violation"],
            "max_vertical_slope": diagnostic_summary["max_success_vertical_slope"],
            "constraint_residual_ok": diagnostic_summary["successful_repairs_certified"],
            "vertical_slope_ok": diagnostic_summary["successful_repairs_certified"],
        }
    )
    if module in {"targeted_nn", "seed_nn"}:
        model_path = run_dir / "nnproj_oos_model_summary.json"
        model = json.loads(model_path.read_text(encoding="utf-8")) if model_path.exists() else {}
        info = model.get("model_info", {})
        row.update(
            {
                "nn_model_status": info.get("status", "missing"),
                "nn_oos_guard_pass": bool(info.get("oos_guard_pass", False)),
                "nn_history_panel_ok": bool(info.get("history_panel_ok", False)),
                "nn_train_months": info.get("n_train_months", np.nan),
            }
        )
    checks = [
        row["files_complete"],
        row["constraint_system_ok"],
        row["pair_scale_metric_ok"],
        row["selected_count_ok"],
        row["analysis_count_ok"],
        row["selected_rows_full_feasible"],
        row["diagnostic_audit_pass"],
    ]
    if module in {"targeted_nn", "seed_nn"}:
        checks.extend([row["nn_model_status"] == "trained_oos", row["nn_oos_guard_pass"], row["nn_history_panel_ok"]])
    row["audit_pass"] = bool(all(checks))
    return row


def write_working_hashes() -> None:
    working = ROOT / "code" / "working"
    rows = []
    for path in sorted(p for p in working.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(working).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(REPORTS / "working_code_sha256.csv", index=False)


def seed_run_specs(manifest_path: Path) -> list[dict[str, object]]:
    manifest = pd.read_csv(manifest_path)
    specs = []
    for _, record in manifest.iterrows():
        specs.append(
            {
                "run_dir": Path(record["output_dir"]),
                "raw_slice_count": int(record["raw_slice_count"]),
                "analysis_slice_cap": int(record["analysis_slice_cap"]),
            }
        )
    return specs


def audit_seed_manifest() -> dict[str, object]:
    path = SEED_MANIFEST
    manifest = pd.read_csv(path)
    seeds = {20260605, 20260606, 20260607, 20260608, 20260609}
    cells = manifest.groupby(["currency", "month"], as_index=False).agg(
        rows=("run", "size"), unique_seeds=("seed", "nunique")
    )
    result = {
        "manifest": str(path),
        "rows": int(len(manifest)),
        "unique_runs": int(manifest["run"].nunique()),
        "calendar_months": int(manifest["month"].nunique()),
        "currency_month_cells": int(len(cells)),
        "seeds": sorted(int(value) for value in manifest["seed"].unique()),
        "all_cells_have_five_rows": bool((cells["rows"] == 5).all()),
        "all_cells_have_five_unique_seeds": bool((cells["unique_seeds"] == 5).all()),
        "top_selection_only": bool(manifest["selection"].astype(str).eq("top").all()),
        "analysis_slice_cap": sorted(
            pd.to_numeric(manifest["analysis_slice_cap"], errors="raise").astype(int).unique().tolist()
        ),
        "raw_slice_counts": {
            str(int(key)): int(value)
            for key, value in pd.to_numeric(
                manifest["raw_slice_count"], errors="raise"
            ).astype(int).value_counts().sort_index().items()
        },
        "pilot_months_excluded": not bool(
            manifest["month"].astype(str).isin({"2025-01", "2026-05"}).any()
        ),
        "no_reused_runs": not bool(
            manifest["reused_existing_stage2"].astype(str).str.lower().eq("true").any()
        ),
        "expected_seed_set": set(pd.to_numeric(manifest["seed"], errors="raise").astype(int)) == seeds,
        "commands_require_full_lookback": bool(
            manifest["command"].astype(str).str.contains("--nnproj-min-train-months 12", regex=False).all()
        ),
        "commands_match_raw_slice_count": bool(
            all(
                f"--danger-slices {int(record.raw_slice_count)}" in str(record.command)
                for record in manifest.itertuples(index=False)
            )
        ),
        "all_outputs_complete": bool(
            manifest["output_dir"].astype(str).map(
                lambda value: (Path(value) / "phase2_summary.csv").exists()
            ).all()
        ),
        "amendment_recorded": bool(
            (ROOT / "results" / "nn_seed_18m" / "seed_design_amendment.json").exists()
        ),
    }
    result["audit_pass"] = bool(
        result["rows"] == 180
        and result["unique_runs"] == 180
        and result["calendar_months"] == 18
        and result["currency_month_cells"] == 36
        and all(
            result[key]
            for key in [
                "all_cells_have_five_rows",
                "all_cells_have_five_unique_seeds",
                "top_selection_only",
                "pilot_months_excluded",
                "no_reused_runs",
                "expected_seed_set",
                "commands_require_full_lookback",
                "commands_match_raw_slice_count",
                "all_outputs_complete",
                "amendment_recorded",
            ]
        )
        and result["analysis_slice_cap"] == [47]
        and result["raw_slice_counts"] == {"47": 5, "49": 175}
    )
    (REPORTS / "seed_replication_design_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def audit_seed_baseline_consistency(manifest: pd.DataFrame) -> dict[str, object]:
    slice_key_cols = ["currency", "date", "expiration_date"]
    surface_key_cols = ["currency", "date", "expiration_date", "strike", "arm"]
    slice_set_mismatches = 0
    baseline_key_mismatches = 0
    missing_deterministic_arm_runs = 0
    max_difference = 0.0
    cells = 0

    for _, group in manifest.groupby(["currency", "month"], sort=True):
        cells += 1
        reference_slices = None
        reference_baseline = None
        for _, record in group.sort_values("seed").iterrows():
            run_dir = Path(record["output_dir"])
            panel = pd.read_parquet(run_dir / "phase2_input_panel_repair.parquet")
            if "core_repair_slice" in panel.columns:
                panel = panel[panel["core_repair_slice"].astype(bool)]
            cap_value = pd.to_numeric(
                pd.Series([record.get("analysis_slice_cap", np.nan)]), errors="coerce"
            ).iloc[0]
            if np.isfinite(cap_value):
                if "slice_rank_desc" not in panel.columns:
                    raise ValueError(f"Missing slice_rank_desc in {run_dir}")
                ranks = pd.to_numeric(panel["slice_rank_desc"], errors="coerce")
                panel = panel[ranks <= int(cap_value)].copy()
            slices = set(map(tuple, panel[slice_key_cols].drop_duplicates().itertuples(index=False, name=None)))
            if reference_slices is None:
                reference_slices = slices
            elif slices != reference_slices:
                slice_set_mismatches += 1

            surfaces = pd.read_parquet(run_dir / "repaired_surfaces.parquet")
            if "core_repair_slice" in surfaces.columns:
                surfaces = surfaces[surfaces["core_repair_slice"].astype(bool)]
            surface_keys = pd.DataFrame(list(slices), columns=slice_key_cols)
            surfaces = surfaces.merge(surface_keys, on=slice_key_cols, how="inner", validate="many_to_one")
            baseline = surfaces[surfaces["arm"].astype(str).isin(DETERMINISTIC_SEED_ARMS)].copy()
            if set(baseline["arm"].astype(str).unique()) != DETERMINISTIC_SEED_ARMS:
                missing_deterministic_arm_runs += 1
            baseline = baseline[surface_key_cols + ["C_repaired"]].sort_values(surface_key_cols).reset_index(drop=True)
            if reference_baseline is None:
                reference_baseline = baseline
                continue
            if not reference_baseline[surface_key_cols].equals(baseline[surface_key_cols]):
                baseline_key_mismatches += 1
                continue
            difference = np.abs(
                pd.to_numeric(reference_baseline["C_repaired"], errors="coerce").to_numpy()
                - pd.to_numeric(baseline["C_repaired"], errors="coerce").to_numpy()
            )
            if len(difference):
                max_difference = max(max_difference, float(np.nanmax(difference)))

    result = {
        "currency_month_cells": int(cells),
        "slice_set_mismatches": int(slice_set_mismatches),
        "baseline_key_mismatches": int(baseline_key_mismatches),
        "missing_deterministic_arm_runs": int(missing_deterministic_arm_runs),
        "max_baseline_abs_difference": float(max_difference),
        "comparison_tolerance": 1e-12,
    }
    result["audit_pass"] = bool(
        not slice_set_mismatches
        and not baseline_key_mismatches
        and not missing_deterministic_arm_runs
        and max_difference <= result["comparison_tolerance"]
    )
    return result


def audit_seed_equivalence(report: dict[str, object]) -> dict[str, object]:
    comparisons = report.get("comparisons", {})
    frame_passes = [bool(item.get("pass", False)) for item in comparisons.values()]
    differences = [
        float(item.get("max_abs_difference", np.nan)) for item in comparisons.values()
    ]
    finite_differences = [value for value in differences if np.isfinite(value)]
    result = {
        "reported_pass": bool(report.get("pass", False)),
        "model_fields_equal": bool(report.get("model_fields_equal", False)),
        "frame_checks": int(len(frame_passes)),
        "frame_checks_passed": int(sum(frame_passes)),
        "max_abs_difference": max(finite_differences) if finite_differences else np.nan,
    }
    result["audit_pass"] = bool(
        result["reported_pass"]
        and result["model_fields_equal"]
        and frame_passes
        and all(frame_passes)
        and np.isfinite(result["max_abs_difference"])
        and result["max_abs_difference"] <= 1e-12
    )
    return result


def audit_calendar_results(results: pd.DataFrame, expected_pairs: int = 12) -> dict[str, object]:
    selected = results[results["selected"].astype(str).str.lower().eq("true")].copy()
    pair_keys = ["currency", "month", "date", "candidate_rank"]
    sets = selected.groupby(pair_keys)["constraint_set"].agg(lambda values: set(values.astype(str)))
    matched = sets.map(lambda values: values == {"strike_only", "calendar"})
    optimal = selected["l2_status"].astype(str).eq("optimal") & selected["smooth_status"].astype(str).eq("optimal")
    calendar = selected[selected["constraint_set"].astype(str).eq("calendar")]
    l2_violation = pd.to_numeric(calendar["l2_calendar_violation"], errors="coerce")
    smooth_violation = pd.to_numeric(calendar["smooth_calendar_violation"], errors="coerce")
    max_calendar_violation = float(
        pd.concat([l2_violation, smooth_violation], ignore_index=True).dropna().max()
    ) if len(calendar) else np.nan
    result = {
        "selected_rows": int(len(selected)),
        "matched_pairs": int(len(sets)),
        "pairs_with_both_specifications": int(matched.sum()),
        "optimal_rows": int(optimal.sum()),
        "max_calendar_violation": max_calendar_violation,
        "expected_pairs": int(expected_pairs),
    }
    result["audit_pass"] = bool(
        len(sets) == expected_pairs
        and matched.all()
        and optimal.all()
        and len(calendar) == expected_pairs
        and np.isfinite(max_calendar_violation)
        and max_calendar_violation <= 5e-6
    )
    return result


def audit_lp_configuration(config: dict[str, object]) -> dict[str, object]:
    expected_constraints = (
        "effective normalized-call bounds + monotonicity + "
        "vertical-spread upper bound + convexity"
    )
    solver_tolerance = float(config.get("highs_solver_tolerance", np.nan))
    certification_tolerance = float(
        config.get("repair_matched_certification_tolerance", np.nan)
    )
    empty_box_tolerance = float(config.get("empty_box_tolerance", np.nan))
    result = {
        "constraint_system_ok": config.get("lp_constraints") == expected_constraints,
        "highs_solver_tolerance": solver_tolerance,
        "repair_matched_certification_tolerance": certification_tolerance,
        "empty_box_tolerance": empty_box_tolerance,
    }
    result["audit_pass"] = bool(
        result["constraint_system_ok"]
        and np.isclose(certification_tolerance, 2e-6, rtol=0.0, atol=1e-15)
        and np.isclose(empty_box_tolerance, 1e-12, rtol=0.0, atol=1e-18)
        and np.isfinite(solver_tolerance)
        and 0.0 < solver_tolerance <= certification_tolerance
    )
    return result


def audit_lp_flag_polish(report: dict[str, object]) -> dict[str, object]:
    original_flags = int(report.get("original_range_flags", -1))
    strict_polished_flags = int(report.get("polished_range_flags", -1))
    conditioned_polished_flags = int(
        report.get("polished_conditioned_range_flags", -1)
    )
    residual_scaled_flags = int(
        report.get("flags_within_repair_residual_d2_scale", -1)
    )
    projection_failures = int(report.get("projection_failures", -1))
    max_violation = float(report.get("max_polished_constraint_violation", np.nan))
    polish_tolerance = float(report.get("polish_solver_tolerance", np.nan))
    result = {
        "reported_pass": bool(report.get("audit_pass", False)),
        "original_range_flags": original_flags,
        "projection_failures": projection_failures,
        "polished_range_flags": strict_polished_flags,
        "polished_conditioned_range_flags": conditioned_polished_flags,
        "flags_within_repair_residual_d2_scale": residual_scaled_flags,
        "max_polished_constraint_violation": max_violation,
        "polish_solver_tolerance": polish_tolerance,
    }
    result["audit_pass"] = bool(
        result["reported_pass"]
        and original_flags >= 0
        and projection_failures == 0
        and conditioned_polished_flags == 0
        and strict_polished_flags >= conditioned_polished_flags
        and residual_scaled_flags == original_flags
        and np.isfinite(max_violation)
        and np.isfinite(polish_tolerance)
        and 0.0 < polish_tolerance
        and max_violation <= polish_tolerance
    )
    return result


def main() -> None:
    rows = []
    expected_summary = []
    for module, (root, expected, required_slices) in MODULES.items():
        run_dirs = sorted(
            path for path in root.iterdir()
            if path.is_dir() and not path.name.startswith("_")
        ) if root.exists() else []
        for run_dir in run_dirs:
            rows.append(audit_run(module, run_dir, required_slices))
        expected_summary.append(
            {
                "module": module,
                "expected_runs": expected,
                "found_run_dirs": len(run_dirs),
                "run_count_ok": len(run_dirs) == expected,
            }
        )
    seed_specs = seed_run_specs(SEED_MANIFEST)
    for spec in seed_specs:
        rows.append(
            audit_run(
                "seed_nn",
                spec["run_dir"],
                spec["raw_slice_count"],
                spec["analysis_slice_cap"],
            )
        )
    expected_summary.append(
        {
            "module": "seed_nn",
            "expected_runs": 180,
            "found_run_dirs": len(seed_specs),
            "run_count_ok": len(seed_specs) == 180,
        }
    )
    audit = pd.DataFrame(rows)
    expected_df = pd.DataFrame(expected_summary)
    audit.to_csv(REPORTS / "rerun_integrity_by_run.csv", index=False)
    expected_df.to_csv(REPORTS / "rerun_integrity_counts.csv", index=False)
    write_working_hashes()
    seed_design = audit_seed_manifest()
    seed_manifest = pd.read_csv(SEED_MANIFEST)
    seed_baseline = audit_seed_baseline_consistency(seed_manifest)
    (REPORTS / "seed_baseline_consistency_audit.json").write_text(
        json.dumps(seed_baseline, indent=2), encoding="utf-8"
    )
    equivalence_report = json.loads(
        (
            ROOT
            / "results"
            / "nn_seed_18m"
            / "seed_top49_subset_vs_native_top47_equivalence.json"
        ).read_text(encoding="utf-8")
    )
    seed_equivalence = audit_seed_equivalence(equivalence_report)
    (REPORTS / "seed_slice_subset_equivalence_audit.json").write_text(
        json.dumps(seed_equivalence, indent=2), encoding="utf-8"
    )
    calendar_results = pd.read_csv(ROOT / "results" / "calendar" / "calendar_certified_pair_results.csv")
    calendar_audit = audit_calendar_results(calendar_results)
    (REPORTS / "calendar_integrity_audit.json").write_text(
        json.dumps(calendar_audit, indent=2), encoding="utf-8"
    )
    lp_config = json.loads((ROOT / "results" / "lp_range" / "config.json").read_text(encoding="utf-8"))
    lp_config_audit = audit_lp_configuration(lp_config)
    (REPORTS / "lp_configuration_audit.json").write_text(
        json.dumps(lp_config_audit, indent=2), encoding="utf-8"
    )
    lp_flag_report = json.loads(
        (ROOT / "results" / "lp_range" / "range_flag_polish_audit.json").read_text(
            encoding="utf-8"
        )
    )
    lp_flag_audit = audit_lp_flag_polish(lp_flag_report)
    (REPORTS / "lp_range_flag_polish_integrity_audit.json").write_text(
        json.dumps(lp_flag_audit, indent=2), encoding="utf-8"
    )
    summary = {
        "expected_counts_pass": bool(expected_df["run_count_ok"].all()),
        "runs_audited": int(len(audit)),
        "runs_passed": int(audit.get("audit_pass", pd.Series(dtype=bool)).sum()),
        "runs_failed": int((~audit.get("audit_pass", pd.Series(dtype=bool))).sum()),
        "max_constraint_violation": float(pd.to_numeric(audit.get("max_constraint_violation"), errors="coerce").max()),
        "max_attempted_constraint_violation": float(pd.to_numeric(audit.get("max_all_constraint_violation"), errors="coerce").max()),
        "max_vertical_slope": float(pd.to_numeric(audit.get("max_vertical_slope"), errors="coerce").max()),
        "repair_failures": int(pd.to_numeric(audit.get("repair_failures"), errors="coerce").fillna(0).sum()),
        "unexpected_repair_failures": int(pd.to_numeric(audit.get("unexpected_failures"), errors="coerce").fillna(0).sum()),
        "seed_design_pass": bool(seed_design["audit_pass"]),
        "seed_baseline_consistency_pass": bool(seed_baseline["audit_pass"]),
        "seed_slice_subset_equivalence_pass": bool(seed_equivalence["audit_pass"]),
        "calendar_audit_pass": bool(calendar_audit["audit_pass"]),
        "lp_configuration_audit_pass": bool(lp_config_audit["audit_pass"]),
        "lp_range_flag_polish_audit_pass": bool(lp_flag_audit["audit_pass"]),
    }
    (REPORTS / "rerun_integrity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(expected_df.to_string(index=False))
    if (
        not summary["expected_counts_pass"]
        or summary["runs_failed"]
        or not summary["seed_design_pass"]
        or not summary["seed_baseline_consistency_pass"]
        or not summary["seed_slice_subset_equivalence_pass"]
        or not summary["calendar_audit_pass"]
        or not summary["lp_configuration_audit_pass"]
        or not summary["lp_range_flag_polish_audit_pass"]
    ):
        raise RuntimeError("Rerun integrity audit failed; inspect manifests/rerun_integrity_by_run.csv")


if __name__ == "__main__":
    main()
