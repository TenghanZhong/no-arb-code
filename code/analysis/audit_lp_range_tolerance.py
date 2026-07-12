from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "code" / "analysis"
COMMON = ANALYSIS / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
from full_strike_constraints import (  # noqa: E402
    certify_call_curve,
    effective_bounds,
)


LP_SCRIPT = ANALYSIS / "diagnostics" / "run_feasible_curvature_range.py"
LP_DIR = ROOT / "results" / "lp_range"
MERGED = LP_DIR / "feasible_curvature_range_merged_pairwise.csv"
CLASSICAL = ROOT / "results" / "classical_18m"
RELATIVE_TOLERANCE = 1e-5
STRICT_ABSOLUTE_TOLERANCE = 1e-10
AUDIT_ABSOLUTE_TOLERANCE = 1e-9
REFINEMENT_SOLVER_TOLERANCE = 2e-11
REFINEMENT_LP_TOLERANCE = 1e-10


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def refine_feasible_curve(
    approximate: np.ndarray,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    approximate = np.asarray(approximate, dtype=float)
    x = np.asarray(x, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    n = len(approximate)
    dx = np.diff(x)
    cumulative_dx = np.zeros((n, n - 1), dtype=float)
    for i in range(1, n):
        cumulative_dx[i, :i] = dx[:i]

    # Variables are first price, n-1 vertical slopes, and the L-infinity radius.
    # This parameterization enforces all strike-shape constraints without the
    # poorly scaled reciprocal-mesh coefficients in the direct curve matrix.
    curve_map = np.zeros((n, n + 1), dtype=float)
    curve_map[:, 0] = 1.0
    curve_map[:, 1:n] = -cumulative_dx
    radius_column = n

    upper_distance = curve_map.copy()
    upper_distance[:, radius_column] = -1.0
    lower_distance = -curve_map.copy()
    lower_distance[:, radius_column] = -1.0
    slope_order = np.zeros((max(n - 2, 0), n + 1), dtype=float)
    for i in range(n - 2):
        slope_order[i, i + 1] = -1.0
        slope_order[i, i + 2] = 1.0

    a_ub = np.vstack(
        [curve_map, -curve_map, upper_distance, lower_distance, slope_order]
    )
    b_ub = np.concatenate([upper, -lower, approximate, -approximate, np.zeros(n - 2)])
    objective = np.zeros(n + 1, dtype=float)
    objective[radius_column] = 1.0
    bounds = [(None, None)] + [(0.0, 1.0)] * (n - 1) + [(0.0, None)]
    result = linprog(
        c=objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs-ds",
        options={
            "primal_feasibility_tolerance": REFINEMENT_LP_TOLERANCE,
            "dual_feasibility_tolerance": REFINEMENT_LP_TOLERANCE,
        },
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"L-infinity feasibility refinement failed: {result.message}")

    slopes = np.minimum.accumulate(np.clip(result.x[1:n], 0.0, 1.0))
    first_price_lower = float(np.max(lower + cumulative_dx @ slopes))
    first_price_upper = float(np.min(upper + cumulative_dx @ slopes))
    interval_deficit = max(first_price_lower - first_price_upper, 0.0)
    # The midpoint of an inverted interval violates either side by at most
    # half the deficit, so allow twice the certification tolerance here.
    if interval_deficit > 2.0 * REFINEMENT_SOLVER_TOLERANCE:
        raise RuntimeError(
            f"Slope refinement leaves an empty first-price interval: {interval_deficit:.3e}"
        )
    if first_price_lower <= first_price_upper:
        first_price = float(np.clip(result.x[0], first_price_lower, first_price_upper))
    else:
        first_price = 0.5 * (first_price_lower + first_price_upper)
    refined = first_price - cumulative_dx @ slopes
    certification = certify_call_curve(
        refined,
        x,
        lower,
        upper,
        tolerance=REFINEMENT_SOLVER_TOLERANCE,
    )
    report = {
        "success": bool(certification["certified_feasible"]),
        "solver_status": "optimal",
        "parameterization": "nonincreasing_vertical_slopes",
        "first_price_interval_deficit": float(interval_deficit),
        "max_adjustment": float(np.max(np.abs(refined - approximate))),
        "max_constraint_violation": float(certification["max_constraint_violation"]),
    }
    return refined, report


def main() -> None:
    lp = load_module(LP_SCRIPT, "lp_range_tolerance")
    merged = pd.read_csv(MERGED)
    successful = merged[merged["range_status"].astype(str).eq("ok")].copy()
    gap = pd.to_numeric(successful["cxx_diff"], errors="coerce").abs()
    feasible_range = pd.to_numeric(successful["feasible_cxx_range"], errors="coerce")
    exceedances = gap > (
        feasible_range * (1.0 + RELATIVE_TOLERANCE) + STRICT_ABSOLUTE_TOLERANCE
    )
    exceeding = successful[exceedances].copy()
    exceeding["initial_abs_gap"] = gap.loc[exceeding.index]
    exceeding["initial_exceedance"] = (
        exceeding["initial_abs_gap"]
        - pd.to_numeric(exceeding["feasible_cxx_range"], errors="coerce")
    )

    refined_by_slice: dict[tuple, dict[str, object]] = {}
    slice_columns = ["run", "currency", "date", "expiration_date"]
    for slice_key, rows in exceeding.groupby(slice_columns, sort=True):
        run, currency, date, expiration = slice_key
        if not str(rows["source_family"].iloc[0]).startswith("classical"):
            raise ValueError(f"Unsupported exceeding source family: {rows['source_family'].iloc[0]}")
        run_dir = CLASSICAL / str(run)
        surfaces = pd.read_parquet(run_dir / "repaired_surfaces.parquet")
        surface_slice = surfaces[
            (surfaces["currency"].astype(str) == str(currency))
            & (pd.to_numeric(surfaces["date"], errors="coerce") == int(date))
            & (pd.to_numeric(surfaces["expiration_date"], errors="coerce") == int(expiration))
            & surfaces["arm"].astype(str).isin(["L2", "SMOOTH_RHO_10"])
        ].copy()
        base = (
            surface_slice[surface_slice["arm"].astype(str).eq("L2")]
            .sort_values("strike")
            .reset_index(drop=True)
        )
        if base.empty:
            raise RuntimeError(f"Missing L2 slice in {run_dir}: {slice_key}")
        strikes = pd.to_numeric(base["strike"], errors="raise").to_numpy(float)
        forward = pd.to_numeric(base["forward_price"], errors="raise").to_numpy(float)
        x = strikes / forward
        lower, upper = effective_bounds(
            x,
            pd.to_numeric(base["C_bid"], errors="raise").to_numpy(float),
            pd.to_numeric(base["C_ask"], errors="raise").to_numpy(float),
            pd.to_numeric(base["intrinsic"], errors="raise").to_numpy(float),
        )
        diagnostics = pd.read_csv(run_dir / "repair_diagnostics.csv")
        diagnostics = diagnostics[
            (diagnostics["currency"].astype(str) == str(currency))
            & (pd.to_numeric(diagnostics["date"], errors="coerce") == int(date))
            & (pd.to_numeric(diagnostics["expiration_date"], errors="coerce") == int(expiration))
            & diagnostics["arm"].astype(str).isin(["L2", "SMOOTH_RHO_10"])
        ].set_index("arm")
        arm_data = {}
        for arm in ("L2", "SMOOTH_RHO_10"):
            arm_slice = (
                surface_slice[surface_slice["arm"].astype(str).eq(arm)]
                .sort_values("strike")
                .reset_index(drop=True)
            )
            arm_strikes = pd.to_numeric(arm_slice["strike"], errors="raise").to_numpy(float)
            if not np.array_equal(arm_strikes, strikes):
                raise RuntimeError(f"Arm strike mismatch for {slice_key}: {arm}")
            curve = pd.to_numeric(arm_slice["C_repaired"], errors="raise").to_numpy(float)
            refined, report = refine_feasible_curve(curve, x, lower, upper)
            report["original_max_constraint_violation"] = float(
                pd.to_numeric(diagnostics.loc[arm, "max_constraint_violation"], errors="raise")
            )
            arm_data[arm] = {"curve": curve, "refined": refined, "report": report}
        refined_by_slice[slice_key] = {
            "strikes": strikes,
            "x": x,
            "arms": arm_data,
        }

    audit_rows = []
    for record in exceeding.to_dict("records"):
        slice_key = tuple(record[column] for column in slice_columns)
        data = refined_by_slice[slice_key]
        strikes = data["strikes"]
        matches = np.where(np.isclose(strikes, float(record["strike"]), rtol=0.0, atol=1e-9))[0]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely match strike for exceeding row: {record}")
        index = int(matches[0])
        row = lp.d2_row_nonuniform(data["x"], index)
        l2 = data["arms"]["L2"]
        smooth = data["arms"]["SMOOTH_RHO_10"]
        refined_gap = abs(float(row @ l2["refined"] - row @ smooth["refined"]))
        range_value = float(record["feasible_cxx_range"])
        strict_exceedance = refined_gap > (
            range_value * (1.0 + RELATIVE_TOLERANCE) + STRICT_ABSOLUTE_TOLERANCE
        )
        audit_exceedance = refined_gap > (
            range_value * (1.0 + RELATIVE_TOLERANCE)
            + AUDIT_ABSOLUTE_TOLERANCE
        )
        residual_scale = float(record["d2_row_l1_norm"]) * max(
            l2["report"]["original_max_constraint_violation"],
            smooth["report"]["original_max_constraint_violation"],
        )
        audit_rows.append(
            {
                **record,
                "refined_abs_gap": refined_gap,
                "strict_exceedance_after_refinement": strict_exceedance,
                "audit_tolerance_exceedance": audit_exceedance,
                "l2_max_adjustment": l2["report"]["max_adjustment"],
                "smooth10_max_adjustment": smooth["report"]["max_adjustment"],
                "l2_refined_max_violation": l2["report"]["max_constraint_violation"],
                "smooth10_refined_max_violation": smooth["report"]["max_constraint_violation"],
                "repair_residual_d2_scale": residual_scale,
                "initial_exceedance_within_residual_scale": bool(
                    float(record["initial_exceedance"]) <= residual_scale + 1e-12
                ),
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(LP_DIR / "lp_range_tolerance_audit.csv", index=False)
    projection_reports = [
        arm["report"]
        for value in refined_by_slice.values()
        for arm in value["arms"].values()
    ]
    summary = {
        "successful_lp_rows": int(len(successful)),
        "initial_exceedances": int(len(exceeding)),
        "affected_runs": int(exceeding["run"].nunique()),
        "affected_slices": int(exceeding[slice_columns].drop_duplicates().shape[0]),
        "curve_refinements": int(len(projection_reports)),
        "projection_failures": int(sum(not item["success"] for item in projection_reports)),
        "strict_tolerance_exceedances_after_refinement": int(audit["strict_exceedance_after_refinement"].sum()) if len(audit) else 0,
        "audit_tolerance_exceedances": int(
            audit["audit_tolerance_exceedance"].sum()
        ) if len(audit) else 0,
        "max_initial_exceedance": float(audit["initial_exceedance"].max()) if len(audit) else 0.0,
        "max_refined_gap": float(audit["refined_abs_gap"].max()) if len(audit) else 0.0,
        "max_refinement_adjustment": float(
            max(item["max_adjustment"] for item in projection_reports)
        ) if projection_reports else 0.0,
        "max_refined_constraint_violation": float(
            max(item["max_constraint_violation"] for item in projection_reports)
        ) if projection_reports else 0.0,
        "max_first_price_interval_deficit": float(
            max(item["first_price_interval_deficit"] for item in projection_reports)
        ) if projection_reports else 0.0,
        "exceedances_within_repair_residual_scale": int(
            audit["initial_exceedance_within_residual_scale"].sum()
        ) if len(audit) else 0,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "strict_absolute_tolerance": STRICT_ABSOLUTE_TOLERANCE,
        "audit_absolute_tolerance": AUDIT_ABSOLUTE_TOLERANCE,
        "refinement_solver_tolerance": REFINEMENT_SOLVER_TOLERANCE,
    }
    summary["audit_pass"] = bool(
        summary["projection_failures"] == 0
        and summary["audit_tolerance_exceedances"] == 0
        and summary["exceedances_within_repair_residual_scale"] == summary["initial_exceedances"]
        and summary["max_refined_constraint_violation"] <= REFINEMENT_SOLVER_TOLERANCE
    )
    (LP_DIR / "lp_range_tolerance_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not summary["audit_pass"]:
        raise RuntimeError("LP range-tolerance audit failed")


if __name__ == "__main__":
    main()
