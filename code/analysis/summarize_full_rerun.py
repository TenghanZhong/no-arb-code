from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DERIVED = ROOT / "derived"
CLASSICAL = RESULTS / "classical_18m" / "_aggregate"
SVI = RESULTS / "svi_18m" / "_aggregate"
TARGETED = RESULTS / "nn_targeted" / "_aggregate"
SEED = RESULTS / "nn_seed_18m"
LP = RESULTS / "lp_range"
CALENDAR = RESULTS / "calendar"
HEDGE_PERSISTENCE = DERIVED / "hedge_persistence"
STANDARD_BOOKS = ["ATM_STRADDLE", "PUT_SPREAD_5_10", "PUT_FLY_0_5_10", "RISK_REVERSAL_5"]


def json_sanitize(value):
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def pair_rows(df: pd.DataFrame, arm_a: str, arm_b: str) -> pd.DataFrame:
    return df[
        ((df["arm_a"] == arm_a) & (df["arm_b"] == arm_b))
        | ((df["arm_a"] == arm_b) & (df["arm_b"] == arm_a))
    ].copy()


def exact_sign(values: pd.Series) -> dict[str, object]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    positive = int((values > 0).sum())
    n = int(len(values))
    result = binomtest(positive, n, 0.5) if n else None
    one_sided = binomtest(positive, n, 0.5, alternative="greater").pvalue if n else math.nan
    return {
        "positive": positive,
        "n": n,
        "one_sided_p": float(one_sided),
        "two_sided_p": float(result.pvalue) if result else math.nan,
    }


def targeted_heldout_summary(root: Path) -> dict[str, object]:
    path = root / "nnproj_heldout_fit_by_run.csv"
    audits = pd.read_csv(path)
    status = audits["audit_status"].astype(str)
    successful = audits[status.eq("ok")].copy()

    def median(column: str) -> float:
        values = pd.to_numeric(successful[column], errors="coerce").dropna()
        return float(values.median()) if len(values) else math.nan

    return {
        "attempts": int(len(audits)),
        "successes": int(len(successful)),
        "status_counts": {str(key): int(value) for key, value in status.value_counts().items()},
        "median_oos_log_total_variance_r2": median("oos_log_tv_r2"),
        "median_raw_price_outside_box_share": median("oos_price_box_outside_rate"),
        "median_spread_scaled_price_mae": median("oos_price_mae_scaled"),
    }


def lp_range_audit(root: Path) -> dict[str, object]:
    merged = pd.read_csv(root / "feasible_curvature_range_merged_pairwise.csv")
    status = merged["range_status"].fillna("missing").astype(str)
    successful = merged[status.eq("ok")].copy()
    gap = pd.to_numeric(successful["cxx_diff"], errors="coerce").abs()
    feasible_range = pd.to_numeric(successful["feasible_cxx_range"], errors="coerce")
    raw_exceedance = gap - feasible_range
    exceedances = gap > (feasible_range * (1.0 + 1e-5) + 1e-10)
    residuals = pd.concat(
        [
            pd.to_numeric(successful["lp_min_max_constraint_violation"], errors="coerce"),
            pd.to_numeric(successful["lp_max_max_constraint_violation"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    relative_exceedance = gap / feasible_range.where(feasible_range > 1e-12) - 1.0
    tolerance_path = root / "lp_range_tolerance_audit.json"
    lp_tolerance_audit = (
        json.loads(tolerance_path.read_text(encoding="utf-8"))
        if tolerance_path.exists()
        else None
    )
    return {
        "rows": int(len(merged)),
        "successful_rows": int(len(successful)),
        "status_counts": {str(key): int(value) for key, value in status.value_counts().items()},
        "range_exceedances": int(exceedances.fillna(False).sum()),
        "max_raw_exceedance": float(max(0.0, raw_exceedance.max())) if len(raw_exceedance.dropna()) else math.nan,
        "max_relative_exceedance": float(max(0.0, relative_exceedance.max())) if len(relative_exceedance.dropna()) else math.nan,
        "max_lp_constraint_violation": float(residuals.max()) if len(residuals) else math.nan,
        "lp_tolerance_audit": lp_tolerance_audit,
    }


def hedge_notional_anchor(root: Path) -> dict[str, object]:
    summary = pd.read_csv(root / "risk_reversal_hedge_persistence_summary.csv")
    if len(summary) != 1:
        raise ValueError(f"Expected one hedge-persistence summary row, found {len(summary)}")
    row = summary.iloc[0]
    return {
        "runs_total": int(row["n_runs_total"]),
        "coverage_qualified_runs": int(row["n_runs_coverage_pass"]),
        "coverage_qualified_pairs": int(row["n_pairs_coverage_pass"]),
        "median_run_p90_hedge_notional_bps": float(row["median_run_p90_abs_gap_bps"]),
        "usd_per_1m_gross_notional": float(row["median_run_p90_abs_gap_usd_per_1mm"]),
    }


def surface_mechanism_summary(root: Path) -> dict[str, object]:
    summary_frame = pd.read_csv(root / "surface_mechanism_summary.csv")
    audit = json.loads((root / "surface_mechanism_audit.json").read_text(encoding="utf-8"))
    pairs: dict[str, dict[str, object]] = {}
    for record in summary_frame.to_dict("records"):
        key = f"{record.pop('source')}:{record.pop('pair')}"
        pairs[key] = {
            str(name): (
                None
                if pd.isna(value)
                else value.item()
                if hasattr(value, "item")
                else value
            )
            for name, value in record.items()
        }
    return {"audit": audit, "pairs": pairs}


def calendar_screen_summary(root: Path) -> dict[str, object]:
    rows = pd.read_csv(root / "calendar_spread_pair_results.csv")
    if "selected" in rows:
        selected = rows["selected"].astype(str).str.lower().isin({"true", "1"})
        rows = rows[selected].copy()
    keys = ["currency", "month", "date", "short_expiration", "long_expiration"]
    pivot = rows.pivot(index=keys, columns="constraint_set", values="p90_scaled_curvature_diff")
    matched = pivot.dropna(subset=["strike_only", "calendar"])
    ratios = pd.to_numeric(matched["calendar"], errors="coerce") / pd.to_numeric(
        matched["strike_only"], errors="coerce"
    ).replace(0.0, np.nan)
    calendar_rows = rows[rows["constraint_set"].astype(str).eq("calendar")]
    residuals = pd.concat(
        [
            pd.to_numeric(calendar_rows["l2_calendar_violation"], errors="coerce"),
            pd.to_numeric(calendar_rows["smooth_calendar_violation"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    l2_ok = rows["l2_status"].astype(str).str.lower().eq("optimal")
    smooth_ok = rows["smooth_status"].astype(str).str.lower().eq("optimal")
    return {
        "matched_maturity_pairs": int(len(matched)),
        "optimization_rows": int(len(rows)),
        "all_rows_optimal": bool((l2_ok & smooth_ok).all()),
        "median_post_pre_dispersion_ratio": float(ratios.median()),
        "max_calendar_constraint_violation": float(residuals.max()) if len(residuals) else math.nan,
    }


def classical_summary() -> tuple[dict[str, object], pd.DataFrame]:
    arms = pd.read_csv(CLASSICAL / "ALL_arm_pair_summary.csv")
    strike = arms[arms["level"] == "strike"].copy()
    smooth = pair_rows(strike, "L2", "SMOOTH_RHO_10")[
        ["run", "currency", "month", "slice_selection_from_run", "p90_gamma_diff_scaled"]
    ].rename(columns={"p90_gamma_diff_scaled": "smooth_p90"})
    control = pair_rows(strike, "L1", "L2")[["run", "p90_gamma_diff_scaled"]].rename(
        columns={"p90_gamma_diff_scaled": "control_p90"}
    )
    paired = smooth.merge(control, on="run", how="inner")
    paired["excess"] = paired["smooth_p90"] - paired["control_p90"]
    cell = (
        paired.groupby(["currency", "month"], as_index=False)
        .agg(
            smooth_p90=("smooth_p90", "median"),
            control_p90=("control_p90", "median"),
            excess=("excess", "median"),
        )
    )
    calendar = (
        cell.groupby("month", as_index=False)
        .agg(
            smooth_p90=("smooth_p90", "median"),
            control_p90=("control_p90", "median"),
            excess=("excess", "median"),
        )
        .sort_values("month")
    )

    rho = pd.read_csv(CLASSICAL / "ALL_rho_vs_baseline_summary.csv")
    monotonic = []
    for (run, baseline), group in rho.groupby(["run", "baseline"]):
        ordered = group.sort_values("rho")
        gamma = pd.to_numeric(ordered["p90_gamma_diff_scaled"], errors="coerce").to_numpy()
        hedge = pd.to_numeric(ordered["p90_hedge_slippage_scaled"], errors="coerce").to_numpy()
        monotonic.append(
            {
                "run": run,
                "baseline": baseline,
                "gamma_nondecreasing": bool(np.all(np.diff(gamma) >= -1e-10)),
                "hedge_nondecreasing": bool(np.all(np.diff(hedge[np.isfinite(hedge)]) >= -1e-10)),
            }
        )
    monotonic_df = pd.DataFrame(monotonic)

    hedge_anchor = hedge_notional_anchor(HEDGE_PERSISTENCE)

    var = pd.read_csv(CLASSICAL / "ALL_var_style_pairwise.csv")
    var = pair_rows(var, "L2", "SMOOTH_RHO_10")
    var = var[
        var["book"].isin(STANDARD_BOOKS)
        & (pd.to_numeric(var["n_loss_a"], errors="coerce") >= 10)
        & (pd.to_numeric(var["n_loss_b"], errors="coerce") >= 10)
    ].copy()
    var_stats = {}
    for loss_type in ("common", "pipeline"):
        values = pd.to_numeric(
            var.loc[var["loss_type"] == loss_type, "var_a0p95_diff_scaled"], errors="coerce"
        ).dropna()
        var_stats[loss_type] = {
            "cells": int(len(values)),
            "median": float(values.median()),
            "p90": float(values.quantile(0.9)),
        }

    result = {
        "evaluations": int(len(paired)),
        "currency_month_cells": int(len(cell)),
        "calendar_months": int(len(calendar)),
        "median_evaluation_p90_smooth10_gamma_gap": float(paired["smooth_p90"].median()),
        "median_evaluation_p90_l1_l2_control": float(paired["control_p90"].median()),
        "median_evaluation_paired_excess": float(paired["excess"].median()),
        "median_calendar_month_excess": float(calendar["excess"].median()),
        "calendar_sign_test": exact_sign(calendar["excess"]),
        "gamma_path_nondecreasing": f"{int(monotonic_df['gamma_nondecreasing'].sum())}/{len(monotonic_df)}",
        "hedge_path_nondecreasing": f"{int(monotonic_df['hedge_nondecreasing'].sum())}/{len(monotonic_df)}",
        "risk_reversal_hedge_anchor": hedge_anchor,
        "coverage_qualified_risk_reversal_runs": hedge_anchor["coverage_qualified_runs"],
        "risk_reversal_p90_hedge_notional_bps": hedge_anchor["median_run_p90_hedge_notional_bps"],
        "risk_reversal_usd_per_1m": hedge_anchor["usd_per_1m_gross_notional"],
        "var_style": var_stats,
    }
    return result, calendar


def projection_success(root: Path, arm: str) -> dict[str, object]:
    frames = [pd.read_csv(path).assign(run=path.parent.name) for path in root.glob("*/repair_diagnostics.csv")]
    if not frames:
        return {"arm": arm, "attempts": 0, "successes": 0}
    diagnostics = pd.concat(frames, ignore_index=True)
    diagnostics = diagnostics[diagnostics["arm"].astype(str).str.upper() == arm.upper()].copy()
    success = diagnostics["success"].astype(str).str.lower().eq("true")
    failures = diagnostics.loc[~success]
    verified = diagnostics.get("verified_feasible", pd.Series(False, index=diagnostics.index)).astype(str).str.lower().eq("true")
    optimal = diagnostics.get("status", pd.Series("", index=diagnostics.index)).astype(str).str.lower().isin(
        {"0", "optimal", "optimal_inaccurate"}
    )
    failure_reason = pd.Series("solver_or_other_failure", index=diagnostics.index)
    failure_reason.loc[~success & optimal & ~verified] = "postsolve_constraint_certification"
    residual = pd.to_numeric(diagnostics["max_constraint_violation"], errors="coerce")
    return {
        "arm": arm,
        "attempts": int(len(diagnostics)),
        "successes": int(success.sum()),
        "success_rate": float(success.mean()) if len(success) else math.nan,
        "failure_status_counts": {
            str(key): int(value)
            for key, value in failures.get("status", pd.Series(dtype=str)).astype(str).value_counts().items()
        },
        "failure_message_counts": {
            str(key): int(value)
            for key, value in failures.get("message", pd.Series(dtype=str)).astype(str).value_counts().items()
        },
        "failure_reason_counts": {
            str(key): int(value)
            for key, value in failure_reason.loc[~success].value_counts().items()
        },
        "max_constraint_violation": float(residual.max()),
        "max_success_constraint_violation": float(residual.loc[success].max()),
        "max_failed_constraint_violation": float(residual.loc[~success].max()),
        "max_vertical_slope": float(pd.to_numeric(diagnostics["max_vertical_slope"], errors="coerce").max()),
    }


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    classical, calendar = classical_summary()
    calendar.to_csv(DERIVED / "classical_calendar_month_excess.csv", index=False)

    svi_contrast = pd.read_csv(SVI / "svi_run_level_contrasts.csv")
    svi_result = {
        "evaluations": int(len(svi_contrast)),
        "median_l2_svi_p90_gamma_gap": float(pd.to_numeric(svi_contrast["L2__SVI"], errors="coerce").median()),
        "median_l1_l2_control": float(pd.to_numeric(svi_contrast["L1__L2"], errors="coerce").median()),
        "positive_excess_evaluations": int((pd.to_numeric(svi_contrast["svi_minus_l1l2"], errors="coerce") > 0).sum()),
        "projection": projection_success(RESULTS / "svi_18m", "SVI"),
    }

    targeted_pairs = pd.read_csv(TARGETED / "ALL_arm_pair_summary.csv")
    targeted_strike = targeted_pairs[targeted_pairs["level"] == "strike"]
    targeted_result = {
        "evaluations": int(targeted_strike["run"].nunique()),
        "projection": projection_success(RESULTS / "nn_targeted", "NNPROJ"),
        "heldout_fit": targeted_heldout_summary(DERIVED / "targeted_nn_heldout_fit"),
    }
    for label, arm_a, arm_b in [
        ("l2_nnproj", "L2", "NNPROJ"),
        ("nnproj_smooth10", "NNPROJ", "SMOOTH_RHO_10"),
        ("l1_l2", "L1", "L2"),
    ]:
        values = pd.to_numeric(
            pair_rows(targeted_strike, arm_a, arm_b)["p90_gamma_diff_scaled"], errors="coerce"
        ).dropna()
        targeted_result[f"median_p90_gamma_gap_{label}"] = float(values.median())

    seed_result = json.loads((SEED / "analysis_results.json").read_text(encoding="utf-8"))
    lp_summary = pd.read_csv(LP / "feasible_curvature_range_family_summary.csv")
    lp_target = lp_summary[
        (lp_summary["source"] == "classical_selected_slices")
        & (lp_summary["pair_label"] == "L2_vs_SMOOTH_RHO_10")
        & (lp_summary["selection"] == "top")
    ]
    lp_result = lp_target.iloc[0].to_dict() if len(lp_target) else {}
    lp_result["audit"] = lp_range_audit(LP)

    final = {
        "classical": classical,
        "svi": svi_result,
        "targeted_nn": targeted_result,
        "seed_replication": seed_result,
        "lp_range": lp_result,
        "calendar_screen": calendar_screen_summary(CALENDAR),
        "surface_mechanism": surface_mechanism_summary(DERIVED),
    }
    final_json = json_sanitize(final)
    (DERIVED / "full_rerun_summary.json").write_text(
        json.dumps(final_json, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# Full Strike-Arbitrage Rerun Summary",
        "",
        f"- Classical: {classical['evaluations']} evaluations; Smooth10 gamma gap {classical['median_evaluation_p90_smooth10_gamma_gap']:.6g} versus control {classical['median_evaluation_p90_l1_l2_control']:.6g}.",
        f"- Calendar excess: median {classical['median_calendar_month_excess']:.6g}; positive {classical['calendar_sign_test']['positive']}/{classical['calendar_sign_test']['n']}; two-sided p={classical['calendar_sign_test']['two_sided_p']:.6g}.",
        f"- Risk-reversal hedge anchor: {classical['risk_reversal_p90_hedge_notional_bps']:.2f} bps, or USD {classical['risk_reversal_usd_per_1m']:.0f} per USD 1m gross notional.",
        f"- SVI: median L2/SVI gap {svi_result['median_l2_svi_p90_gamma_gap']:.6g} versus control {svi_result['median_l1_l2_control']:.6g}.",
        f"- Seed replication: {seed_result['common_support_d2_positive_calendar_months']} positive months.",
        f"- LP: n={int(lp_result.get('n_rows', 0)):,}, Spearman={float(lp_result.get('spearman_range_vs_abs_cxx', math.nan)):.4f}.",
    ]
    (DERIVED / "FULL_RERUN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final_json, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
