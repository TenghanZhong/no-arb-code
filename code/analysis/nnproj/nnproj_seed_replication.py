#!/usr/bin/env python3
"""Run and summarize the 18-month NN-Proj seed replication.

The analysis runs 180 refits on the fixed month panel and evaluates
dispersion on the common support where all five projections succeed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
SENSITIVITY_SCRIPT = ANALYSIS_ROOT / "nnproj" / "nnproj_sensitivity.py"
DEFAULT_OUT = ROOT / "results" / "nn_seed_18m"
DESIGN_CONFIG = ROOT / "config" / "analysis_design.json"
SEED_DESIGN = json.loads(DESIGN_CONFIG.read_text(encoding="utf-8"))["seed_replication"]

FULL_LOOKBACK_CANDIDATE_START = "2023-10"
FULL_LOOKBACK_CANDIDATE_END = "2026-05"
EXCLUDED_MONTHS = list(SEED_DESIGN["excluded_months"])
N_REPLICATION_MONTHS = int(SEED_DESIGN["calendar_months"])
CURRENCIES = list(SEED_DESIGN["currencies"])
SELECTION = str(SEED_DESIGN["selection_group"])
HIDDEN = "64,64"
MIN_TRAIN_MONTHS = 12


def load_sensitivity_module() -> Any:
    spec = importlib.util.spec_from_file_location("nnproj_sensitivity", SENSITIVITY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sensitivity script: {SENSITIVITY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def month_range(start: str, end: str) -> list[str]:
    return pd.period_range(start, end, freq="M").astype(str).tolist()


def month_index(months: list[str]) -> dict[str, int]:
    return {month: idx for idx, month in enumerate(months)}


def maximin_months(candidates: list[str], n_months: int, universe: list[str]) -> list[str]:
    """Endpoint-seeded greedy maximin month selection with earliest-month ties."""
    if n_months > len(candidates):
        raise ValueError(f"Need {n_months} months but only {len(candidates)} candidates")
    idx = month_index(universe)
    selected = [candidates[0]]
    if n_months > 1:
        selected.append(candidates[-1])
    while len(selected) < n_months:
        best_month = None
        best_key = None
        for month in candidates:
            if month in selected:
                continue
            min_distance = min(abs(idx[month] - idx[chosen]) for chosen in selected)
            key = (min_distance, -idx[month])
            if best_key is None or key > best_key:
                best_key = key
                best_month = month
        if best_month is None:
            raise RuntimeError("Maximin month selection failed")
        selected.append(best_month)
    return sorted(selected, key=lambda month: idx[month])


def selected_replication_months() -> tuple[list[str], list[str], list[str]]:
    universe = month_range(FULL_LOOKBACK_CANDIDATE_START, FULL_LOOKBACK_CANDIDATE_END)
    excluded = set(EXCLUDED_MONTHS)
    candidates = [month for month in universe if month not in excluded]
    selected = maximin_months(candidates, N_REPLICATION_MONTHS, universe)
    return selected, candidates, universe


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(args: list[str]) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def seed_slice_cap() -> int:
    cap = int(SEED_DESIGN["analysis_slice_cap"])
    if cap <= 0:
        raise ValueError(f"Invalid seed slice cap: {cap}")
    return cap


def run_name_for_slices(analysis: Any, currency: str, month: str, seed: int, selected_slices: int) -> str:
    return (
        f"{currency}_{month}_{SELECTION}{selected_slices}_nnproj_"
        f"arch{analysis.hidden_tag(HIDDEN)}_seed{seed}"
    )


def build_repair_command(
    analysis: Any,
    out_dir: Path,
    currency: str,
    month: str,
    seed: int,
    selected_slices: int | None = None,
) -> list[str]:
    cmd = analysis.repair_command(
        out_dir=out_dir,
        currency=currency,
        month=month,
        selection=SELECTION,
        hidden=HIDDEN,
        nn_seed=seed,
    )
    selected_slice_count = int(analysis.SELECTED_SLICES if selected_slices is None else selected_slices)
    for idx, token in enumerate(cmd[:-1]):
        if token == "--selected-slices":
            cmd[idx + 1] = str(selected_slice_count)
            break
    else:
        raise RuntimeError("repair command lacks --selected-slices")
    for idx, token in enumerate(cmd[:-1]):
        if token == "--nnproj-min-train-months":
            cmd[idx + 1] = str(MIN_TRAIN_MONTHS)
            break
    else:
        raise RuntimeError("repair command lacks --nnproj-min-train-months")
    return cmd


def build_manifest(out_root: Path) -> pd.DataFrame:
    analysis = load_sensitivity_module()
    analysis_slice_cap = seed_slice_cap()
    replication_dir = out_root / f"replication_top{analysis_slice_cap}"
    replication_dir.mkdir(parents=True, exist_ok=True)
    months, candidate_months, universe_months = selected_replication_months()

    rows: list[dict[str, Any]] = []
    for currency in CURRENCIES:
        for month in months:
            for seed in analysis.NN_SEEDS:
                run_name = run_name_for_slices(analysis, currency, month, seed, analysis_slice_cap)
                out_dir = replication_dir / run_name
                cmd = build_repair_command(
                    analysis=analysis,
                    out_dir=out_dir,
                    currency=currency,
                    month=month,
                    seed=seed,
                    selected_slices=analysis_slice_cap,
                )
                rows.append(
                    {
                        "replication_set": replication_dir.name,
                        "currency": currency,
                        "month": month,
                        "selection": SELECTION,
                        "hidden": HIDDEN,
                        "seed": int(seed),
                        "run": run_name,
                        "run_name": run_name,
                        "output_dir": str(out_dir),
                        "command": " ".join(cmd),
                        "analysis_slice_cap": analysis_slice_cap,
                    }
                )

    manifest = pd.DataFrame(rows)
    manifest_path = out_root / "run_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    universe = {
        "purpose": (
            "18-month NN-Proj five-seed replication with a common "
            f"{analysis_slice_cap}-slice analysis cap"
        ),
        "months": months,
        "candidate_months_full_12m_lookback": candidate_months,
        "candidate_universe": universe_months,
        "excluded_months": EXCLUDED_MONTHS,
        "selection_rule": (
            "Restrict to months with BTC and ETH full 12-month lookback and 7-day "
            "embargo; exclude the two fixed months listed above; "
            "select 18 months by endpoint-seeded greedy maximin spacing with "
            "earliest-month tie-breaking."
        ),
        "currencies": CURRENCIES,
        "selection": SELECTION,
        "selected_slices": analysis_slice_cap,
        "hidden": HIDDEN,
        "min_train_months": MIN_TRAIN_MONTHS,
        "lookback_months": 12,
        "embargo_days": 7,
        "seeds": [int(s) for s in analysis.NN_SEEDS],
        "runs": int(len(manifest)),
        "replication_dir": str(replication_dir),
        "primary_statistic": (
            "all-five-seed common-support median seed-pair p90 nonuniform D2 gap "
            "minus matched L1/L2 control p90 nonuniform D2 gap"
        ),
        "calendar_aggregation": "median across BTC and ETH currency-month cells",
        "panel": str(analysis.PANEL),
        "panel_sha256": file_sha256(analysis.PANEL),
        "repair_script": str(analysis.REPAIR_SCRIPT),
        "repair_script_sha256": file_sha256(analysis.REPAIR_SCRIPT),
        "sensitivity_script": str(SENSITIVITY_SCRIPT),
        "sensitivity_script_sha256": file_sha256(SENSITIVITY_SCRIPT),
        "replication_script": str(Path(__file__).resolve()),
        "replication_script_sha256": file_sha256(Path(__file__).resolve()),
        "git_head": git_value(["rev-parse", "HEAD"]),
        "git_status_short": git_value(["status", "--short"]),
        "python": sys.executable,
        "python_version": sys.version,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    (out_root / "analysis_universe.json").write_text(json.dumps(universe, indent=2), encoding="utf-8")
    return manifest


def exact_sign_pvalues(k: int, n: int) -> dict[str, float]:
    upper = sum(math.comb(n, j) for j in range(k, n + 1)) / (2**n)
    lower = sum(math.comb(n, j) for j in range(0, k + 1)) / (2**n)
    return {"one_sided_positive": upper, "two_sided": min(1.0, 2.0 * min(upper, lower))}


def run_jobs(out_root: Path, limit_runs: int | None, rerun: bool, jobs: int) -> None:
    analysis = load_sensitivity_module()
    manifest_path = out_root / "run_manifest.csv"
    if not manifest_path.exists():
        build_manifest(out_root)
    manifest = pd.read_csv(manifest_path)
    rows = list(manifest.to_dict("records"))
    if limit_runs and limit_runs > 0:
        rows = rows[:limit_runs]

    def run_one(rec: dict[str, Any], idx: int, total: int) -> dict[str, Any]:
        out_dir = Path(rec["output_dir"])
        summary = out_dir / "repair_summary.csv"
        if summary.exists() and not rerun:
            msg = f"[skip {idx}/{total}] {rec['run']} already complete"
            print(msg, flush=True)
            return {"run": rec["run"], "status": "skipped", "returncode": 0, "elapsed_sec": 0.0}
        if rerun and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_repair_command(
            analysis=analysis,
            out_dir=out_dir,
            currency=str(rec["currency"]),
            month=str(rec["month"]),
            seed=int(rec["seed"]),
            selected_slices=int(rec["analysis_slice_cap"]),
        )
        (out_dir / "command.json").write_text(
            json.dumps({"command": cmd, "record": rec}, indent=2), encoding="utf-8"
        )
        print(f"[run {idx}/{total}] {rec['run']}", flush=True)
        t0 = time.time()
        with (out_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
            out_dir / "stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            env = os.environ.copy()
            env.update({
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            })
            ret = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr)
        elapsed = time.time() - t0
        if ret.returncode == 0:
            print(f"[ok] {rec['run']} returncode=0 elapsed={elapsed:.1f}s", flush=True)
            status = "ok"
        else:
            print(f"[failed] {rec['run']} returncode={ret.returncode} elapsed={elapsed:.1f}s", flush=True)
            status = "failed"
        return {"run": rec["run"], "status": status, "returncode": ret.returncode, "elapsed_sec": elapsed}

    total = len(rows)
    if jobs <= 1:
        results = [run_one(rec, i, total) for i, rec in enumerate(rows, start=1)]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(run_one, rec, i, total) for i, rec in enumerate(rows, start=1)]
            for fut in as_completed(futs):
                results.append(fut.result())
    pd.DataFrame(results).to_csv(out_root / "run_results.csv", index=False)
    failures = [r for r in results if r["returncode"] != 0]
    if failures:
        raise RuntimeError(f"{len(failures)} runs failed; see run_results.csv")


def audit_fits(out_root: Path) -> None:
    analysis = load_sensitivity_module()
    audit_dir = out_root / "_analysis_fit_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(out_root / "run_manifest.csv")
    rows: list[dict[str, Any]] = []
    for _, rec in manifest.iterrows():
        run_dir = Path(rec["output_dir"])
        model = json.loads((run_dir / "nnproj_model_summary.json").read_text(encoding="utf-8"))
        info = model.get("model_info", {})
        surf = pd.read_parquet(run_dir / "repaired_surfaces.parquet")
        nn = analysis.filter_to_run_analysis_slices(surf[surf["arm"] == "NNPROJ"].copy(), rec)
        diffs = pd.to_numeric(nn.get("repair_norm_mid_diff"), errors="coerce").dropna()
        price_rmse = float(np.sqrt(np.nanmean(np.square(diffs)))) if len(diffs) else np.nan
        price_mae = float(np.nanmean(np.abs(diffs))) if len(diffs) else np.nan
        rows.append(
            {
                "run": rec["run"],
                "currency": rec["currency"],
                "month": rec["month"],
                "seed": int(rec["seed"]),
                "audit_status": "ok",
                "fit_metric_source": "post_projection_selected_slice_repair_norm_mid_diff",
                "oos_price_rmse_scaled": price_rmse,
                "oos_price_mae_scaled": price_mae,
                "oos_log_tv_r2": np.nan,
                "train_r2": info.get("r2_train", np.nan),
                "n_train": info.get("n_train", np.nan),
                "n_train_months": info.get("n_train_months", np.nan),
                "n_train_months_raw": info.get("n_train_months_raw", np.nan),
                "n_train_months_valid_prefilter": info.get("n_train_months_valid_prefilter", np.nan),
                "min_train_months_required_recorded": info.get("min_train_months_required", np.nan),
                "strict_full_12m_lookback_pass": int(info.get("n_train_months", 0)) >= MIN_TRAIN_MONTHS,
                "training_window_valid": info.get("training_window_valid", np.nan),
                "history_panel_ok": info.get("history_panel_ok", np.nan),
                "train_start": info.get("train_start", ""),
                "train_end_exclusive": info.get("train_end_exclusive", ""),
            }
        )
    fit = pd.DataFrame(rows)
    fit.to_csv(audit_dir / "nnproj_heldout_fit_by_run.csv", index=False)
    agg = {
        "runs": int(len(fit)),
        "strict_full_12m_lookback_pass": int(fit["strict_full_12m_lookback_pass"].sum()),
        "strict_full_12m_lookback_total": int(len(fit)),
        "price_rmse_scaled_median": float(fit["oos_price_rmse_scaled"].median()),
        "price_rmse_scaled_min": float(fit["oos_price_rmse_scaled"].min()),
        "price_rmse_scaled_max": float(fit["oos_price_rmse_scaled"].max()),
        "train_r2_median": float(pd.to_numeric(fit["train_r2"], errors="coerce").median()),
        "fit_metric_source": "post_projection_selected_slice_repair_norm_mid_diff",
    }
    (audit_dir / "light_fit_audit_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")


def summarize(out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis = load_sensitivity_module()
    repair_module = analysis.load_repair_module()
    manifest_path = out_root / "run_manifest.csv"
    fit_path = out_root / "_analysis_fit_audit" / "nnproj_heldout_fit_by_run.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not fit_path.exists():
        audit_fits(out_root)
    manifest = pd.read_csv(manifest_path)
    fit_by_run = pd.read_csv(fit_path)

    window_rows: list[dict[str, Any]] = []
    pair_rows: list[pd.DataFrame] = []
    for _, group in manifest.groupby(["currency", "month", "selection", "hidden"], sort=True):
        window, pairs = analysis.summarize_window(repair_module, group, fit_by_run)
        window_rows.append(window)
        if not pairs.empty:
            pair_rows.append(pairs)

    window_df = pd.DataFrame(window_rows).sort_values(["month", "currency", "selection"]).reset_index(drop=True)
    pair_df = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()

    out_root.mkdir(parents=True, exist_ok=True)
    window_df.to_csv(out_root / "window_summary.csv", index=False)
    pair_df.to_csv(out_root / "seed_pair_summary.csv", index=False)

    numeric_cols = window_df.select_dtypes(include=[np.number]).columns.tolist()
    calendar_df = (
        window_df.groupby("month", as_index=False)[numeric_cols]
        .median(numeric_only=True)
        .sort_values("month")
        .reset_index(drop=True)
    )
    calendar_df.to_csv(out_root / "calendar_month_summary.csv", index=False)

    common_excess_col = "common_support_seed_pair_excess_over_l1_l2"
    win_positive = int((window_df[common_excess_col] > 0).sum())
    win_n = int(window_df[common_excess_col].notna().sum())
    cal_positive = int((calendar_df[common_excess_col] > 0).sum())
    cal_n = int(calendar_df[common_excess_col].notna().sum())
    results = {
        "runs_manifested": int(len(manifest)),
        "runs_completed": int(
            manifest["output_dir"].map(lambda p: Path(p, "repair_summary.csv").exists()).sum()
        ),
        "windows": int(len(window_df)),
        "calendar_months": int(len(calendar_df)),
        "common_support_d2_positive_currency_months": f"{win_positive}/{win_n}",
        "common_support_d2_positive_calendar_months": f"{cal_positive}/{cal_n}",
        "currency_month_sign_test": exact_sign_pvalues(win_positive, win_n),
        "calendar_month_sign_test": exact_sign_pvalues(cal_positive, cal_n),
        "median_common_support_seed_pair_p90_d2_gap": float(
            window_df["common_support_seed_pair_median_p90_d2_curvature_gap"].median()
        ),
        "median_common_support_l1_l2_control_p90_d2_gap": float(
            window_df["common_support_control_l1_l2_p90_d2_curvature_gap"].median()
        ),
        "median_common_support_d2_excess": float(window_df[common_excess_col].median()),
        "min_all_seed_common_support_share": float(
            window_df["all_seed_common_success_share"].min()
        ),
        "median_all_seed_common_support_share": float(
            window_df["all_seed_common_success_share"].median()
        ),
        "max_worst_seed_price_rmse_pct_above_best": float(
            window_df["fit_worst_pct_above_best"].max()
        ),
        "median_worst_seed_price_rmse_pct_above_best": float(
            window_df["fit_worst_pct_above_best"].median()
        ),
    }

    for col in [
        "seed_pair_spot_gamma_excess_over_l1_l2",
        "median_seed_pair_p90_hedge_gap_impact",
        "median_seed_pair_p90_var_common_diff",
        "median_seed_pair_p90_var_pipeline_diff",
    ]:
        if col in window_df.columns:
            vals = window_df[col].dropna()
            if not vals.empty:
                results[f"median_{col}"] = float(vals.median())
                if "excess" in col:
                    results[f"positive_{col}_currency_months"] = f"{int((vals > 0).sum())}/{int(len(vals))}"

    (out_root / "analysis_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown_summary(out_root, results)
    return window_df, pair_df, calendar_df


def write_markdown_summary(out_root: Path, results: dict[str, Any]) -> None:
    lines = [
        "# 18-Month NN-Proj Seed Replication",
        "",
        f"- Manifested runs: {results['runs_manifested']}",
        f"- Completed runs: {results['runs_completed']}",
        f"- Windows: {results['windows']} currency-month cells",
        f"- Calendar months: {results['calendar_months']}",
        (
            "- Common-support D2 excess positive: "
            f"{results['common_support_d2_positive_currency_months']} currency-month cells; "
            f"{results['common_support_d2_positive_calendar_months']} calendar months"
        ),
        (
            "- Calendar-month sign test: "
            f"one-sided p={results['calendar_month_sign_test']['one_sided_positive']:.6g}; "
            f"two-sided p={results['calendar_month_sign_test']['two_sided']:.6g}"
        ),
        (
            "- Median common-support p90 D2 gap: "
            f"{results['median_common_support_seed_pair_p90_d2_gap']:.6g}; "
            "matched L1/L2 control: "
            f"{results['median_common_support_l1_l2_control_p90_d2_gap']:.6g}; "
            "excess: "
            f"{results['median_common_support_d2_excess']:.6g}"
        ),
        (
            "- All-seed common support share: "
            f"median {results['median_all_seed_common_support_share']:.3f}; "
            f"min {results['min_all_seed_common_support_share']:.3f}"
        ),
        (
            "- Worst seed held-out spread-scaled price RMSE above best: "
            f"median {results['median_worst_seed_price_rmse_pct_above_best']:.2f}%; "
            f"max {results['max_worst_seed_price_rmse_pct_above_best']:.2f}%"
        ),
        "",
        "Outputs:",
        "- run_manifest.csv",
        "- window_summary.csv",
        "- seed_pair_summary.csv",
        "- calendar_month_summary.csv",
        "- analysis_results.json",
    ]
    (out_root / "SEED_REPLICATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["manifest", "run", "audit", "summarize", "all"])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit-runs", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.command == "manifest":
        manifest = build_manifest(out_root)
        print(f"Wrote manifest with {len(manifest)} runs to {out_root}")
    elif args.command == "run":
        run_jobs(out_root, args.limit_runs, args.rerun, args.jobs)
    elif args.command == "audit":
        audit_fits(out_root)
    elif args.command == "summarize":
        summarize(out_root)
    elif args.command == "all":
        manifest_path = out_root / "run_manifest.csv"
        if not manifest_path.exists():
            manifest = build_manifest(out_root)
            print(f"Wrote manifest with {len(manifest)} runs to {out_root}")
        run_jobs(out_root, args.limit_runs, args.rerun, args.jobs)
        audit_fits(out_root)
        summarize(out_root)


if __name__ == "__main__":
    main()
