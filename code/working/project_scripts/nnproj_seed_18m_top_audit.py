#!/usr/bin/env python3
"""Run and summarize the 18-month top-selection NN-Proj seed audit.

This wrapper keeps the seed-multiplicity replication isolated from the original
stage2 output. It runs all 180 refits from scratch on the preselected 18-month
panel and reuses only the gate summarizer so the all-seed common-support
definition stays identical.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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


WORKING_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
GATE_SCRIPT = WORKING_ROOT / "project_scripts" / "nnproj_gate_escalation.py"
DEFAULT_OUT = ROOT / "results" / "nn_seed_18m"
PREFLIGHT_CONFIG = ROOT / "manifests" / "full_feasibility_preflight_config.json"
PREFLIGHT_SUMMARY = ROOT / "manifests" / "full_feasibility_preflight_by_cell.csv"

FULL_LOOKBACK_CANDIDATE_START = "2023-10"
FULL_LOOKBACK_CANDIDATE_END = "2026-05"
EXCLUDED_PILOT_MONTHS = ["2025-01", "2026-05"]
N_REPLICATION_MONTHS = 18
CURRENCIES = ["BTC", "ETH"]
SELECTION = "top"
HIDDEN = "64,64"
MIN_TRAIN_MONTHS = 12


def load_gate_module() -> Any:
    spec = importlib.util.spec_from_file_location("nnproj_gate_escalation", GATE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load gate script: {GATE_SCRIPT}")
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
    excluded = set(EXCLUDED_PILOT_MONTHS)
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
    data = json.loads(PREFLIGHT_CONFIG.read_text(encoding="utf-8"))
    cap = int(data["seed_replication_slices_per_cell"])
    if cap <= 0:
        raise ValueError(f"Invalid seed slice cap: {cap}")
    return cap


def run_name_for_slices(gate: Any, currency: str, month: str, seed: int, danger_slices: int) -> str:
    return (
        f"{currency}_{month}_{SELECTION}{danger_slices}_nnproj_"
        f"arch{gate.hidden_tag(HIDDEN)}_seed{seed}"
    )


def strict_phase2_command(
    gate: Any,
    out_dir: Path,
    currency: str,
    month: str,
    seed: int,
    danger_slices: int | None = None,
) -> list[str]:
    cmd = gate.phase2_command(
        out_dir=out_dir,
        currency=currency,
        month=month,
        selection=SELECTION,
        hidden=HIDDEN,
        nn_seed=seed,
    )
    raw_slice_count = int(gate.DANGER_SLICES if danger_slices is None else danger_slices)
    for idx, token in enumerate(cmd[:-1]):
        if token == "--danger-slices":
            cmd[idx + 1] = str(raw_slice_count)
            break
    else:
        raise RuntimeError("phase2 command lacks --danger-slices")
    for idx, token in enumerate(cmd[:-1]):
        if token == "--nnproj-min-train-months":
            cmd[idx + 1] = str(MIN_TRAIN_MONTHS)
            break
    else:
        raise RuntimeError("phase2 command lacks --nnproj-min-train-months")
    return cmd


def build_amended_manifest(
    manifest: pd.DataFrame,
    *,
    out_root: Path,
    gate: Any,
    analysis_slice_cap: int,
    rerun_cells: set[tuple[str, str]],
) -> pd.DataFrame:
    amended = manifest.copy()
    amended["source_output_dir"] = amended["output_dir"].astype(str)
    amended["analysis_slice_cap"] = int(analysis_slice_cap)
    amended["raw_slice_count"] = int(gate.DANGER_SLICES)
    amended["amendment_action"] = "reuse_ranked_subset"
    rerun_stage = out_root / f"stage18_top{analysis_slice_cap}_missing"
    for idx, rec in amended.iterrows():
        cell = (str(rec["currency"]), str(rec["month"]))
        if cell not in rerun_cells:
            continue
        run = run_name_for_slices(
            gate,
            str(rec["currency"]),
            str(rec["month"]),
            int(rec["seed"]),
            analysis_slice_cap,
        )
        out_dir = rerun_stage / run
        cmd = strict_phase2_command(
            gate,
            out_dir,
            str(rec["currency"]),
            str(rec["month"]),
            int(rec["seed"]),
            danger_slices=analysis_slice_cap,
        )
        amended.loc[idx, "stage"] = rerun_stage.name
        amended.loc[idx, "run"] = run
        amended.loc[idx, "run_name"] = run
        amended.loc[idx, "output_dir"] = str(out_dir)
        amended.loc[idx, "command"] = " ".join(cmd)
        amended.loc[idx, "raw_slice_count"] = int(analysis_slice_cap)
        amended.loc[idx, "amendment_action"] = "rerun_at_analysis_cap"
    return amended


def build_manifest(out_root: Path) -> pd.DataFrame:
    gate = load_gate_module()
    analysis_slice_cap = seed_slice_cap()
    stage_dir = out_root / f"stage18_top{analysis_slice_cap}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    months, candidate_months, universe_months = selected_replication_months()

    rows: list[dict[str, Any]] = []
    for currency in CURRENCIES:
        for month in months:
            for seed in gate.NN_SEEDS:
                run_name = run_name_for_slices(gate, currency, month, seed, analysis_slice_cap)
                out_dir = stage_dir / run_name
                cmd = strict_phase2_command(
                    gate=gate,
                    out_dir=out_dir,
                    currency=currency,
                    month=month,
                    seed=seed,
                    danger_slices=analysis_slice_cap,
                )
                rows.append(
                    {
                        "stage": stage_dir.name,
                        "currency": currency,
                        "month": month,
                        "selection": SELECTION,
                        "hidden": HIDDEN,
                        "seed": int(seed),
                        "run": run_name,
                        "run_name": run_name,
                        "output_dir": str(out_dir),
                        "reused_existing_stage2": False,
                        "command": " ".join(cmd),
                        "raw_slice_count": analysis_slice_cap,
                        "analysis_slice_cap": analysis_slice_cap,
                        "amendment_action": "fresh_common_cap_run",
                    }
                )

    manifest = pd.DataFrame(rows)
    manifest_path = out_root / "gate_run_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    universe = {
        "purpose": (
            "strict 18-month NN-Proj 5-seed multiplicity replication with a common "
            f"{analysis_slice_cap}-slice analysis cap"
        ),
        "months": months,
        "candidate_months_full_12m_lookback": candidate_months,
        "candidate_universe": universe_months,
        "excluded_pilot_months": EXCLUDED_PILOT_MONTHS,
        "selection_rule": (
            "Restrict to months with BTC and ETH full 12-month lookback and 7-day "
            "embargo; exclude previously inspected seed-multiplicity pilot months; "
            "select 18 months by endpoint-seeded greedy maximin spacing with "
            "earliest-month tie-breaking."
        ),
        "currencies": CURRENCIES,
        "selection": SELECTION,
        "danger_slices": analysis_slice_cap,
        "hidden": HIDDEN,
        "min_train_months": MIN_TRAIN_MONTHS,
        "lookback_months": 12,
        "embargo_days": 7,
        "seeds": [int(s) for s in gate.NN_SEEDS],
        "runs": int(len(manifest)),
        "stage_dir": str(stage_dir),
        "reused_completed_stage2_sources": 0,
        "primary_statistic": (
            "all-five-seed common-support median seed-pair p90 nonuniform D2 gap "
            "minus matched L1/L2 control p90 nonuniform D2 gap"
        ),
        "calendar_aggregation": "median across BTC and ETH currency-month cells",
        "panel": str(gate.PANEL),
        "panel_sha256": file_sha256(gate.PANEL),
        "phase2_script": str(gate.PHASE2),
        "phase2_sha256": file_sha256(gate.PHASE2),
        "gate_script": str(GATE_SCRIPT),
        "gate_script_sha256": file_sha256(GATE_SCRIPT),
        "wrapper_script": str(Path(__file__).resolve()),
        "wrapper_script_sha256": file_sha256(Path(__file__).resolve()),
        "git_head": git_value(["rev-parse", "HEAD"]),
        "git_status_short": git_value(["status", "--short"]),
        "python": sys.executable,
        "python_version": sys.version,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    (out_root / "gate_universe.json").write_text(json.dumps(universe, indent=2), encoding="utf-8")
    return manifest


def amend_existing_manifest(out_root: Path) -> pd.DataFrame:
    gate = load_gate_module()
    manifest_path = out_root / "gate_run_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    original_cap = int(gate.DANGER_SLICES)
    analysis_cap = seed_slice_cap()
    preflight = pd.read_csv(PREFLIGHT_SUMMARY)
    seed_rows = preflight[preflight["design"] == "seed"].copy()
    rerun_cells = set(
        map(
            tuple,
            seed_rows.loc[
                pd.to_numeric(seed_rows["eligible_slices"], errors="coerce") < original_cap,
                ["currency", "month"],
            ].itertuples(index=False, name=None),
        )
    )
    if not rerun_cells:
        raise RuntimeError("No seed cells require an amended raw slice count")
    result_files = [
        out_root / "gate_window_summary.csv",
        out_root / "gate_seed_pair_summary.csv",
        out_root / "calendar_month_summary.csv",
        out_root / "gate_decision.json",
    ]
    if any(path.exists() for path in result_files):
        raise RuntimeError("Seed outcomes were already summarized; refusing outcome-blind amendment")

    backup = out_root / "gate_run_manifest_pre_amendment.csv"
    if backup.exists():
        raise FileExistsError(f"Amendment backup already exists: {backup}")
    shutil.copy2(manifest_path, backup)
    for name in ("gate_universe.json", "run_results.csv"):
        source = out_root / name
        if source.exists():
            shutil.copy2(source, out_root / f"{source.stem}_pre_amendment{source.suffix}")

    amended = build_amended_manifest(
        manifest,
        out_root=out_root,
        gate=gate,
        analysis_slice_cap=analysis_cap,
        rerun_cells=rerun_cells,
    )
    amended.to_csv(manifest_path, index=False)
    changed = amended[amended["amendment_action"] == "rerun_at_analysis_cap"]
    amendment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "The original preflight used min_strikes=5 while the frozen Phase-2 "
            "pipeline uses min_strikes=7. The exact preflight leaves 47 eligible "
            "slices in ETH 2026-04."
        ),
        "outcomes_summarized_before_amendment": False,
        "original_raw_slice_count": original_cap,
        "analysis_slice_cap": analysis_cap,
        "rerun_cells": [list(cell) for cell in sorted(rerun_cells)],
        "rerun_count": int(len(changed)),
        "reused_ranked_subset_count": int(len(amended) - len(changed)),
        "reuse_basis": (
            "Neural training uses the embargoed historical panel rather than selected "
            "evaluation slices; projections and downstream calculations are slice-local."
        ),
        "manifest_before_sha256": file_sha256(backup),
        "manifest_after_sha256": file_sha256(manifest_path),
        "preflight_config_sha256": file_sha256(PREFLIGHT_CONFIG),
        "preflight_summary_sha256": file_sha256(PREFLIGHT_SUMMARY),
    }
    (out_root / "seed_design_amendment.json").write_text(
        json.dumps(amendment, indent=2), encoding="utf-8"
    )
    return amended


def exact_sign_pvalues(k: int, n: int) -> dict[str, float]:
    upper = sum(math.comb(n, j) for j in range(k, n + 1)) / (2**n)
    lower = sum(math.comb(n, j) for j in range(0, k + 1)) / (2**n)
    return {"one_sided_positive": upper, "two_sided": min(1.0, 2.0 * min(upper, lower))}


def run_jobs(out_root: Path, limit_runs: int | None, rerun: bool, jobs: int) -> None:
    gate = load_gate_module()
    manifest_path = out_root / "gate_run_manifest.csv"
    if not manifest_path.exists():
        build_manifest(out_root)
    manifest = pd.read_csv(manifest_path)
    rows = list(manifest.to_dict("records"))
    if limit_runs and limit_runs > 0:
        rows = rows[:limit_runs]

    def run_one(rec: dict[str, Any], idx: int, total: int) -> dict[str, Any]:
        out_dir = Path(rec["output_dir"])
        summary = out_dir / "phase2_summary.csv"
        if summary.exists() and not rerun:
            msg = f"[skip {idx}/{total}] {rec['run']} already complete"
            print(msg, flush=True)
            return {"run": rec["run"], "status": "skipped", "returncode": 0, "elapsed_sec": 0.0}
        if rerun and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = strict_phase2_command(
            gate=gate,
            out_dir=out_dir,
            currency=str(rec["currency"]),
            month=str(rec["month"]),
            seed=int(rec["seed"]),
            danger_slices=int(rec.get("raw_slice_count", gate.DANGER_SLICES)),
        )
        (out_dir / "gate_command.json").write_text(
            json.dumps({"command": cmd, "record": rec}, indent=2), encoding="utf-8"
        )
        print(f"[run {idx}/{total}] {rec['run']}", flush=True)
        t0 = time.time()
        with (out_dir / "gate_stdout.log").open("w", encoding="utf-8") as stdout, (
            out_dir / "gate_stderr.log"
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
    gate = load_gate_module()
    audit_dir = out_root / "_analysis_fit_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(out_root / "gate_run_manifest.csv")
    rows: list[dict[str, Any]] = []
    for _, rec in manifest.iterrows():
        run_dir = Path(rec["output_dir"])
        model = json.loads((run_dir / "nnproj_oos_model_summary.json").read_text(encoding="utf-8"))
        info = model.get("model_info", {})
        surf = pd.read_parquet(run_dir / "repaired_surfaces.parquet")
        nn = gate.filter_to_run_analysis_slices(surf[surf["arm"] == "NNPROJ"].copy(), rec)
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
                "oos_guard_pass": info.get("oos_guard_pass", np.nan),
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
    gate = load_gate_module()
    phase2 = gate.load_phase2_module()
    manifest_path = out_root / "gate_run_manifest.csv"
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
        window, pairs = gate.summarize_window(phase2, group, fit_by_run)
        window_rows.append(window)
        if not pairs.empty:
            pair_rows.append(pairs)

    window_df = pd.DataFrame(window_rows).sort_values(["month", "currency", "selection"]).reset_index(drop=True)
    pair_df = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()

    out_root.mkdir(parents=True, exist_ok=True)
    window_df.to_csv(out_root / "gate_window_summary.csv", index=False)
    pair_df.to_csv(out_root / "gate_seed_pair_summary.csv", index=False)

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
    decision = {
        "runs_manifested": int(len(manifest)),
        "runs_completed": int(
            manifest["output_dir"].map(lambda p: Path(p, "phase2_summary.csv").exists()).sum()
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
                decision[f"median_{col}"] = float(vals.median())
                if "excess" in col:
                    decision[f"positive_{col}_currency_months"] = f"{int((vals > 0).sum())}/{int(len(vals))}"

    (out_root / "gate_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_markdown_summary(out_root, decision)
    return window_df, pair_df, calendar_df


def write_markdown_summary(out_root: Path, decision: dict[str, Any]) -> None:
    lines = [
        "# 18-Month Top-Selection NN-Proj Seed Multiplicity Audit",
        "",
        f"- Manifested runs: {decision['runs_manifested']}",
        f"- Completed runs: {decision['runs_completed']}",
        f"- Windows: {decision['windows']} currency-month cells",
        f"- Calendar months: {decision['calendar_months']}",
        (
            "- Common-support D2 excess positive: "
            f"{decision['common_support_d2_positive_currency_months']} currency-month cells; "
            f"{decision['common_support_d2_positive_calendar_months']} calendar months"
        ),
        (
            "- Calendar-month sign test: "
            f"one-sided p={decision['calendar_month_sign_test']['one_sided_positive']:.6g}; "
            f"two-sided p={decision['calendar_month_sign_test']['two_sided']:.6g}"
        ),
        (
            "- Median common-support p90 D2 gap: "
            f"{decision['median_common_support_seed_pair_p90_d2_gap']:.6g}; "
            "matched L1/L2 control: "
            f"{decision['median_common_support_l1_l2_control_p90_d2_gap']:.6g}; "
            "excess: "
            f"{decision['median_common_support_d2_excess']:.6g}"
        ),
        (
            "- All-seed common support share: "
            f"median {decision['median_all_seed_common_support_share']:.3f}; "
            f"min {decision['min_all_seed_common_support_share']:.3f}"
        ),
        (
            "- Worst seed held-out spread-scaled price RMSE above best: "
            f"median {decision['median_worst_seed_price_rmse_pct_above_best']:.2f}%; "
            f"max {decision['max_worst_seed_price_rmse_pct_above_best']:.2f}%"
        ),
        "",
        "Outputs:",
        "- gate_run_manifest.csv",
        "- gate_window_summary.csv",
        "- gate_seed_pair_summary.csv",
        "- calendar_month_summary.csv",
        "- gate_decision.json",
    ]
    (out_root / "GATE_SUMMARY_18M_TOP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["manifest", "amend", "run", "audit", "summarize", "stage"])
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
    elif args.command == "amend":
        manifest = amend_existing_manifest(out_root)
        print(f"Amended manifest with {len(manifest)} runs at analysis cap {seed_slice_cap()}")
    elif args.command == "run":
        run_jobs(out_root, args.limit_runs, args.rerun, args.jobs)
    elif args.command == "audit":
        audit_fits(out_root)
    elif args.command == "summarize":
        summarize(out_root)
    elif args.command == "stage":
        manifest_path = out_root / "gate_run_manifest.csv"
        if not manifest_path.exists():
            manifest = build_manifest(out_root)
            print(f"Wrote manifest with {len(manifest)} runs to {out_root}")
        run_jobs(out_root, args.limit_runs, args.rerun, args.jobs)
        audit_fits(out_root)
        summarize(out_root)


if __name__ == "__main__":
    main()
