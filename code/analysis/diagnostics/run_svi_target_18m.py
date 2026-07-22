#!/usr/bin/env python3
"""Run and aggregate the 18-month SVI-target repair check."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
from artifact_paths import PANEL, RESULTS  # noqa: E402

SCRIPT = EXPERIMENT_ROOT / "code" / "analysis" / "learned_pipeline" / "repair_analysis.py"
DEFAULT_OUT = RESULTS / "svi_18m"

ANCHOR_MONTHS = ["2022-11", "2023-03", "2023-06", "2024-03", "2025-01", "2026-05"]
AVAILABLE_START = "2022-11"
AVAILABLE_END = "2026-05"
N_MONTHS = 18
REQUIRED = (
    "repair_summary.csv",
    "repair_arm_summary.csv",
    "arm_pair_summary.csv",
    "strike_pairwise_dispersion.csv",
    "book_pairwise_dispersion.csv",
    "var_style_pairwise.csv",
)


def selected_months() -> list[str]:
    available = pd.period_range(AVAILABLE_START, AVAILABLE_END, freq="M").astype(str).tolist()
    month_index = {month: i for i, month in enumerate(available)}
    selected = list(ANCHOR_MONTHS)
    for _ in range(N_MONTHS - len(selected)):
        best: tuple[tuple[int, int], str] | None = None
        for month in available:
            if month in selected:
                continue
            nearest = min(abs(month_index[month] - month_index[chosen]) for chosen in selected)
            key = (nearest, -month_index[month])
            if best is None or key > best[0]:
                best = (key, month)
        assert best is not None
        selected.append(best[1])
    return sorted(selected, key=month_index.__getitem__)


def run_name(currency: str, month: str, selection: str) -> str:
    return f"{currency}_{month}_{selection}50_svi"


def complete_run(out_dir: Path) -> bool:
    return all((out_dir / filename).exists() for filename in REQUIRED)


def command(out_dir: Path, currency: str, month: str, selection: str) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--call-equivalent-panel",
        str(PANEL),
        "--output-dir",
        str(out_dir),
        "--currency",
        currency,
        "--month",
        month,
        "--snapshot-hour",
        "12",
        "--dte-min-repair",
        "2",
        "--dte-max-repair",
        "30",
        "--dte-min-marking",
        "0",
        "--dte-max-marking",
        "31",
        "--marking-extra-days",
        "1",
        "--price-floor",
        "0.0005",
        "--strike-tol",
        "0.02",
        "--min-strikes",
        "7",
        "--arms",
        "L1",
        "L2",
        "SVI",
        "--smooth-rhos",
        "10",
        "--qp-backend",
        "cvxpy",
        "--qp-solver",
        "auto",
        "--svi-max-nfev",
        "5000",
        "--ranking-score-method",
        "relspread_h2",
        "--ranking-slice-agg",
        "p90",
        "--selected-slices",
        "50",
        "--slice-selection",
        selection,
        "--random-state",
        "20260605",
        "--no-plots",
    ]


def run_one(out_root: Path, logs: Path, currency: str, month: str, selection: str, *, force: bool = False) -> str:
    out_dir = out_root / run_name(currency, month, selection)
    if complete_run(out_dir) and not force:
        return f"[skip] {out_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = command(out_dir, currency, month, selection)
    log_path = logs / f"{out_dir.name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{out_dir.name} failed with exit code {proc.returncode}; see {log_path}")
    if not complete_run(out_dir):
        missing = [filename for filename in REQUIRED if not (out_dir / filename).exists()]
        raise FileNotFoundError(f"{out_dir.name} missing outputs after run: {missing}")
    return f"[done] {out_dir.name}"


def run_panel(out_root: Path, *, force: bool = False, jobs: int = 1) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    logs = out_root / "_logs"
    logs.mkdir(exist_ok=True)
    months = selected_months()
    specs = [(currency, month, selection) for currency in ("BTC", "ETH") for month in months for selection in ("top", "random", "bottom")]
    jobs = max(1, int(jobs))
    if jobs == 1:
        for currency, month, selection in specs:
            name = run_name(currency, month, selection)
            print(f"[run] {name}", flush=True)
            print(run_one(out_root, logs, currency, month, selection, force=force), flush=True)
        return

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        future_map = {
            pool.submit(run_one, out_root, logs, currency, month, selection, force=force): run_name(currency, month, selection)
            for currency, month, selection in specs
        }
        for future in as_completed(future_map):
            print(future.result(), flush=True)


def read_run_csv(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / filename
    df = pd.read_csv(path)
    df["run"] = run_dir.name
    parts = run_dir.name.split("_")
    df["currency_from_run"] = parts[0]
    df["month_from_run"] = parts[1]
    df["selection_from_run"] = parts[2].replace("50", "")
    return df


def pair_label(row: pd.Series) -> str:
    return "__".join(sorted([str(row["arm_a"]), str(row["arm_b"])]))


def aggregate(out_root: Path) -> Path:
    agg = out_root / "_aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(p for p in out_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    complete = [p for p in run_dirs if complete_run(p)]
    expected = 2 * len(selected_months()) * 3
    if len(complete) != expected:
        raise RuntimeError(f"Expected {expected} complete SVI runs, found {len(complete)}.")

    repair_summary = pd.concat([read_run_csv(p, "repair_summary.csv") for p in complete], ignore_index=True)
    arms = pd.concat([read_run_csv(p, "repair_arm_summary.csv") for p in complete], ignore_index=True)
    pairs = pd.concat([read_run_csv(p, "arm_pair_summary.csv") for p in complete], ignore_index=True)
    var = pd.concat([read_run_csv(p, "var_style_pairwise.csv") for p in complete], ignore_index=True)

    repair_summary.to_csv(agg / "svi_repair_summary.csv", index=False)
    arms.to_csv(agg / "svi_arm_summary.csv", index=False)
    pairs.to_csv(agg / "svi_pair_summary_all.csv", index=False)
    var.to_csv(agg / "svi_var_style_pairwise_all.csv", index=False)

    strike = pairs[pairs["level"].eq("strike")].copy()
    strike["pair"] = strike.apply(pair_label, axis=1)
    run_contrast = strike.pivot_table(index="run", columns="pair", values="p90_gamma_diff_scaled", aggfunc="first")
    for col in ("L1__L2", "L2__SVI", "L2__SMOOTH_RHO_10", "SMOOTH_RHO_10__SVI"):
        if col not in run_contrast.columns:
            run_contrast[col] = np.nan
    run_contrast["svi_minus_l1l2"] = run_contrast["L2__SVI"] - run_contrast["L1__L2"]
    run_contrast["svi_minus_smooth10"] = run_contrast["L2__SVI"] - run_contrast["L2__SMOOTH_RHO_10"]
    run_contrast.reset_index().to_csv(agg / "svi_run_level_contrasts.csv", index=False)

    key_pairs = (
        strike.groupby("pair", as_index=False)
        .agg(
            n_runs=("run", "nunique"),
            median_p90_spot_gamma_gap=("p90_gamma_diff_scaled", "median"),
            min_p90_spot_gamma_gap=("p90_gamma_diff_scaled", "min"),
            max_p90_spot_gamma_gap=("p90_gamma_diff_scaled", "max"),
            median_p90_d2_gap=("p90_cxx_diff_scaled", "median"),
        )
        .sort_values("pair")
    )
    key_pairs.to_csv(agg / "svi_strike_key_pairs.csv", index=False)

    book = pairs[pairs["level"].eq("book")].copy()
    book["pair"] = book.apply(pair_label, axis=1)
    book_summary = (
        book.groupby(["pair", "book"], as_index=False)
        .agg(
            n_cells=("run", "size"),
            median_p90_hedge_gap=("p90_hedge_slippage_scaled", "median"),
            median_p90_spot_gamma_gap=("p90_gamma_rr", "median"),
        )
        .sort_values(["pair", "book"])
    )
    book_summary.to_csv(agg / "svi_book_key_pairs.csv", index=False)

    arm_summary = (
        arms.groupby("arm", as_index=False)
        .agg(
            attempts=("attempts", "sum"),
            successes=("successes", "sum"),
            median_success_rate=("success_rate", "median"),
            max_bad_box_attempt_rate=("bad_box_attempt_rate", "max"),
            median_optimal_inaccurate_rate=("optimal_inaccurate_rate", "median"),
            max_optimal_inaccurate_rate=("optimal_inaccurate_rate", "max"),
        )
        .sort_values("arm")
    )
    arm_summary["success_rate"] = arm_summary["successes"] / arm_summary["attempts"]
    arm_summary.to_csv(agg / "svi_arm_success_summary.csv", index=False)

    summary = {
        "n_runs": int(len(complete)),
        "n_candidate_slices": int(repair_summary["candidate_slices"].sum()),
        "median_complete_slice_rate": float(repair_summary["complete_slice_rate"].median()),
        "min_complete_slice_rate": float(repair_summary["complete_slice_rate"].min()),
        "l2_svi_median_p90_spot_gamma_gap": float(run_contrast["L2__SVI"].median()),
        "l1_l2_median_p90_spot_gamma_gap": float(run_contrast["L1__L2"].median()),
        "l2_smooth10_median_p90_spot_gamma_gap": float(run_contrast["L2__SMOOTH_RHO_10"].median()),
        "svi_exceeds_l1l2_runs": int((run_contrast["svi_minus_l1l2"] > 0).sum()),
        "svi_exceeds_smooth10_runs": int((run_contrast["svi_minus_smooth10"] > 0).sum()),
        "median_svi_minus_l1l2": float(run_contrast["svi_minus_l1l2"].median()),
        "median_svi_minus_smooth10": float(run_contrast["svi_minus_smooth10"].median()),
    }
    pd.DataFrame([summary]).to_csv(agg / "svi_18m_audit_summary.csv", index=False)

    lines = [
        "# SVI-target 18-calendar-month audit",
        "",
        f"- Runs completed: {summary['n_runs']}",
        f"- Candidate slices: {summary['n_candidate_slices']}; median complete-slice rate {summary['median_complete_slice_rate']:.3f}; min {summary['min_complete_slice_rate']:.3f}.",
    ]
    for _, row in arm_summary.iterrows():
        lines.append(
            f"- {row['arm']}: {int(row['successes'])}/{int(row['attempts'])} successes ({row['success_rate']:.3f})."
        )
    lines.extend(
        [
            "",
            "## Strike-level repair risk",
            f"- L1/L2 median p90 scaled spot-gamma gap: {summary['l1_l2_median_p90_spot_gamma_gap']:.4f}.",
            f"- L2/Smooth10 median p90 scaled spot-gamma gap: {summary['l2_smooth10_median_p90_spot_gamma_gap']:.4f}.",
            f"- L2/SVI median p90 scaled spot-gamma gap: {summary['l2_svi_median_p90_spot_gamma_gap']:.4f}.",
            f"- L2/SVI exceeds L1/L2 in {summary['svi_exceeds_l1l2_runs']}/{summary['n_runs']} runs; median excess {summary['median_svi_minus_l1l2']:.4f}.",
            f"- L2/SVI exceeds L2/Smooth10 in {summary['svi_exceeds_smooth10_runs']}/{summary['n_runs']} runs; median excess {summary['median_svi_minus_smooth10']:.4f}.",
            "",
        ]
    )
    (agg / "SVI_18M_AUDIT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return agg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="Number of independent runs to execute in parallel.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not SCRIPT.exists():
        raise FileNotFoundError(SCRIPT)
    if not PANEL.exists():
        raise FileNotFoundError(PANEL)
    if not args.aggregate_only:
        run_panel(args.output_root, force=args.force, jobs=args.jobs)
    agg = aggregate(args.output_root)
    print(f"Wrote SVI-target 18m aggregate to {agg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
