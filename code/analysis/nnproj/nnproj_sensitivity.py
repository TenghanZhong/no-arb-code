#!/usr/bin/env python3
"""Run NNProj seed and architecture sensitivity analyses."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
from pairwise_metrics import apply_pair_specific_gamma_scale  # noqa: E402


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[3]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
from artifact_paths import PANEL, RESULTS  # noqa: E402

ROOT = EXPERIMENT_ROOT
REPAIR_SCRIPT = ANALYSIS_ROOT / "learned_pipeline" / "repair_analysis.py"
AUDIT = ANALYSIS_ROOT / "nnproj" / "audit_nnproj_heldout_fit.py"
DEFAULT_OUT = RESULTS / "nn_seed_18m"
SLICE_COLS = ["currency", "date", "expiration_date"]
D2_GAP_COL = "cxx_diff_scaled"
SPOT_GAMMA_GAP_COL = "gamma_diff_scaled"

BASE_RANDOM_SEED = 20260605
NN_SEEDS = [20260605, 20260606, 20260607, 20260608, 20260609]
SELECTED_SLICES = 49
CORE_WINDOWS = [
    ("BTC", "2023-03", "top"),
    ("ETH", "2023-03", "top"),
    ("BTC", "2026-05", "top"),
]
EXTENDED_WINDOWS = [
    ("BTC", "2023-03", "top"),
    ("ETH", "2023-03", "top"),
    ("BTC", "2023-06", "top"),
    ("ETH", "2023-06", "top"),
    ("ETH", "2025-01", "top"),
    ("BTC", "2026-05", "top"),
]
ARCHITECTURES = ["32,32", "64,64", "64,64,64"]


def load_repair_module():
    spec = importlib.util.spec_from_file_location("nnproj_repair_mod", REPAIR_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import repair script: {REPAIR_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def repair_config_from_json(mod, run_cfg: Dict):
    kwargs = {}
    for f in fields(mod.RepairConfig):
        if f.name in run_cfg:
            val = run_cfg[f.name]
            if f.name == "arms" and val is not None:
                val = tuple(str(x).upper() for x in val)
            kwargs[f.name] = val
    return mod.RepairConfig(**kwargs)


def hidden_tag(hidden: str) -> str:
    return "x".join(part.strip() for part in str(hidden).split(",") if part.strip())


def run_name(currency: str, month: str, selection: str, hidden: str, seed: int) -> str:
    return f"{currency}_{month}_{selection}{SELECTED_SLICES}_nnproj_arch{hidden_tag(hidden)}_seed{seed}"


def repair_command(out_dir: Path, currency: str, month: str, selection: str, hidden: str, nn_seed: int) -> List[str]:
    return [
        sys.executable,
        str(REPAIR_SCRIPT),
        "--call-equivalent-panel", str(PANEL),
        "--output-dir", str(out_dir),
        "--currency", currency,
        "--snapshot-hour", "12",
        "--month", month,
        "--dte-min-repair", "2",
        "--dte-max-repair", "30",
        "--dte-min-marking", "0",
        "--dte-max-marking", "31",
        "--marking-extra-days", "1",
        "--arms", "L1", "L2",
        "--smooth-rhos", "0.01", "0.1", "1", "10",
        "--include-ml-arm",
        "--nnproj-train-mode", "past",
        "--nnproj-train-lookback-months", "12",
        "--nnproj-embargo-days", "7",
        "--nnproj-min-train-months", "2",
        "--nnproj-hidden", hidden,
        "--nnproj-random-state", str(nn_seed),
        "--ranking-score-method", "relspread_h2",
        "--ranking-slice-agg", "p90",
        "--selected-slices", str(SELECTED_SLICES),
        "--slice-selection", selection,
        "--random-seed", str(BASE_RANDOM_SEED),
        "--no-plots",
    ]


def analysis_specs(analysis_set: str) -> List[Dict]:
    if analysis_set == "core":
        rows = []
        for currency, month, selection in CORE_WINDOWS:
            for seed in NN_SEEDS:
                rows.append({"analysis_set": analysis_set, "currency": currency, "month": month, "selection": selection, "hidden": "64,64", "seed": seed})
        return rows
    if analysis_set == "extended":
        rows = []
        for currency, month, selection in EXTENDED_WINDOWS:
            for seed in NN_SEEDS:
                rows.append({"analysis_set": analysis_set, "currency": currency, "month": month, "selection": selection, "hidden": "64,64", "seed": seed})
        return rows
    if analysis_set == "architecture":
        rows = []
        for currency, month, selection in CORE_WINDOWS:
            for hidden in ARCHITECTURES:
                for seed in NN_SEEDS[:3]:
                    rows.append({"analysis_set": analysis_set, "currency": currency, "month": month, "selection": selection, "hidden": hidden, "seed": seed})
        return rows
    raise ValueError(f"Unknown analysis set: {analysis_set}")


def write_manifest(analysis_dir: Path, rows: List[Dict]) -> pd.DataFrame:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        name = run_name(row["currency"], row["month"], row["selection"], row["hidden"], int(row["seed"]))
        out_dir = analysis_dir / name
        cmd = repair_command(out_dir, row["currency"], row["month"], row["selection"], row["hidden"], int(row["seed"]))
        rec = dict(row)
        rec.update({
            "run": name,
            "output_dir": str(out_dir),
            "repair_script": str(REPAIR_SCRIPT),
            "synthetic_call_panel": str(PANEL),
            "slice_random_seed": BASE_RANDOM_SEED,
            "command": " ".join(cmd),
        })
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(analysis_dir / "run_manifest.csv", index=False)
    (analysis_dir / "analysis_universe.json").write_text(json.dumps({
        "analysis_dir": str(analysis_dir),
        "repair_script": str(REPAIR_SCRIPT),
        "synthetic_call_panel": str(PANEL),
        "slice_random_seed": BASE_RANDOM_SEED,
        "nn_seeds": NN_SEEDS,
        "core_windows": CORE_WINDOWS,
        "extended_windows": EXTENDED_WINDOWS,
        "architectures": ARCHITECTURES,
        "fixed_features_and_projection": "Uses the shared learned-repair analysis pipeline",
    }, indent=2), encoding="utf-8")
    return df


def run_one(rec: Dict, idx: int, total: int, rerun: bool = False) -> Dict:
    out_dir = Path(rec["output_dir"])
    done = out_dir / "repair_summary.csv"
    if done.exists() and not rerun:
        print(f"[skip {idx}/{total}] {rec['run']} already complete", flush=True)
        return {**rec, "status": "skipped_existing", "returncode": 0, "elapsed_sec": 0.0}
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = repair_command(out_dir, rec["currency"], rec["month"], rec["selection"], rec["hidden"], int(rec["seed"]))
    print(f"[run {idx}/{total}] {rec['run']}", flush=True)
    t0 = time.time()
    with (out_dir / "command.json").open("w", encoding="utf-8") as f:
        json.dump({"command": cmd, "record": rec}, f, indent=2)
    with (out_dir / "stdout.log").open("w", encoding="utf-8") as so, (out_dir / "stderr.log").open("w", encoding="utf-8") as se:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=so, stderr=se, text=True)
    elapsed = time.time() - t0
    status = "ok" if proc.returncode == 0 else "failed"
    print(f"[{status}] {rec['run']} returncode={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    out = {**rec, "status": status, "returncode": proc.returncode, "elapsed_sec": elapsed}
    if proc.returncode != 0:
        out["error"] = f"Run failed; see {out_dir / 'stderr.log'}"
    return out


def run_manifest(analysis_dir: Path, manifest: pd.DataFrame, limit: int = 0, rerun: bool = False, jobs: int = 1) -> None:
    log_rows = []
    rows = manifest.to_dict("records")
    if limit > 0:
        rows = rows[:limit]
    jobs = max(1, int(jobs))
    if jobs == 1:
        for i, rec in enumerate(rows, start=1):
            result = run_one(rec, i, len(rows), rerun=rerun)
            log_rows.append(result)
            pd.DataFrame(log_rows).to_csv(analysis_dir / "execution_log.csv", index=False)
            if result.get("returncode") != 0:
                raise SystemExit(result.get("error", f"Run failed: {result.get('run')}"))
        return
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(run_one, rec, i, len(rows), rerun): rec for i, rec in enumerate(rows, start=1)}
        for fut in as_completed(futures):
            result = fut.result()
            log_rows.append(result)
            pd.DataFrame(log_rows).sort_values(["currency", "month", "selection", "seed"]).to_csv(analysis_dir / "execution_log.csv", index=False)
            if result.get("returncode") != 0:
                raise SystemExit(result.get("error", f"Run failed: {result.get('run')}"))


def run_fit_audit(analysis_dir: Path) -> Path:
    out = analysis_dir / "heldout_fit"
    cmd = [
        sys.executable,
        str(AUDIT),
        "--results-root", str(analysis_dir),
        "--repair-script", str(REPAIR_SCRIPT),
        "--call-equivalent-panel", str(PANEL),
        "--output-dir", str(out),
        "--eval-source", "selected",
    ]
    print("[audit] held-out fit audit", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return out / "nnproj_heldout_fit_by_run.csv"


def is_seed_arm(a: str) -> bool:
    return str(a).startswith("NN_SEED_")


def is_arch_arm(a: str) -> bool:
    return str(a).startswith("NN_ARCH_")


def seed_arm_name(rec) -> str:
    return f"NN_SEED_{int(rec['seed'])}"


def arch_arm_name(rec) -> str:
    return f"NN_ARCH_{hidden_tag(rec['hidden'])}_SEED_{int(rec['seed'])}"


def architecture_pair_kind(arch_a: str, seed_a: int, arch_b: str, seed_b: int) -> str:
    if str(arch_a) == str(arch_b):
        return "within_arch"
    if int(seed_a) == int(seed_b):
        return "same_seed_between_arch"
    return "cross_seed_between_arch"


def pair_mask(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    return ((df["arm_a"] == a) & (df["arm_b"] == b)) | ((df["arm_a"] == b) & (df["arm_b"] == a))


def p90(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(np.nanpercentile(vals, 90)) if len(vals) else np.nan


def median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(np.nanmedian(vals)) if len(vals) else np.nan


def slice_key_tuples(df: pd.DataFrame) -> set[tuple]:
    if df.empty or not set(SLICE_COLS).issubset(df.columns):
        return set()
    return set(map(tuple, df[SLICE_COLS].drop_duplicates().itertuples(index=False, name=None)))


def filter_to_slice_keys(df: pd.DataFrame, keys: set[tuple]) -> pd.DataFrame:
    if df.empty or not keys:
        return df.iloc[0:0].copy()
    vals = list(map(tuple, df[SLICE_COLS].itertuples(index=False, name=None)))
    return df[pd.Series(vals, index=df.index).isin(keys)].copy()


def analysis_slice_keys(rec: pd.Series | Dict) -> set[tuple]:
    run_dir = Path(rec["output_dir"])
    panel = pd.read_parquet(run_dir / "repair_panel.parquet")
    if "core_repair_slice" in panel.columns:
        panel = panel[panel["core_repair_slice"].astype(bool)].copy()
    cap = pd.to_numeric(pd.Series([rec.get("analysis_slice_cap", np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(cap):
        if "slice_rank_desc" not in panel.columns:
            raise ValueError(f"Missing slice_rank_desc in {run_dir}")
        ranks = pd.to_numeric(panel["slice_rank_desc"], errors="coerce")
        panel = panel[ranks <= int(cap)].copy()
    return slice_key_tuples(panel)


def filter_to_run_analysis_slices(df: pd.DataFrame, rec: pd.Series | Dict) -> pd.DataFrame:
    return filter_to_slice_keys(df, analysis_slice_keys(rec))


def nnproj_support_sets(group: pd.DataFrame, arm_name_fn: Callable = seed_arm_name) -> Dict:
    attempts: Dict[str, set[tuple]] = {}
    successes: Dict[str, set[tuple]] = {}
    for _, rec in group.iterrows():
        arm = arm_name_fn(rec)
        diag_path = Path(rec["output_dir"]) / "repair_diagnostics.csv"
        diag = pd.read_csv(diag_path)
        nn = filter_to_run_analysis_slices(diag[diag["arm"] == "NNPROJ"].copy(), rec)
        attempts[arm] = slice_key_tuples(nn)
        successes[arm] = slice_key_tuples(nn[nn["success"].astype(bool)])
    attempt_sets = list(attempts.values())
    success_sets = list(successes.values())
    attempt_union = set().union(*attempt_sets) if attempt_sets else set()
    attempt_intersection = set.intersection(*attempt_sets) if attempt_sets else set()
    success_intersection = set.intersection(*success_sets) if success_sets else set()
    return {
        "attempts": attempts,
        "successes": successes,
        "attempt_union": attempt_union,
        "attempt_intersection": attempt_intersection,
        "success_intersection": success_intersection,
        "attempt_sets_identical": all(s == attempt_sets[0] for s in attempt_sets[1:]) if attempt_sets else True,
    }


def pair_support_size(support: Dict, a: str, b: str) -> int:
    return len(support["successes"].get(a, set()) & support["successes"].get(b, set()))


def combine_window_outputs(mod, group: pd.DataFrame, arm_name_fn: Callable = seed_arm_name) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    base_dir = Path(group.iloc[0]["output_dir"])
    run_cfg = json.loads((base_dir / "repair_config.json").read_text(encoding="utf-8"))
    cfg = repair_config_from_json(mod, run_cfg)

    greek_parts = []
    book_parts = []
    base_rec = group.iloc[0]
    base_g = filter_to_run_analysis_slices(pd.read_parquet(base_dir / "greeks.parquet"), base_rec)
    base_b = filter_to_run_analysis_slices(pd.read_parquet(base_dir / "book_outputs.parquet"), base_rec)
    baseline_arms = {"L1", "L2", "SMOOTH_RHO_10"}
    greek_parts.append(base_g[base_g["arm"].isin(baseline_arms)].copy())
    book_parts.append(base_b[base_b["arm"].isin(baseline_arms)].copy())

    for _, rec in group.iterrows():
        arm = arm_name_fn(rec)
        rd = Path(rec["output_dir"])
        g = pd.read_parquet(rd / "greeks.parquet")
        b = pd.read_parquet(rd / "book_outputs.parquet")
        g = filter_to_run_analysis_slices(g[g["arm"] == "NNPROJ"].copy(), rec)
        b = filter_to_run_analysis_slices(b[b["arm"] == "NNPROJ"].copy(), rec)
        g["arm"] = arm
        b["arm"] = arm
        greek_parts.append(g)
        book_parts.append(b)

    return pd.concat(greek_parts, ignore_index=True), pd.concat(book_parts, ignore_index=True), cfg


def summarize_window(mod, group: pd.DataFrame, fit_by_run: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    key_cols = ["currency", "month", "selection", "hidden"]
    key = {c: group.iloc[0][c] for c in key_cols}
    greeks, books, cfg = combine_window_outputs(mod, group)
    support = nnproj_support_sets(group)

    strike = apply_pair_specific_gamma_scale(
        mod.compute_strike_pairwise_dispersion(greeks, cfg), greeks, cfg
    )
    book = mod.compute_book_pairwise_dispersion(books)
    var_args = SimpleNamespace(
        var_alphas=(0.95, 0.99),
        var_min_losses=10,
        var_absolute_limits=(0.05, 0.10, 0.25),
        var_capital_quantiles=(0.50, 0.75, 0.90),
        var_baseline_arm="L2",
        var_baseline_multipliers=(0.90, 1.00, 1.10),
    )
    _, var_pair, _, _ = mod.compute_var_style_outputs(books, var_args)

    seed_arms = sorted([a for a in greeks["arm"].dropna().unique() if is_seed_arm(a)])
    seed_pairs = []
    for i, a in enumerate(seed_arms):
        for b in seed_arms[i + 1:]:
            sp = strike[pair_mask(strike, a, b)]
            bp = book[pair_mask(book, a, b)] if not book.empty else pd.DataFrame()
            vp = var_pair[pair_mask(var_pair, a, b)] if not var_pair.empty else pd.DataFrame()
            seed_pairs.append({
                **key,
                "arm_a": a,
                "arm_b": b,
                "pair_common_support_slices": pair_support_size(support, a, b),
                "pair_common_support_share": pair_support_size(support, a, b) / max(len(support["attempt_union"]), 1),
                "p90_d2_curvature_gap": p90(sp.get(D2_GAP_COL, pd.Series(dtype=float))),
                "p90_curvature_gap": p90(sp.get(D2_GAP_COL, pd.Series(dtype=float))),
                "p90_spot_gamma_gap": p90(sp.get(SPOT_GAMMA_GAP_COL, pd.Series(dtype=float))),
                "p90_hedge_gap_impact": p90(bp.get("hedge_slippage_scaled", pd.Series(dtype=float))) if not bp.empty else np.nan,
                "p90_var_common_gap": p90(vp.loc[vp.get("loss_type", "") == "common", "var_a0p95_diff_scaled"]) if not vp.empty else np.nan,
                "p90_var_pipeline_gap": p90(vp.loc[vp.get("loss_type", "") == "pipeline", "var_a0p95_diff_scaled"]) if not vp.empty else np.nan,
            })
    seed_pair_df = pd.DataFrame(seed_pairs)

    control = strike[pair_mask(strike, "L1", "L2")]
    smooth = strike[pair_mask(strike, "L2", "SMOOTH_RHO_10")]
    control_p90 = p90(control[D2_GAP_COL]) if not control.empty else np.nan
    smooth_p90 = p90(smooth[D2_GAP_COL]) if not smooth.empty else np.nan
    control_spot_gamma_p90 = p90(control[SPOT_GAMMA_GAP_COL]) if not control.empty else np.nan
    smooth_spot_gamma_p90 = p90(smooth[SPOT_GAMMA_GAP_COL]) if not smooth.empty else np.nan
    seed_median_p90 = median(seed_pair_df["p90_curvature_gap"]) if not seed_pair_df.empty else np.nan
    seed_median_spot_gamma_p90 = median(seed_pair_df["p90_spot_gamma_gap"]) if not seed_pair_df.empty else np.nan

    common_greeks = filter_to_slice_keys(greeks, support["success_intersection"])
    common_strike = apply_pair_specific_gamma_scale(
        mod.compute_strike_pairwise_dispersion(common_greeks, cfg), common_greeks, cfg
    )
    common_seed_pairs = []
    for i, a in enumerate(seed_arms):
        for b in seed_arms[i + 1:]:
            sp = common_strike[pair_mask(common_strike, a, b)] if not common_strike.empty else pd.DataFrame()
            common_seed_pairs.append({
                "arm_a": a,
                "arm_b": b,
                "p90_d2_curvature_gap": p90(sp.get(D2_GAP_COL, pd.Series(dtype=float))),
                "p90_spot_gamma_gap": p90(sp.get(SPOT_GAMMA_GAP_COL, pd.Series(dtype=float))),
            })
    common_seed_pair_df = pd.DataFrame(common_seed_pairs)
    common_control = common_strike[pair_mask(common_strike, "L1", "L2")] if not common_strike.empty else pd.DataFrame()
    common_control_p90 = p90(common_control[D2_GAP_COL]) if not common_control.empty else np.nan
    common_seed_median_p90 = median(common_seed_pair_df["p90_d2_curvature_gap"]) if not common_seed_pair_df.empty else np.nan

    # Group seeds by spread-scaled held-out RMSE.
    run_names = set(group["run"].astype(str))
    fit = fit_by_run[fit_by_run["run"].astype(str).isin(run_names)].copy()
    fit["seed"] = fit["run"].map(group.set_index("run")["seed"].to_dict())
    fit["price_rmse_scaled"] = pd.to_numeric(fit.get("oos_price_rmse_scaled"), errors="coerce")
    min_loss = float(fit["price_rmse_scaled"].min()) if fit["price_rmse_scaled"].notna().any() else np.nan
    max_loss = float(fit["price_rmse_scaled"].max()) if fit["price_rmse_scaled"].notna().any() else np.nan
    fit10 = set(fit.loc[fit["price_rmse_scaled"] <= 1.10 * min_loss, "seed"].dropna().astype(int).map(lambda s: f"NN_SEED_{s}")) if np.isfinite(min_loss) else set()
    fit20 = set(fit.loc[fit["price_rmse_scaled"] <= 1.20 * min_loss, "seed"].dropna().astype(int).map(lambda s: f"NN_SEED_{s}")) if np.isfinite(min_loss) else set()

    def fit_pair_median(pair_df: pd.DataFrame, fitset: set[str], col: str = "p90_d2_curvature_gap") -> float:
        if pair_df.empty or len(fitset) < 2:
            return np.nan
        m = pair_df["arm_a"].isin(fitset) & pair_df["arm_b"].isin(fitset)
        return median(pair_df.loc[m, col])

    seed_med_10 = fit_pair_median(seed_pair_df, fit10)
    seed_med_20 = fit_pair_median(seed_pair_df, fit20)
    common_seed_med_10 = fit_pair_median(common_seed_pair_df, fit10, "p90_d2_curvature_gap")
    common_seed_med_20 = fit_pair_median(common_seed_pair_df, fit20, "p90_d2_curvature_gap")

    leave_one = []
    for arm in seed_arms:
        m = (seed_pair_df["arm_a"] != arm) & (seed_pair_df["arm_b"] != arm)
        leave_one.append(median(seed_pair_df.loc[m, "p90_d2_curvature_gap"]) - control_p90)
    common_leave_one = []
    for arm in seed_arms:
        if common_seed_pair_df.empty:
            continue
        m = (common_seed_pair_df["arm_a"] != arm) & (common_seed_pair_df["arm_b"] != arm)
        common_leave_one.append(median(common_seed_pair_df.loc[m, "p90_d2_curvature_gap"]) - common_control_p90)

    diag_rows = []
    for _, rec in group.iterrows():
        d = pd.read_csv(Path(rec["output_dir"]) / "repair_diagnostics.csv")
        nn = filter_to_run_analysis_slices(d[d["arm"] == "NNPROJ"].copy(), rec)
        attempts = int(len(nn))
        successes = int(nn["success"].astype(bool).sum()) if attempts else 0
        diag_rows.append({
            "run": rec["run"],
            "seed": int(rec["seed"]),
            "nn_attempts": attempts,
            "nn_successes": successes,
            "nn_success_rate": successes / attempts if attempts else np.nan,
        })
    diag = pd.DataFrame(diag_rows)

    row = {
        **key,
        "n_seeds": int(len(seed_arms)),
        "nn_success_rate_min": float(diag["nn_success_rate"].min()),
        "nn_success_rate_median": float(diag["nn_success_rate"].median()),
        "n_seeds_success_ge_90pct": int((diag["nn_success_rate"] >= 0.90).sum()),
        "selected_slice_count": int(len(support["attempt_union"])),
        "all_seed_attempt_intersection_slices": int(len(support["attempt_intersection"])),
        "all_seed_attempt_sets_identical": bool(support["attempt_sets_identical"]),
        "all_seed_common_success_slices": int(len(support["success_intersection"])),
        "all_seed_common_success_share": float(len(support["success_intersection"]) / max(len(support["attempt_union"]), 1)),
        "control_l1_l2_p90_d2_curvature_gap": control_p90,
        "control_l1_l2_p90_curvature_gap": control_p90,
        "control_l1_l2_p90_spot_gamma_gap": control_spot_gamma_p90,
        "smooth_l2_s10_p90_d2_curvature_gap": smooth_p90,
        "smooth_l2_s10_p90_curvature_gap": smooth_p90,
        "smooth_l2_s10_p90_spot_gamma_gap": smooth_spot_gamma_p90,
        "seed_pair_median_p90_d2_curvature_gap": seed_median_p90,
        "seed_pair_median_p90_curvature_gap": seed_median_p90,
        "seed_pair_median_p90_spot_gamma_gap": seed_median_spot_gamma_p90,
        "seed_pair_excess_over_l1_l2": seed_median_p90 - control_p90 if np.isfinite(seed_median_p90) and np.isfinite(control_p90) else np.nan,
        "common_support_control_l1_l2_p90_d2_curvature_gap": common_control_p90,
        "common_support_seed_pair_median_p90_d2_curvature_gap": common_seed_median_p90,
        "common_support_seed_pair_excess_over_l1_l2": common_seed_median_p90 - common_control_p90 if np.isfinite(common_seed_median_p90) and np.isfinite(common_control_p90) else np.nan,
        "fit_best_price_rmse_scaled": min_loss,
        "fit_worst_price_rmse_scaled": max_loss,
        "fit_worst_over_best": max_loss / min_loss if np.isfinite(max_loss) and np.isfinite(min_loss) and min_loss > 0 else np.nan,
        "fit_worst_pct_above_best": 100.0 * (max_loss / min_loss - 1.0) if np.isfinite(max_loss) and np.isfinite(min_loss) and min_loss > 0 else np.nan,
        "fit_10pct_seed_count": int(len(fit10)),
        "fit_10pct_seed_pair_median_p90_d2_curvature_gap": seed_med_10,
        "fit_10pct_seed_pair_median_p90_curvature_gap": seed_med_10,
        "fit_10pct_seed_pair_excess": seed_med_10 - control_p90 if np.isfinite(seed_med_10) and np.isfinite(control_p90) else np.nan,
        "common_support_fit_10pct_seed_pair_median_p90_d2_curvature_gap": common_seed_med_10,
        "common_support_fit_10pct_seed_pair_excess": common_seed_med_10 - common_control_p90 if np.isfinite(common_seed_med_10) and np.isfinite(common_control_p90) else np.nan,
        "fit_20pct_seed_count": int(len(fit20)),
        "fit_20pct_seed_pair_median_p90_d2_curvature_gap": seed_med_20,
        "fit_20pct_seed_pair_median_p90_curvature_gap": seed_med_20,
        "fit_20pct_seed_pair_excess": seed_med_20 - control_p90 if np.isfinite(seed_med_20) and np.isfinite(control_p90) else np.nan,
        "common_support_fit_20pct_seed_pair_median_p90_d2_curvature_gap": common_seed_med_20,
        "common_support_fit_20pct_seed_pair_excess": common_seed_med_20 - common_control_p90 if np.isfinite(common_seed_med_20) and np.isfinite(common_control_p90) else np.nan,
        "min_leave_one_seed_excess": float(np.nanmin(leave_one)) if leave_one else np.nan,
        "common_support_min_leave_one_seed_excess": float(np.nanmin(common_leave_one)) if common_leave_one else np.nan,
        "median_seed_pair_p90_hedge_gap_impact": median(seed_pair_df["p90_hedge_gap_impact"]) if not seed_pair_df.empty else np.nan,
        "median_seed_pair_p90_var_common_gap": median(seed_pair_df["p90_var_common_gap"]) if not seed_pair_df.empty else np.nan,
        "median_seed_pair_p90_var_pipeline_gap": median(seed_pair_df["p90_var_pipeline_gap"]) if not seed_pair_df.empty else np.nan,
    }
    return row, seed_pair_df


def summarize_architecture_window(mod, group: pd.DataFrame, fit_by_run: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    key_cols = ["currency", "month", "selection"]
    key = {c: group.iloc[0][c] for c in key_cols}
    greeks, books, cfg = combine_window_outputs(mod, group, arch_arm_name)
    support = nnproj_support_sets(group, arch_arm_name)

    strike = mod.compute_strike_pairwise_dispersion(greeks, cfg)
    book = mod.compute_book_pairwise_dispersion(books)

    arm_meta = {}
    run_to_arm = {}
    for _, rec in group.iterrows():
        arm = arch_arm_name(rec)
        arm_meta[arm] = {
            "arch": hidden_tag(rec["hidden"]),
            "seed": int(rec["seed"]),
            "run": str(rec["run"]),
        }
        run_to_arm[str(rec["run"])] = arm

    arch_arms = sorted([a for a in greeks["arm"].dropna().unique() if is_arch_arm(a)])
    fit = fit_by_run[fit_by_run["run"].astype(str).isin(run_to_arm)].copy()
    fit["arm"] = fit["run"].astype(str).map(run_to_arm)
    fit["price_rmse_scaled"] = pd.to_numeric(fit.get("oos_price_rmse_scaled"), errors="coerce")
    loss_by_arm = fit.dropna(subset=["arm"]).set_index("arm")["price_rmse_scaled"].to_dict()
    min_loss = float(fit["price_rmse_scaled"].min()) if fit["price_rmse_scaled"].notna().any() else np.nan
    max_loss = float(fit["price_rmse_scaled"].max()) if fit["price_rmse_scaled"].notna().any() else np.nan
    fit10 = set(fit.loc[fit["price_rmse_scaled"] <= 1.10 * min_loss, "arm"].dropna()) if np.isfinite(min_loss) else set()
    fit20 = set(fit.loc[fit["price_rmse_scaled"] <= 1.20 * min_loss, "arm"].dropna()) if np.isfinite(min_loss) else set()

    pair_rows = []
    for i, a in enumerate(arch_arms):
        for b in arch_arms[i + 1:]:
            sp = strike[pair_mask(strike, a, b)]
            bp = book[pair_mask(book, a, b)] if not book.empty else pd.DataFrame()
            arch_a = arm_meta[a]["arch"]
            arch_b = arm_meta[b]["arch"]
            seed_a = arm_meta[a]["seed"]
            seed_b = arm_meta[b]["seed"]
            pair_kind = "within_arch" if arch_a == arch_b else "between_arch"
            pair_subkind = architecture_pair_kind(arch_a, seed_a, arch_b, seed_b)
            pair_rows.append({
                **key,
                "arm_a": a,
                "arm_b": b,
                "arch_a": arch_a,
                "arch_b": arch_b,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "pair_kind": pair_kind,
                "pair_subkind": pair_subkind,
                "pair_common_support_slices": pair_support_size(support, a, b),
                "pair_common_support_share": pair_support_size(support, a, b) / max(len(support["attempt_union"]), 1),
                "p90_d2_curvature_gap": p90(sp.get(D2_GAP_COL, pd.Series(dtype=float))),
                "p90_curvature_gap": p90(sp.get(D2_GAP_COL, pd.Series(dtype=float))),
                "p90_spot_gamma_gap": p90(sp.get(SPOT_GAMMA_GAP_COL, pd.Series(dtype=float))),
                "p90_hedge_gap_impact": p90(bp.get("hedge_slippage_scaled", pd.Series(dtype=float))) if not bp.empty else np.nan,
                "price_rmse_scaled_a": loss_by_arm.get(a, np.nan),
                "price_rmse_scaled_b": loss_by_arm.get(b, np.nan),
                "fit10_pair": a in fit10 and b in fit10,
                "fit20_pair": a in fit20 and b in fit20,
            })
    pair_df = pd.DataFrame(pair_rows)

    control = strike[pair_mask(strike, "L1", "L2")]
    smooth = strike[pair_mask(strike, "L2", "SMOOTH_RHO_10")]
    control_p90 = p90(control[D2_GAP_COL]) if not control.empty else np.nan
    smooth_p90 = p90(smooth[D2_GAP_COL]) if not smooth.empty else np.nan
    control_spot_gamma_p90 = p90(control[SPOT_GAMMA_GAP_COL]) if not control.empty else np.nan
    smooth_spot_gamma_p90 = p90(smooth[SPOT_GAMMA_GAP_COL]) if not smooth.empty else np.nan

    common_greeks = filter_to_slice_keys(greeks, support["success_intersection"])
    common_strike = mod.compute_strike_pairwise_dispersion(common_greeks, cfg)
    common_control = common_strike[pair_mask(common_strike, "L1", "L2")] if not common_strike.empty else pd.DataFrame()
    common_control_p90 = p90(common_control[D2_GAP_COL]) if not common_control.empty else np.nan

    common_pair_rows = []
    for i, a in enumerate(arch_arms):
        for b in arch_arms[i + 1:]:
            sp = common_strike[pair_mask(common_strike, a, b)] if not common_strike.empty else pd.DataFrame()
            arch_a = arm_meta[a]["arch"]
            arch_b = arm_meta[b]["arch"]
            seed_a = arm_meta[a]["seed"]
            seed_b = arm_meta[b]["seed"]
            common_pair_rows.append({
                "arm_a": a,
                "arm_b": b,
                "pair_kind": "within_arch" if arch_a == arch_b else "between_arch",
                "pair_subkind": architecture_pair_kind(arch_a, seed_a, arch_b, seed_b),
                "p90_d2_curvature_gap": p90(sp.get(D2_GAP_COL, pd.Series(dtype=float))),
            })
    common_pair_df = pd.DataFrame(common_pair_rows)

    row = {
        **key,
        "n_models": int(len(arch_arms)),
        "n_architectures": int(group["hidden"].nunique()),
        "selected_slice_count": int(len(support["attempt_union"])),
        "all_model_attempt_intersection_slices": int(len(support["attempt_intersection"])),
        "all_model_attempt_sets_identical": bool(support["attempt_sets_identical"]),
        "all_model_common_success_slices": int(len(support["success_intersection"])),
        "all_model_common_success_share": float(len(support["success_intersection"]) / max(len(support["attempt_union"]), 1)),
        "control_l1_l2_p90_d2_curvature_gap": control_p90,
        "control_l1_l2_p90_curvature_gap": control_p90,
        "control_l1_l2_p90_spot_gamma_gap": control_spot_gamma_p90,
        "smooth_l2_s10_p90_d2_curvature_gap": smooth_p90,
        "smooth_l2_s10_p90_curvature_gap": smooth_p90,
        "smooth_l2_s10_p90_spot_gamma_gap": smooth_spot_gamma_p90,
        "common_support_control_l1_l2_p90_d2_curvature_gap": common_control_p90,
        "fit_best_price_rmse_scaled": min_loss,
        "fit_worst_price_rmse_scaled": max_loss,
        "fit_worst_over_best": max_loss / min_loss if np.isfinite(max_loss) and np.isfinite(min_loss) and min_loss > 0 else np.nan,
        "fit_worst_pct_above_best": 100.0 * (max_loss / min_loss - 1.0) if np.isfinite(max_loss) and np.isfinite(min_loss) and min_loss > 0 else np.nan,
        "fit_10pct_model_count": int(len(fit10)),
        "fit_20pct_model_count": int(len(fit20)),
    }

    for kind in ("within_arch", "between_arch"):
        sub = pair_df[pair_df["pair_kind"] == kind] if not pair_df.empty else pd.DataFrame()
        common_sub = common_pair_df[common_pair_df["pair_kind"] == kind] if not common_pair_df.empty else pd.DataFrame()
        fit10_sub = sub[sub["fit10_pair"]] if not sub.empty else pd.DataFrame()
        fit20_sub = sub[sub["fit20_pair"]] if not sub.empty else pd.DataFrame()
        med = median(sub["p90_d2_curvature_gap"]) if not sub.empty else np.nan
        common_med = median(common_sub["p90_d2_curvature_gap"]) if not common_sub.empty else np.nan
        med10 = median(fit10_sub["p90_d2_curvature_gap"]) if not fit10_sub.empty else np.nan
        med20 = median(fit20_sub["p90_d2_curvature_gap"]) if not fit20_sub.empty else np.nan
        row.update({
            f"{kind}_pair_count": int(len(sub)),
            f"{kind}_median_p90_d2_curvature_gap": med,
            f"{kind}_median_p90_curvature_gap": med,
            f"{kind}_excess_over_l1_l2": med - control_p90 if np.isfinite(med) and np.isfinite(control_p90) else np.nan,
            f"{kind}_median_p90_spot_gamma_gap": median(sub["p90_spot_gamma_gap"]) if not sub.empty else np.nan,
            f"{kind}_median_p90_hedge_gap_impact": median(sub["p90_hedge_gap_impact"]) if not sub.empty else np.nan,
            f"{kind}_fit10_pair_count": int(len(fit10_sub)),
            f"{kind}_fit10_median_p90_d2_curvature_gap": med10,
            f"{kind}_fit10_excess_over_l1_l2": med10 - control_p90 if np.isfinite(med10) and np.isfinite(control_p90) else np.nan,
            f"{kind}_fit20_pair_count": int(len(fit20_sub)),
            f"{kind}_fit20_median_p90_d2_curvature_gap": med20,
            f"{kind}_fit20_excess_over_l1_l2": med20 - control_p90 if np.isfinite(med20) and np.isfinite(control_p90) else np.nan,
            f"common_support_{kind}_median_p90_d2_curvature_gap": common_med,
            f"common_support_{kind}_excess_over_l1_l2": common_med - common_control_p90 if np.isfinite(common_med) and np.isfinite(common_control_p90) else np.nan,
        })

    for subkind in ("same_seed_between_arch", "cross_seed_between_arch"):
        sub = pair_df[pair_df["pair_subkind"] == subkind] if not pair_df.empty else pd.DataFrame()
        common_sub = common_pair_df[common_pair_df["pair_subkind"] == subkind] if not common_pair_df.empty else pd.DataFrame()
        fit10_sub = sub[sub["fit10_pair"]] if not sub.empty else pd.DataFrame()
        fit20_sub = sub[sub["fit20_pair"]] if not sub.empty else pd.DataFrame()
        med = median(sub["p90_d2_curvature_gap"]) if not sub.empty else np.nan
        common_med = median(common_sub["p90_d2_curvature_gap"]) if not common_sub.empty else np.nan
        med10 = median(fit10_sub["p90_d2_curvature_gap"]) if not fit10_sub.empty else np.nan
        med20 = median(fit20_sub["p90_d2_curvature_gap"]) if not fit20_sub.empty else np.nan
        row.update({
            f"{subkind}_pair_count": int(len(sub)),
            f"{subkind}_median_p90_d2_curvature_gap": med,
            f"{subkind}_excess_over_l1_l2": med - control_p90 if np.isfinite(med) and np.isfinite(control_p90) else np.nan,
            f"{subkind}_fit10_pair_count": int(len(fit10_sub)),
            f"{subkind}_fit10_median_p90_d2_curvature_gap": med10,
            f"{subkind}_fit10_excess_over_l1_l2": med10 - control_p90 if np.isfinite(med10) and np.isfinite(control_p90) else np.nan,
            f"{subkind}_fit20_pair_count": int(len(fit20_sub)),
            f"{subkind}_fit20_median_p90_d2_curvature_gap": med20,
            f"{subkind}_fit20_excess_over_l1_l2": med20 - control_p90 if np.isfinite(med20) and np.isfinite(control_p90) else np.nan,
            f"common_support_{subkind}_median_p90_d2_curvature_gap": common_med,
            f"common_support_{subkind}_excess_over_l1_l2": common_med - common_control_p90 if np.isfinite(common_med) and np.isfinite(common_control_p90) else np.nan,
        })

    return row, pair_df


def summarize_architecture_analysis(analysis_dir: Path, manifest: pd.DataFrame, fit: pd.DataFrame, mod) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pair_parts = []
    for _, group in manifest.groupby(["currency", "month", "selection"], sort=False):
        print(f"[summarize-arch] {group.iloc[0]['currency']} {group.iloc[0]['month']} {group.iloc[0]['selection']}", flush=True)
        row, pairs = summarize_architecture_window(mod, group, fit)
        rows.append(row)
        pair_parts.append(pairs)
    summary = pd.DataFrame(rows)
    pairs = pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame()
    summary.to_csv(analysis_dir / "architecture_window_summary.csv", index=False)
    pairs.to_csv(analysis_dir / "architecture_pair_summary.csv", index=False)
    return summary, pairs


def positive_count(df: pd.DataFrame, col: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int((pd.to_numeric(df[col], errors="coerce") > 0).sum())


def median_col(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").median(skipna=True))


def build_analysis_status(analysis_dir: Path, summary: pd.DataFrame, arch_summary: Optional[pd.DataFrame] = None) -> Dict:
    arch_summary = pd.DataFrame() if arch_summary is None else arch_summary
    required_columns = [
        "common_support_seed_pair_excess_over_l1_l2",
        "common_support_fit_20pct_seed_pair_excess",
        "common_support_min_leave_one_seed_excess",
    ]
    common_support_available = all(col in summary.columns for col in required_columns)
    status = {
        "analysis_dir": str(analysis_dir),
        "n_windows": int(len(summary)),
        "success_in_all_windows": bool((summary["n_seeds_success_ge_90pct"] >= 4).all()),
        "seed_excess_positive_windows": positive_count(summary, "seed_pair_excess_over_l1_l2"),
        "common_support_seed_excess_positive_windows": positive_count(summary, "common_support_seed_pair_excess_over_l1_l2"),
        "fit10_windows_with_pair": int((summary["fit_10pct_seed_count"] >= 2).sum()),
        "fit10_excess_positive_windows": positive_count(summary, "fit_10pct_seed_pair_excess"),
        "common_support_fit10_excess_positive_windows": positive_count(summary, "common_support_fit_10pct_seed_pair_excess"),
        "fit20_windows_with_pair": int((summary["fit_20pct_seed_count"] >= 2).sum()),
        "fit20_excess_positive_windows": positive_count(summary, "fit_20pct_seed_pair_excess"),
        "common_support_fit20_excess_positive_windows": positive_count(summary, "common_support_fit_20pct_seed_pair_excess"),
        "not_single_seed_windows": positive_count(summary, "min_leave_one_seed_excess"),
        "common_support_not_single_seed_windows": positive_count(summary, "common_support_min_leave_one_seed_excess"),
        "median_seed_excess": median_col(summary, "seed_pair_excess_over_l1_l2"),
        "median_common_support_seed_excess": median_col(summary, "common_support_seed_pair_excess_over_l1_l2"),
        "median_fit10_excess": median_col(summary, "fit_10pct_seed_pair_excess"),
        "median_common_support_fit10_excess": median_col(summary, "common_support_fit_10pct_seed_pair_excess"),
        "median_fit20_excess": median_col(summary, "fit_20pct_seed_pair_excess"),
        "median_common_support_fit20_excess": median_col(summary, "common_support_fit_20pct_seed_pair_excess"),
        "common_support_columns_available": bool(common_support_available),
    }
    if not arch_summary.empty:
        arch_windows = int(len(arch_summary))
        same_seed_positive = positive_count(arch_summary, "same_seed_between_arch_excess_over_l1_l2")
        common_same_seed_positive = positive_count(arch_summary, "common_support_same_seed_between_arch_excess_over_l1_l2")
        status.update({
            "architecture_windows": arch_windows,
            "within_arch_excess_positive_windows": positive_count(arch_summary, "within_arch_excess_over_l1_l2"),
            "between_arch_excess_positive_windows": positive_count(arch_summary, "between_arch_excess_over_l1_l2"),
            "common_support_within_arch_excess_positive_windows": positive_count(arch_summary, "common_support_within_arch_excess_over_l1_l2"),
            "common_support_between_arch_excess_positive_windows": positive_count(arch_summary, "common_support_between_arch_excess_over_l1_l2"),
            "same_seed_between_arch_excess_positive_windows": same_seed_positive,
            "common_support_same_seed_between_arch_excess_positive_windows": common_same_seed_positive,
            "median_between_arch_excess": median_col(arch_summary, "between_arch_excess_over_l1_l2"),
            "median_common_support_between_arch_excess": median_col(arch_summary, "common_support_between_arch_excess_over_l1_l2"),
            "median_same_seed_between_arch_excess": median_col(arch_summary, "same_seed_between_arch_excess_over_l1_l2"),
            "median_common_support_same_seed_between_arch_excess": median_col(arch_summary, "common_support_same_seed_between_arch_excess_over_l1_l2"),
            "architecture_sensitivity_pass": bool(
                same_seed_positive == arch_windows
                and common_same_seed_positive == arch_windows
                and (arch_summary["same_seed_between_arch_pair_count"] > 0).all()
            ),
        })
    status["analysis_criteria_pass"] = bool(
        common_support_available
        and status["success_in_all_windows"]
        and status["common_support_seed_excess_positive_windows"] == status["n_windows"]
        and status["common_support_not_single_seed_windows"] == status["n_windows"]
        and status["fit20_windows_with_pair"] == status["n_windows"]
        and status["common_support_fit20_excess_positive_windows"] == status["n_windows"]
    )
    return status


def summarize_analysis(analysis_dir: Path) -> Dict:
    manifest_path = analysis_dir / "run_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    fit_csv = analysis_dir / "heldout_fit" / "nnproj_heldout_fit_by_run.csv"
    if not fit_csv.exists():
        fit_csv = run_fit_audit(analysis_dir)
    fit = pd.read_csv(fit_csv)
    mod = load_repair_module()

    rows = []
    pair_parts = []
    for _, group in manifest.groupby(["currency", "month", "selection", "hidden"], sort=False):
        print(f"[summarize] {group.iloc[0]['currency']} {group.iloc[0]['month']} {group.iloc[0]['selection']} hidden={group.iloc[0]['hidden']}", flush=True)
        row, pairs = summarize_window(mod, group, fit)
        rows.append(row)
        pair_parts.append(pairs)
    summary = pd.DataFrame(rows)
    pairs = pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame()
    summary.to_csv(analysis_dir / "window_summary.csv", index=False)
    pairs.to_csv(analysis_dir / "seed_pair_summary.csv", index=False)
    arch_summary = pd.DataFrame()
    if int(manifest["hidden"].nunique()) > 1:
        arch_summary, _ = summarize_architecture_analysis(analysis_dir, manifest, fit, mod)

    status = build_analysis_status(analysis_dir, summary, arch_summary)
    (analysis_dir / "analysis_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run and summarize NN-Proj seed and architecture sensitivity analyses.")
    p.add_argument("action", choices=["manifest", "run", "audit", "summarize", "all"])
    p.add_argument("--analysis-set", choices=["core", "extended", "architecture"], default="core")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit-runs", type=int, default=0)
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args(argv)

    if not REPAIR_SCRIPT.exists():
        raise SystemExit(f"Missing repair script: {REPAIR_SCRIPT}")
    if not PANEL.exists():
        raise SystemExit(f"Missing panel: {PANEL}")

    analysis_dir = args.out_root / args.analysis_set
    rows = analysis_specs(args.analysis_set)
    if args.action in {"manifest", "run", "all"}:
        manifest = write_manifest(analysis_dir, rows)
    else:
        manifest = pd.read_csv(analysis_dir / "run_manifest.csv")

    if args.action == "manifest":
        print(f"[done] manifest: {analysis_dir / 'run_manifest.csv'}")
        return 0
    if args.action == "run":
        run_manifest(analysis_dir, manifest, limit=args.limit_runs, rerun=args.rerun, jobs=args.jobs)
        return 0
    if args.action == "audit":
        run_fit_audit(analysis_dir)
        return 0
    if args.action == "summarize":
        status = summarize_analysis(analysis_dir)
        print(json.dumps(status, indent=2))
        return 0
    if args.action == "all":
        run_manifest(analysis_dir, manifest, limit=args.limit_runs, rerun=args.rerun, jobs=args.jobs)
        status = summarize_analysis(analysis_dir)
        print(json.dumps(status, indent=2))
        return 0
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
