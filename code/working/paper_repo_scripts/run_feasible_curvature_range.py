"""Empirical feasible-set curvature range diagnostic.

This isolated script estimates the paper's theoretical object

    Risk_D2(i; A) = sup_{c,c' in A} |(D2 c)_i - (D2 c')_i|

on the actual selected Deribit slices used in the repair-risk experiments.  For
each requested strike row it solves two linear programs over the same
per-maturity effective bounds, monotonicity, vertical-spread, and convexity
feasible set:

    max_c (D2 c)_i,    min_c (D2 c)_i.

The output is a bridge diagnostic, not a new cleaner: it asks whether the
available feasible-set curvature width lines up with observed L2--Smooth10 or
L2--NN-Proj downstream curvature dispersion.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linprog

_COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
import sys
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
from full_strike_constraints import (  # noqa: E402
    build_full_constraint_matrix,
    effective_bounds,
)

_WORKING_DIR = Path(__file__).resolve().parents[1]
if str(_WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKING_DIR))
from artifact_paths import PANEL as PANEL_DEFAULT, RESULTS  # noqa: E402


OUTPUT_DEFAULT = RESULTS / "lp_range"

CLASSICAL_ROOTS_DEFAULT = [
    RESULTS / "classical_18m",
]
NNPROJ_ROOT_DEFAULT = RESULTS / "nn_targeted"

PAIR_TARGETS_CLASSICAL = [("L2", "SMOOTH_RHO_10")]
PAIR_TARGETS_NNPROJ = [("L2", "SMOOTH_RHO_10"), ("L2", "NNPROJ")]

EPS = 1e-12
BOX_EMPTY_TOL = 1e-12
HIGHS_SOLVER_TOL = 1e-9
MATCHED_CERTIFICATION_TOL = 2e-6


@dataclass(frozen=True)
class RunInfo:
    source: str
    run: str
    path: Path
    currency: str
    month: str
    selection: str
    pair_targets: tuple[tuple[str, str], ...]


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def parse_run_dir(path: Path, source: str, pair_targets: list[tuple[str, str]]) -> RunInfo:
    name = path.name
    m = re.search(r"^(BTC|ETH)_(\d{4}-\d{2})_", name)
    if not m:
        raise ValueError(f"Cannot parse currency/month from run directory: {name}")
    selection = "unknown"
    if "bottom50" in name:
        selection = "bottom"
    elif "random50" in name:
        selection = "random"
    elif "top50" in name:
        selection = "top"
    elif "top100" in name:
        selection = "top100"
    return RunInfo(
        source=source,
        run=name,
        path=path,
        currency=m.group(1),
        month=m.group(2),
        selection=selection,
        pair_targets=tuple(pair_targets),
    )


def discover_runs(
    classical_roots: list[Path],
    nnproj_root: Path | None,
    *,
    max_runs: int | None,
) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for root in classical_roots:
        if not root.exists():
            continue
        source = root.name
        for path in sorted(root.iterdir()):
            if path.is_dir() and (path / "strike_pairwise_dispersion.csv").exists():
                runs.append(parse_run_dir(path, source, PAIR_TARGETS_CLASSICAL))
    if nnproj_root is not None and nnproj_root.exists():
        for path in sorted(nnproj_root.iterdir()):
            if path.is_dir() and (path / "strike_pairwise_dispersion.csv").exists():
                runs.append(parse_run_dir(path, nnproj_root.name, PAIR_TARGETS_NNPROJ))
    if max_runs is not None:
        runs = runs[:max_runs]
    if not runs:
        raise FileNotFoundError("No runs with strike_pairwise_dispersion.csv were discovered.")
    return runs


def pair_mask(df: pd.DataFrame, arm_a: str, arm_b: str) -> pd.Series:
    return ((df["arm_a"] == arm_a) & (df["arm_b"] == arm_b)) | (
        (df["arm_a"] == arm_b) & (df["arm_b"] == arm_a)
    )


def collect_pairwise_targets(runs: list[RunInfo]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    needed = [
        "currency",
        "date",
        "expiration_date",
        "strike",
        "arm_a",
        "arm_b",
        "DTE",
        "k",
        "h_norm",
        "danger_score_rel_h2",
        "danger_score",
        "relspread_floor",
        "cxx_diff",
        "cxx_diff_scaled",
        "gamma_diff_scaled",
        "price_diff_box",
        "core_repair_slice",
        "edge_strike",
    ]
    for run in runs:
        path = run.path / "strike_pairwise_dispersion.csv"
        df = pd.read_csv(path)
        require_columns(df, needed, str(path))
        for arm_a, arm_b in run.pair_targets:
            sub = df.loc[pair_mask(df, arm_a, arm_b), needed].copy()
            if sub.empty:
                continue
            sub["source"] = run.source
            sub["run"] = run.run
            sub["month"] = run.month
            sub["selection"] = run.selection
            sub["pair_label"] = f"{arm_a}_vs_{arm_b}"
            pieces.append(sub)
    if not pieces:
        raise ValueError("No matching pairwise rows found.")
    out = pd.concat(pieces, ignore_index=True)
    out = out.loc[np.isfinite(out["cxx_diff"])].copy()
    out = out.loc[out["core_repair_slice"].astype(bool) & ~out["edge_strike"].astype(bool)].copy()
    out["target_key"] = (
        out["currency"].astype(str)
        + "|"
        + out["date"].astype("int64").astype(str)
        + "|"
        + out["expiration_date"].astype("int64").astype(str)
        + "|"
        + out["strike"].round(10).astype(str)
    )
    return out


def d2_row_nonuniform(x: np.ndarray, i: int) -> np.ndarray:
    if not (0 < i < len(x) - 1):
        raise IndexError("D2 row requires an interior strike index.")
    hm = float(x[i] - x[i - 1])
    hp = float(x[i + 1] - x[i])
    if hm <= 0 or hp <= 0:
        raise ValueError("Normalized strike grid must be strictly increasing.")
    row = np.zeros(len(x), dtype=float)
    coeff = 2.0 / (hm + hp)
    row[i - 1] = coeff / hm
    row[i] = coeff * (-1.0 / hm - 1.0 / hp)
    row[i + 1] = coeff / hp
    return row


def d2_matrix_nonuniform(x: np.ndarray) -> np.ndarray:
    rows = [d2_row_nonuniform(x, i) for i in range(1, len(x) - 1)]
    return np.asarray(rows, dtype=float) if rows else np.zeros((0, len(x)), dtype=float)


def build_lp_constraints(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a_ub, b_ub, _ = build_full_constraint_matrix(x)
    return a_ub, b_ub


def solve_range_for_row(
    row: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[float, float, str, float, float]:
    bounds = list(zip(lower, upper))
    options = {
        "primal_feasibility_tolerance": HIGHS_SOLVER_TOL,
        "dual_feasibility_tolerance": HIGHS_SOLVER_TOL,
    }
    min_res = linprog(row, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=options)
    max_res = linprog(-row, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=options)
    if not min_res.success or not max_res.success:
        return (math.nan, math.nan, f"min={min_res.status}:{min_res.message}; max={max_res.status}:{max_res.message}", math.nan, math.nan)
    min_val = float(min_res.fun)
    max_val = float(-max_res.fun)
    if max_val < min_val and max_val > min_val - 1e-8:
        max_val = min_val
    # Matrix residuals use the actual nonuniform grid and are the authoritative LP check.
    min_residual = float(max(np.max(a_ub @ min_res.x - b_ub), np.max(lower - min_res.x), np.max(min_res.x - upper), 0.0))
    max_residual = float(max(np.max(a_ub @ max_res.x - b_ub), np.max(lower - max_res.x), np.max(max_res.x - upper), 0.0))
    status = (
        "ok"
        if max(min_residual, max_residual) <= MATCHED_CERTIFICATION_TOL
        else "lp_solution_certification_failed"
    )
    return (min_val, max_val, status, min_residual, max_residual)


def load_panel_slices(panel_path: Path, needed_keys: pd.DataFrame) -> dict[tuple[str, int, int], pd.DataFrame]:
    panel = pd.read_parquet(panel_path)
    require_columns(
        panel,
        ["currency", "date", "expiration_date", "strike", "forward_price", "C_bid", "C_ask", "C_mid"],
        str(panel_path),
    )
    needed = needed_keys[["currency", "date", "expiration_date"]].drop_duplicates().copy()
    needed["date"] = needed["date"].astype("int64")
    needed["expiration_date"] = needed["expiration_date"].astype("int64")
    panel["date"] = panel["date"].astype("int64")
    panel["expiration_date"] = panel["expiration_date"].astype("int64")
    merged = panel.merge(needed, on=["currency", "date", "expiration_date"], how="inner")
    if merged.empty:
        raise ValueError("Panel merge found no selected slices.")
    slices: dict[tuple[str, int, int], pd.DataFrame] = {}
    for key, g in merged.groupby(["currency", "date", "expiration_date"], sort=False):
        slices[(str(key[0]), int(key[1]), int(key[2]))] = g.sort_values("strike").copy()
    return slices


def compute_ranges_for_slice(
    key: tuple[object, object, object],
    slc: pd.DataFrame | None,
    targets: pd.DataFrame,
) -> list[dict[str, object]]:
    """Compute all requested row ranges for one independent maturity slice."""
    skey = (str(key[0]), int(key[1]), int(key[2]))
    records: list[dict[str, object]] = []
    if slc is None:
        for _, target in targets.iterrows():
            records.append({
                "currency": skey[0], "date": skey[1], "expiration_date": skey[2],
                "strike": float(target["strike"]), "range_status": "missing_panel_slice",
            })
        return records

    slc = slc.loc[np.isfinite(slc[["strike", "forward_price", "C_bid", "C_ask"]]).all(axis=1)].copy()
    if "synthetic_valid" in slc.columns:
        slc = slc.loc[slc["synthetic_valid"].astype(bool)].copy()
    slc = slc.sort_values("strike").drop_duplicates("strike", keep="first")
    if len(slc) < 3:
        for _, target in targets.iterrows():
            records.append({
                "currency": skey[0], "date": skey[1], "expiration_date": skey[2],
                "strike": float(target["strike"]), "range_status": "too_few_strikes",
            })
        return records

    fwd = float(np.nanmedian(slc["forward_price"].to_numpy(float)))
    strike = slc["strike"].to_numpy(float)
    x = strike / fwd
    bid = slc["C_bid"].to_numpy(float)
    ask = slc["C_ask"].to_numpy(float)
    finite_box = np.isfinite(x) & np.isfinite(bid) & np.isfinite(ask)
    strike, x, bid, ask = strike[finite_box], x[finite_box], bid[finite_box], ask[finite_box]
    order = np.argsort(strike)
    strike, x, bid, ask = strike[order], x[order], bid[order], ask[order]
    intrinsic = np.maximum(1.0 - x, 0.0)
    lower, upper = effective_bounds(x, bid, ask, intrinsic)
    if len(strike) < 3 or np.any(np.diff(x) <= 0):
        status = "bad_grid"
    elif np.any(lower > upper + BOX_EMPTY_TOL):
        status = "bad_bid_ask_box"
    else:
        status = ""
    if status:
        for _, target in targets.iterrows():
            records.append({
                "currency": skey[0], "date": skey[1], "expiration_date": skey[2],
                "strike": float(target["strike"]), "range_status": status,
            })
        return records

    a_ub, b_ub = build_lp_constraints(x)
    for _, target in targets.iterrows():
        target_strike = float(target["strike"])
        idx = int(np.argmin(np.abs(strike - target_strike)))
        base = {
            "currency": skey[0], "date": skey[1], "expiration_date": skey[2],
            "strike": target_strike,
        }
        if abs(strike[idx] - target_strike) > max(1e-8, 1e-8 * abs(target_strike)):
            records.append({**base, "range_status": "strike_not_found"})
            continue
        if idx == 0 or idx == len(strike) - 1:
            records.append({**base, "range_status": "edge_strike"})
            continue
        row = d2_row_nonuniform(x, idx)
        min_val, max_val, range_status, min_residual, max_residual = solve_range_for_row(
            row, a_ub, b_ub, lower, upper
        )
        records.append({
            **base,
            "range_status": range_status,
            "feasible_cxx_min": min_val,
            "feasible_cxx_max": max_val,
            "feasible_cxx_range": max_val - min_val if range_status == "ok" else math.nan,
            "lp_min_max_constraint_violation": min_residual,
            "lp_max_max_constraint_violation": max_residual,
            "d2_row_l1_norm": float(np.sum(np.abs(row))),
            "local_h_minus": float(x[idx] - x[idx - 1]),
            "local_h_plus": float(x[idx + 1] - x[idx]),
            "local_h_proxy": float(math.sqrt((x[idx] - x[idx - 1]) * (x[idx + 1] - x[idx]))),
            "bid_ask_width_center": float(upper[idx] - lower[idx]),
        })
    return records


def compute_ranges(panel_path: Path, target_rows: pd.DataFrame, jobs: int = 1) -> pd.DataFrame:
    slice_map = load_panel_slices(panel_path, target_rows)
    target_by_slice = {
        key: group.copy()
        for key, group in target_rows.drop_duplicates(
            ["currency", "date", "expiration_date", "strike"]
        ).groupby(["currency", "date", "expiration_date"], sort=False)
    }
    tasks = [
        (key, slice_map.get((str(key[0]), int(key[1]), int(key[2]))), targets)
        for key, targets in target_by_slice.items()
    ]
    records: list[dict[str, object]] = []
    if jobs <= 1:
        for index, task in enumerate(tasks, 1):
            records.extend(compute_ranges_for_slice(*task))
            if index % 25 == 0:
                print(f"[LP] completed {index}/{len(tasks)} slices", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(compute_ranges_for_slice, *task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                records.extend(future.result())
                if index % 25 == 0:
                    print(f"[LP] completed {index}/{len(tasks)} slices", flush=True)
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["currency", "date", "expiration_date", "strike"])
        .reset_index(drop=True)
    )


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    z = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 3 or z["x"].nunique() < 2 or z["y"].nunique() < 2:
        return math.nan
    return float(z["x"].corr(z["y"], method=method))


def summarize(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ok = merged.loc[merged["range_status"] == "ok"].copy()
    ok["obs_to_range"] = ok["cxx_diff"].abs() / ok["feasible_cxx_range"].clip(lower=EPS)
    ok["range_x_relspread_h2"] = ok["feasible_cxx_range"] * ok["danger_score_rel_h2"]
    ok["log_range"] = np.log1p(ok["feasible_cxx_range"].clip(lower=0))
    ok["log_abs_cxx_diff"] = np.log1p(ok["cxx_diff"].abs().clip(lower=0))
    ok["range_violation"] = ok["cxx_diff"].abs() > (ok["feasible_cxx_range"] * (1.0 + 1e-5) + 1e-10)

    group_cols = ["source", "pair_label", "selection"]
    rows: list[dict[str, object]] = []
    for key, g in ok.groupby(group_cols, dropna=False):
        pearson = safe_corr(g["log_range"], g["log_abs_cxx_diff"], "pearson")
        rows.append(
            {
                "source": key[0],
                "pair_label": key[1],
                "selection": key[2],
                "n_rows": len(g),
                "n_runs": g["run"].nunique(),
                "n_clusters": g[["currency", "month"]].drop_duplicates().shape[0],
                "median_feasible_cxx_range": float(g["feasible_cxx_range"].median()),
                "p90_feasible_cxx_range": float(g["feasible_cxx_range"].quantile(0.90)),
                "median_abs_cxx_diff": float(g["cxx_diff"].abs().median()),
                "p90_abs_cxx_diff": float(g["cxx_diff"].abs().quantile(0.90)),
                "median_obs_to_range": float(g["obs_to_range"].median()),
                "p90_obs_to_range": float(g["obs_to_range"].quantile(0.90)),
                "spearman_range_vs_abs_cxx": safe_corr(g["feasible_cxx_range"], g["cxx_diff"].abs(), "spearman"),
                "pearson_log_range_vs_log_abs_cxx": pearson,
                "r2_log_range_vs_log_abs_cxx": pearson * pearson if math.isfinite(pearson) else math.nan,
                "spearman_relspread_h2_vs_range": safe_corr(g["danger_score_rel_h2"], g["feasible_cxx_range"], "spearman"),
                "range_violation_rate": float(g["range_violation"].mean()),
            }
        )
    summary = pd.DataFrame(rows)

    run_rows: list[dict[str, object]] = []
    for key, g in ok.groupby(["source", "run", "currency", "month", "selection", "pair_label"], dropna=False):
        run_rows.append(
            {
                "source": key[0],
                "run": key[1],
                "currency": key[2],
                "month": key[3],
                "selection": key[4],
                "pair_label": key[5],
                "n_rows": len(g),
                "median_feasible_cxx_range": float(g["feasible_cxx_range"].median()),
                "p90_feasible_cxx_range": float(g["feasible_cxx_range"].quantile(0.90)),
                "median_abs_cxx_diff": float(g["cxx_diff"].abs().median()),
                "p90_abs_cxx_diff": float(g["cxx_diff"].abs().quantile(0.90)),
                "median_obs_to_range": float(g["obs_to_range"].median()),
                "p90_obs_to_range": float(g["obs_to_range"].quantile(0.90)),
                "spearman_range_vs_abs_cxx": safe_corr(g["feasible_cxx_range"], g["cxx_diff"].abs(), "spearman"),
            }
        )
    run_summary = pd.DataFrame(run_rows)

    status = (
        merged.groupby(["source", "range_status"], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["source", "range_status"])
    )
    return summary, run_summary, status


def write_markdown_summary(
    output_dir: Path,
    *,
    target_rows: pd.DataFrame,
    merged: pd.DataFrame,
    summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    run_summary: pd.DataFrame,
    status: pd.DataFrame,
    config: dict[str, object],
) -> None:
    def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
        if df.empty:
            return "_No rows._"
        view = df.loc[:, columns].copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
            else:
                view[col] = view[col].astype(str)
        header = "| " + " | ".join(view.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in view.to_numpy(dtype=str)]
        return "\n".join([header, sep, *rows])

    ok = merged.loc[merged["range_status"] == "ok"].copy()
    violation_rate = float((ok["cxx_diff"].abs() > (ok["feasible_cxx_range"] * (1.0 + 1e-5) + 1e-10)).mean()) if len(ok) else math.nan
    lines = [
        "# Empirical feasible-set curvature range",
        "",
        "This diagnostic solves exact LP bounds for each selected strike row:",
        "",
        "`max_c (D2 c)_i - min_c (D2 c)_i` over the complete per-maturity normalized-call strike feasible set.",
        "",
        "It is a bridge diagnostic between the finite-difference theorem and observed repair-risk dispersion; it is not a new repair arm.",
        "",
        "## Coverage",
        "",
        f"- Pairwise target rows collected: {len(target_rows):,}.",
        f"- Rows with successful feasible-range LPs after merge: {len(ok):,} / {len(merged):,}.",
        f"- Range bound violation rate against observed pairwise cxx differences: {violation_rate:.6f}.",
        "",
        "## Main summaries",
        "",
    ]
    display_cols = [
        "source",
        "pair_label",
        "selection",
        "n_rows",
        "n_clusters",
        "median_feasible_cxx_range",
        "p90_feasible_cxx_range",
        "median_abs_cxx_diff",
        "p90_abs_cxx_diff",
        "median_obs_to_range",
        "spearman_range_vs_abs_cxx",
        "r2_log_range_vs_log_abs_cxx",
    ]
    if not summary.empty:
        lines.append(markdown_table(summary, display_cols))
    else:
        lines.append("_No successful summary rows._")
    lines.extend(["", "## Family summaries", ""])
    if not family_summary.empty:
        lines.append(markdown_table(family_summary, display_cols))
    else:
        lines.append("_No family summary rows._")
    lines.extend(["", "## LP status", ""])
    if not status.empty:
        lines.append(markdown_table(status, list(status.columns)))
    lines.extend(["", "## Configuration", "", "```json", json.dumps(config, indent=2, sort_keys=True), "```", ""])
    (output_dir / "FEASIBLE_CURVATURE_RANGE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def self_check_math() -> None:
    x = np.array([0.0, 1.0, 2.0])
    row = d2_row_nonuniform(x, 1)
    if not np.allclose(row, np.array([1.0, -2.0, 1.0])):
        raise AssertionError(f"Uniform D2 row self-check failed: {row}")
    a_ub, b_ub = build_lp_constraints(x)
    lower = np.zeros(3)
    upper = np.ones(3)
    min_val, max_val, status, _, _ = solve_range_for_row(row, a_ub, b_ub, lower, upper)
    if status != "ok":
        raise AssertionError(f"Simple LP self-check failed with status {status}")
    if abs(min_val) > 1e-9 or abs(max_val - 1.0) > 1e-9:
        raise AssertionError(f"Simple LP range expected [0,1], got [{min_val}, {max_val}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--classical-root", type=Path, action="append")
    parser.add_argument("--nnproj-root", type=Path, default=NNPROJ_ROOT_DEFAULT)
    parser.add_argument("--no-nnproj", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    self_check_math()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    classical_roots = args.classical_root if args.classical_root else CLASSICAL_ROOTS_DEFAULT
    nnproj_root = None if args.no_nnproj else args.nnproj_root
    config = {
        "panel": str(args.panel),
        "output_dir": str(output_dir),
        "classical_roots": [str(p) for p in classical_roots],
        "nnproj_root": str(nnproj_root) if nnproj_root else None,
        "max_runs": args.max_runs,
        "jobs": args.jobs,
        "lp_constraints": "effective normalized-call bounds + monotonicity + vertical-spread upper bound + convexity",
        "highs_solver_tolerance": HIGHS_SOLVER_TOL,
        "repair_matched_certification_tolerance": MATCHED_CERTIFICATION_TOL,
        "empty_box_tolerance": BOX_EMPTY_TOL,
    }
    runs = discover_runs(classical_roots, nnproj_root, max_runs=args.max_runs)
    target_rows = collect_pairwise_targets(runs)
    ranges = compute_ranges(args.panel, target_rows, jobs=max(1, args.jobs))
    merge_keys = ["currency", "date", "expiration_date", "strike"]
    merged = target_rows.merge(ranges, on=merge_keys, how="left")
    classical_source_names = {Path(path).name for path in classical_roots}
    merged["source_family"] = np.where(
        merged["source"].astype(str).isin(classical_source_names),
        "classical_all_selected_roots",
        merged["source"],
    )
    summary, run_summary, status = summarize(merged)
    family_input = merged.copy()
    family_input["source"] = family_input["source_family"]
    family_summary, _, _ = summarize(family_input)

    target_rows.to_csv(output_dir / "feasible_curvature_pairwise_targets.csv", index=False)
    ranges.to_csv(output_dir / "feasible_curvature_range_by_row.csv", index=False)
    merged.to_csv(output_dir / "feasible_curvature_range_merged_pairwise.csv", index=False)
    summary.to_csv(output_dir / "feasible_curvature_range_summary.csv", index=False)
    family_summary.to_csv(output_dir / "feasible_curvature_range_family_summary.csv", index=False)
    run_summary.to_csv(output_dir / "feasible_curvature_range_run_summary.csv", index=False)
    status.to_csv(output_dir / "feasible_curvature_range_status.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_summary(
        output_dir,
        target_rows=target_rows,
        merged=merged,
        summary=summary,
        family_summary=family_summary,
        run_summary=run_summary,
        status=status,
        config=config,
    )
    print(f"Wrote feasible curvature range diagnostics to {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
