#!/usr/bin/env python3
"""Aggregate repair-analysis outputs and diagnostic summaries."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_run_name(run: str) -> dict:
    out = {"run": run}
    m = re.match(r"(?P<currency>BTC|ETH)_(?P<month>\d{4}-\d{2})_(?P<tag>.+)", run)
    if m:
        out.update(m.groupdict())
    tag = out.get("tag", run)
    m2 = re.search(r"(?P<selection>top|bottom|random)(?P<N>\d+)", tag)
    if m2:
        out["slice_selection_from_run"] = m2.group("selection")
        out["selected_slices_from_run"] = int(m2.group("N"))
    return out


def collect_csv(root: Path, filename: str) -> pd.DataFrame:
    rows = []
    for f in sorted(root.rglob(filename)):
        if any(part.startswith("_") for part in f.parts):
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[warn] could not read {f}: {e}")
            continue
        run = f.parent.name
        meta = parse_run_name(run)
        for k, v in meta.items():
            if k not in df.columns:
                df.insert(0, k, v)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_csv_columns(root: Path, filename: str, columns: set[str]) -> pd.DataFrame:
    """Collect only selected columns from possibly large CSV outputs."""
    rows = []
    for f in sorted(root.rglob(filename)):
        if any(part.startswith("_") for part in f.parts):
            continue
        try:
            df = pd.read_csv(f, usecols=lambda c: c in columns)
        except Exception as e:
            print(f"[warn] could not read selected columns from {f}: {e}")
            continue
        run = f.parent.name
        meta = parse_run_name(run)
        for k, v in meta.items():
            if k not in df.columns:
                df.insert(0, k, v)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def q(x, p=0.9) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.nanpercentile(x, 100 * p)) if len(x) else np.nan


def q90(x) -> float:
    return q(x, 0.9)


def bool_rate(x) -> float:
    if len(x) == 0:
        return np.nan
    return float(pd.Series(x).astype(bool).mean())


def extract_rho_from_arm(arm: str) -> float:
    m = re.search(r"SMOOTH_RHO_([0-9P]+)", str(arm))
    return float(m.group(1).replace("P", ".")) if m else np.nan


def baseline_and_rho(row: pd.Series):
    a, b = str(row.get("arm_a", "")), str(row.get("arm_b", ""))
    for base, other in [(a, b), (b, a)]:
        if base in {"L1", "L2"} and re.search(r"SMOOTH_RHO_", other):
            return base, extract_rho_from_arm(other)
    return None, np.nan


def ensure_selection_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "slice_selection" not in out.columns and "slice_selection_from_run" in out.columns:
        out["slice_selection"] = out["slice_selection_from_run"]
    if "selected_slices_requested" not in out.columns and "selected_slices_from_run" in out.columns:
        out["selected_slices_requested"] = out["selected_slices_from_run"]
    return out


def make_status_summary(repair_summary: pd.DataFrame) -> pd.DataFrame:
    if repair_summary.empty:
        return pd.DataFrame()
    df = ensure_selection_cols(repair_summary)
    group_cols = [c for c in ["currency", "month", "slice_selection", "selected_slices_requested"] if c in df.columns]
    metrics = {}
    for col in ["joint_gamma_signal", "scaled_hedge_signal", "curvature_signal", "any_signal"]:
        if col in df.columns:
            metrics[col + "_rate"] = (col, bool_rate)
    for col in [
        "high_score_p90_gamma_diff_scaled", "high_score_small_price_large_gamma_scaled_rate",
        "high_score_p90_cxx_diff_scaled", "high_score_small_price_large_curvature_rate",
        "book_p90_hedge_slippage_scaled", "book_p90_hedge_diff_notional_scaled",
        "book_hedge_diff_notional_gt_0p05_rate", "book_hedge_trigger_flip_0p5_rate",
        "book_hedge_trigger_flip_1_rate", "book_hedge_trigger_flip_2_rate", "book_hedge_trigger_flip_any_rate",
        "book_nextday_pair_rate",
    ]:
        if col in df.columns:
            metrics["median_" + col] = (col, "median")
    if not metrics:
        return pd.DataFrame()
    return df.groupby(group_cols, dropna=False).agg(n_runs=("run", "nunique"), **metrics).reset_index()


def make_rho_vs_baseline_summary(arm: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
    if arm.empty:
        return pd.DataFrame()
    df = ensure_selection_cols(arm)
    parsed = df.apply(baseline_and_rho, axis=1, result_type="expand")
    df["baseline"] = parsed[0]
    df["rho"] = parsed[1]
    df = df[df["baseline"].notna() & df["rho"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    keys = [c for c in ["run", "currency", "month", "slice_selection", "selected_slices_from_run", "baseline", "rho"] if c in df.columns]
    agg_map = {}
    for col in ["p90_gamma_diff_scaled", "p90_cxx_diff_scaled", "p90_rnd_diff_scaled", "small_price_large_gamma_scaled_rate", "small_price_large_curvature_rate"]:
        if col in df.columns:
            agg_map[col] = (col, "median")
    out = df.groupby(keys, dropna=False).agg(**agg_map).reset_index()
    if not book.empty and {"arm_a", "arm_b"}.issubset(book.columns):
        bd = ensure_selection_cols(book)
        parsed_b = bd.apply(baseline_and_rho, axis=1, result_type="expand")
        bd["baseline"] = parsed_b[0]
        bd["rho"] = parsed_b[1]
        bd = bd[bd["baseline"].notna() & bd["rho"].notna()].copy()
        if not bd.empty:
            bkeys = [c for c in ["run", "currency", "month", "slice_selection", "selected_slices_from_run", "baseline", "rho"] if c in bd.columns]
            bagg = {}
            for col in ["hedge_slippage_scaled", "gamma_pnl_proxy", "hedge_diff_notional_scaled", "loss_common_diff"]:
                if col in bd.columns:
                    bagg[f"p90_{col}"] = (col, q90)
            for col in ["hedge_diff_notional_gt_0p05", "hedge_diff_notional_gt_0p1", "hedge_diff_notional_gt_0p25", "hedge_trigger_flip_0p5", "hedge_trigger_flip_1", "hedge_trigger_flip_2"]:
                if col in bd.columns:
                    bagg[col + "_rate"] = (col, "mean")
            if "nextday_available" in bd.columns:
                bagg["nextday_pair_rate"] = ("nextday_available", "mean")
            bsum = bd.groupby(bkeys, dropna=False).agg(**bagg).reset_index()
            out = out.merge(bsum, on=bkeys, how="left")
    return out.sort_values([c for c in ["currency", "month", "slice_selection", "selected_slices_from_run", "run", "baseline", "rho"] if c in out.columns])


def make_book_group_summary(book: pd.DataFrame) -> pd.DataFrame:
    if book.empty:
        return pd.DataFrame()
    df = ensure_selection_cols(book)
    df["book_group"] = np.where(df.get("book", "") == "LOCAL_DIAGNOSTIC_FLY", "local_diagnostic", "standardized_books")
    keys = [c for c in ["run", "currency", "month", "slice_selection", "selected_slices_from_run", "book_group", "book"] if c in df.columns]
    agg = {"n": ("book", "size")}
    for col in ["hedge_slippage_scaled", "gamma_pnl_proxy", "hedge_diff_notional_scaled", "loss_common_diff", "gamma_rr", "delta_rr"]:
        if col in df.columns:
            agg[f"p50_{col}"] = (col, "median")
            agg[f"p90_{col}"] = (col, q90)
    for col in ["hedge_diff_notional_gt_0p05", "hedge_diff_notional_gt_0p1", "hedge_diff_notional_gt_0p25", "hedge_trigger_flip_0p5", "hedge_trigger_flip_1", "hedge_trigger_flip_2"]:
        if col in df.columns:
            agg[col + "_rate"] = (col, "mean")
    if "nextday_available" in df.columns:
        agg["nextday_pair_rate"] = ("nextday_available", "mean")
    return df.groupby(keys, dropna=False).agg(**agg).reset_index()


def add_run_book_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Attach next-day coverage at run x book level.

    Run-level coverage can retain a book with low next-day support inside an
    otherwise high-coverage run, so filtering is applied by run and book.
    """
    if df.empty or "nextday_available" not in df.columns or "book" not in df.columns:
        return df.copy()
    cov = (
        df.groupby(["run", "book"], dropna=False)["nextday_available"]
          .mean()
          .rename("run_book_nextday_pair_rate")
          .reset_index()
    )
    return df.merge(cov, on=["run", "book"], how="left")


def make_book_group_summary_coverage_filtered(book: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if book.empty or "nextday_available" not in book.columns:
        return pd.DataFrame()
    df = ensure_selection_cols(book)
    df = add_run_book_coverage(df)
    df = df[df["run_book_nextday_pair_rate"] >= threshold].copy()
    if df.empty:
        return pd.DataFrame()
    out = make_book_group_summary(df)
    out["coverage_threshold"] = threshold
    return out


def make_standardized_book_per_book_summary(book: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    if book.empty or "book" not in book.columns:
        return pd.DataFrame()
    df = ensure_selection_cols(book)
    if threshold is not None and "nextday_available" in df.columns:
        df = add_run_book_coverage(df)
        df = df[df["run_book_nextday_pair_rate"] >= threshold].copy()
    df = df[df["book"] != "LOCAL_DIAGNOSTIC_FLY"].copy()
    if df.empty:
        return pd.DataFrame()
    keys = [c for c in ["book", "currency", "month", "slice_selection", "selected_slices_from_run"] if c in df.columns]
    agg = {"n": ("book", "size")}
    for col in ["hedge_slippage_scaled", "gamma_pnl_proxy", "hedge_diff_notional_scaled", "loss_common_diff", "gamma_rr", "delta_rr"]:
        if col in df.columns:
            agg[f"p50_{col}"] = (col, "median")
            agg[f"p90_{col}"] = (col, q90)
    for col in ["hedge_diff_notional_gt_0p05", "hedge_diff_notional_gt_0p1", "hedge_diff_notional_gt_0p25", "hedge_trigger_flip_0p5", "hedge_trigger_flip_1", "hedge_trigger_flip_2"]:
        if col in df.columns:
            agg[col + "_rate"] = (col, "mean")
    if "nextday_available" in df.columns:
        agg["nextday_pair_rate"] = ("nextday_available", "mean")
    out = df.groupby(keys, dropna=False).agg(**agg).reset_index()
    if threshold is not None:
        out["coverage_threshold"] = threshold
    return out



def make_decision_flip_summary(book: pd.DataFrame) -> pd.DataFrame:
    """Summarize hedge-trigger, hedge-difference, slippage, and gamma-proxy event rates.

    The summary is computed per run and book so coverage filtering can be applied
    separately to standardized books and LOCAL_DIAGNOSTIC_FLY.
    """
    if book.empty:
        return pd.DataFrame()
    df = ensure_selection_cols(book)
    if "book" not in df.columns:
        return pd.DataFrame()
    df["book_group"] = np.where(df["book"].astype(str).eq("LOCAL_DIAGNOSTIC_FLY"), "local_diagnostic", "standardized_books")
    keys = [c for c in ["run", "currency", "month", "slice_selection", "selected_slices_from_run", "book_group", "book"] if c in df.columns]
    event_cols = [
        c for c in df.columns
        if c.startswith("hedge_trigger_flip_")
        or c.startswith("hedge_diff_notional_gt_")
        or c.startswith("hedge_slippage_gt_")
        or c.startswith("gamma_pnl_proxy_gt_")
    ]
    agg = {"n": ("book", "size")}
    if "nextday_available" in df.columns:
        agg["nextday_pair_rate"] = ("nextday_available", "mean")
    for col in event_cols:
        agg[col + "_rate"] = (col, "mean")
    for col in ["hedge_slippage_scaled", "gamma_pnl_proxy", "hedge_diff_notional_scaled", "loss_common_diff"]:
        if col in df.columns:
            agg[f"p50_{col}"] = (col, "median")
            agg[f"p90_{col}"] = (col, q90)
    return df.groupby(keys, dropna=False).agg(**agg).reset_index()


def make_slice_selection_contrast(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    df = ensure_selection_cols(summary)
    if "slice_selection" not in df.columns:
        return pd.DataFrame()
    keys = [c for c in ["currency", "month", "selected_slices_requested", "selected_slices_from_run", "slice_selection"] if c in df.columns]
    metrics = [c for c in [
        "high_score_p90_gamma_diff_scaled", "high_score_p90_cxx_diff_scaled", "high_score_p90_rnd_diff_scaled",
        "high_score_small_price_large_gamma_scaled_rate", "high_score_small_price_large_curvature_rate",
        "book_p90_hedge_slippage_scaled", "book_p90_gamma_pnl_proxy", "book_p90_hedge_diff_notional_scaled",
        "book_hedge_diff_notional_gt_0p05_rate", "book_hedge_trigger_flip_0p5_rate",
    ] if c in df.columns]
    if not metrics:
        return pd.DataFrame()
    return df.groupby(keys, dropna=False)[metrics].median().reset_index()


def _safe_qcut(s: pd.Series, qn: int = 3, labels=None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if labels is None:
        labels = list(range(qn))
    try:
        return pd.qcut(x, q=qn, labels=labels, duplicates="drop")
    except Exception:
        return pd.Series(pd.NA, index=s.index)




def collect_geometry_summaries(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream strike_pairwise files and compute geometry summaries per run."""
    strike_cols = {
        "relspread_floor", "h_norm", "gamma_diff_scaled", "cxx_diff_scaled", "rnd_diff_scaled",
        "price_diff_box", "amp_x_h2", "small_price_large_gamma_scaled", "small_price_large_curvature",
        "ranking_score", "ranking_score_rel_h2", "ranking_score_abs_h2", "core_repair_slice",
        "arm_a", "arm_b", "DTE", "k"
    }
    geo_parts, reg_parts = [], []
    for f in sorted(root.rglob("strike_pairwise_dispersion.csv")):
        if any(part.startswith("_") for part in f.parts):
            continue
        try:
            df = pd.read_csv(f, usecols=lambda c: c in strike_cols)
        except Exception as e:
            print(f"[warn] geometry read failed {f}: {e}")
            continue
        run = f.parent.name
        meta = parse_run_name(run)
        for k, v in meta.items():
            if k not in df.columns:
                df.insert(0, k, v)
        g, r = make_geometry_control_summary(df)
        if not g.empty:
            geo_parts.append(g)
        if not r.empty:
            reg_parts.append(r)
    geo = pd.concat(geo_parts, ignore_index=True) if geo_parts else pd.DataFrame()
    reg = pd.concat(reg_parts, ignore_index=True) if reg_parts else pd.DataFrame()
    return geo, reg
def make_geometry_control_summary(strike: pd.DataFrame):
    if strike.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = ensure_selection_cols(strike)
    if "core_repair_slice" in df.columns:
        df = df[df["core_repair_slice"].astype(bool)].copy()
    needed = ["gamma_diff_scaled", "relspread_floor", "h_norm"]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame(), pd.DataFrame()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[(df["gamma_diff_scaled"].notna()) & (df["relspread_floor"] > 0) & (df["h_norm"] > 0)].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["inv_h2"] = 1.0 / (df["h_norm"].astype(float) ** 2)
    df["log_gamma_scaled"] = np.log1p(df["gamma_diff_scaled"].clip(lower=0))
    df["log_spread"] = np.log(df["relspread_floor"].clip(lower=1e-12))
    df["log_inv_h2"] = np.log(df["inv_h2"].clip(lower=1e-12))
    rows = []
    reg_rows = []
    for run, g in df.groupby("run", dropna=False):
        g = g.copy()
        if len(g) < 50:
            continue
        g["spread_bin"] = _safe_qcut(g["relspread_floor"], 3, labels=["low_spread", "mid_spread", "high_spread"])
        g["inv_h2_bin"] = _safe_qcut(g["inv_h2"], 3, labels=["low_inv_h2", "mid_inv_h2", "high_inv_h2"])
        for keys, gg in g.groupby(["spread_bin", "inv_h2_bin"], dropna=True):
            if len(gg) == 0:
                continue
            row = parse_run_name(str(run))
            row.update({
                "run": run, "spread_bin": str(keys[0]), "inv_h2_bin": str(keys[1]), "n": len(gg),
                "p50_gamma_diff_scaled": float(np.nanmedian(gg["gamma_diff_scaled"])),
                "p90_gamma_diff_scaled": float(np.nanpercentile(gg["gamma_diff_scaled"], 90)),
                "small_price_large_gamma_scaled_rate": float(np.nanmean(gg.get("small_price_large_gamma_scaled", pd.Series(False, index=gg.index)).astype(bool))) if "small_price_large_gamma_scaled" in gg else np.nan,
                "p50_relspread_floor": float(np.nanmedian(gg["relspread_floor"])),
                "p50_h_norm": float(np.nanmedian(gg["h_norm"])),
                "p50_inv_h2": float(np.nanmedian(gg["inv_h2"])),
            })
            rows.append(row)
        rg = g.dropna(subset=["log_gamma_scaled", "log_spread", "log_inv_h2"])
        if len(rg) >= 100:
            X = np.column_stack([np.ones(len(rg)), rg["log_spread"].to_numpy(float), rg["log_inv_h2"].to_numpy(float)])
            y = rg["log_gamma_scaled"].to_numpy(float)
            try:
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                yhat = X @ beta
                ssr = float(np.sum((y - yhat) ** 2)); sst = float(np.sum((y - y.mean()) ** 2))
                r2 = 1 - ssr / sst if sst > 0 else np.nan
            except Exception:
                beta = [np.nan, np.nan, np.nan]; r2 = np.nan
            row = parse_run_name(str(run))
            row.update({"run": run, "n": len(rg), "beta_log_spread": float(beta[1]), "beta_log_inv_h2": float(beta[2]), "r2": r2})
            reg_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(reg_rows)


def plot_rho_vs_baseline(rho: pd.DataFrame, outdir: Path):
    if plt is None or rho.empty:
        return
    fig_dir = ensure_dir(outdir / "figures")
    for metric, fname, ylabel in [
        ("p90_gamma_diff_scaled", "rho_vs_baseline_gamma.png", "p90 gamma_diff_scaled"),
        ("p90_hedge_slippage_scaled", "rho_vs_baseline_hedge_slippage.png", "p90 hedge_slippage_scaled"),
        ("p90_hedge_diff_notional_scaled", "rho_vs_baseline_hedge_diff_notional.png", "p90 hedge_diff_notional_scaled"),
        ("p90_gamma_pnl_proxy", "rho_vs_baseline_gamma_pnl_proxy.png", "p90 gamma_pnl_proxy"),
    ]:
        if metric not in rho.columns:
            continue
        plt.figure(figsize=(8, 5))
        for baseline, g0 in rho.groupby("baseline"):
            med = g0.groupby("rho")[metric].median().reset_index().sort_values("rho")
            plt.plot(np.log10(med["rho"].astype(float)), med[metric].astype(float), marker="o", linewidth=3, label=f"{baseline} vs Smooth")
            for run, g in g0.groupby("run"):
                g = g.sort_values("rho")
                plt.plot(np.log10(g["rho"].astype(float)), g[metric].astype(float), marker=".", alpha=0.18)
        plt.xlabel("log10(rho)")
        plt.ylabel(ylabel)
        plt.title(ylabel + " by smoothness preference vs L1/L2 baselines")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=180)
        plt.close()


def plot_slice_selection(contrast: pd.DataFrame, outdir: Path):
    if plt is None or contrast.empty:
        return
    fig_dir = ensure_dir(outdir / "figures")
    metric = "high_score_p90_gamma_diff_scaled"
    if metric not in contrast.columns or "slice_selection" not in contrast.columns:
        return
    order = ["bottom", "random", "top"]
    data = [contrast.loc[contrast["slice_selection"].astype(str).eq(s), metric].dropna().to_numpy() for s in order]
    plt.figure(figsize=(7, 4))
    plt.boxplot(data, labels=order, showmeans=True)
    plt.ylabel(metric)
    plt.title("High-score-score selection contrast")
    plt.tight_layout()
    plt.savefig(fig_dir / "slice_selection_contrast.png", dpi=180)
    plt.close()


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing per-run output directories")
    ap.add_argument("--output-dir", default=None, help="Aggregate output directory; default ROOT/_aggregate")
    ap.add_argument("--coverage-thresholds", nargs="*", type=float, default=[0.70, 0.80, 0.90])
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--write-strike-raw", action="store_true", help="Read/write raw strike rows and enable geometry-control summaries. Required for ALL_geometry_*.csv outputs; off by default because the file can be large.")
    ap.add_argument("--skip-geometry", action="store_true", help="Skip spread/h^2 geometry-control summaries even if --write-strike-raw is used.")
    args = ap.parse_args(argv)
    root = Path(args.root)
    outdir = ensure_dir(args.output_dir or (root / "_aggregate"))

    repair_summary = collect_csv(root, "repair_summary.csv")
    arm = collect_csv(root, "arm_pair_summary.csv")
    protocol = collect_csv(root, "protocol_comparison_summary.csv")
    book = collect_csv(root, "book_pairwise_dispersion.csv")
    support = collect_csv(root, "arm_pair_common_support.csv")
    repair = collect_csv(root, "repair_arm_summary.csv")
    strike = pd.DataFrame()
    # Geometry controls need strike-level pairwise rows. They can be streamed per run
    # without writing the raw combined file; use --write-strike-raw only when you
    # also want ALL_strike_pairwise_dispersion.csv for manual inspection.
    if args.write_strike_raw:
        strike_cols = {
            "relspread_floor", "h_norm", "gamma_diff_scaled", "cxx_diff_scaled", "rnd_diff_scaled",
            "price_diff_box", "amp_x_h2", "small_price_large_gamma_scaled", "small_price_large_curvature",
            "ranking_score", "ranking_score_rel_h2", "ranking_score_abs_h2", "core_repair_slice",
            "arm_a", "arm_b", "DTE", "k"
        }
        strike = collect_csv_columns(root, "strike_pairwise_dispersion.csv", strike_cols)

    if not repair_summary.empty:
        repair_summary.to_csv(outdir / "ALL_repair_summary.csv", index=False)
        status = make_status_summary(repair_summary)
        if not status.empty:
            status.to_csv(outdir / "ALL_status_summary.csv", index=False)
    if not arm.empty:
        arm.to_csv(outdir / "ALL_arm_pair_summary.csv", index=False)
    if not protocol.empty:
        protocol.to_csv(outdir / "ALL_protocol_comparison_summary.csv", index=False)
    if not book.empty:
        book.to_csv(outdir / "ALL_book_pairwise_dispersion.csv", index=False)
        rb_cov = add_run_book_coverage(ensure_selection_cols(book)) if "nextday_available" in book.columns else pd.DataFrame()
        if not rb_cov.empty:
            cov_cols = [c for c in ["run", "currency", "month", "slice_selection", "selected_slices_from_run", "book", "run_book_nextday_pair_rate"] if c in rb_cov.columns]
            rb_cov[cov_cols].drop_duplicates().to_csv(outdir / "ALL_run_book_nextday_coverage.csv", index=False)
    if not support.empty:
        support.to_csv(outdir / "ALL_arm_pair_common_support.csv", index=False)
    if not repair.empty:
        repair.to_csv(outdir / "ALL_repair_arm_summary.csv", index=False)
    if not strike.empty and args.write_strike_raw:
        strike.to_csv(outdir / "ALL_strike_pairwise_dispersion.csv", index=False)

    book_group = make_book_group_summary(book)
    if not book_group.empty:
        book_group.to_csv(outdir / "ALL_book_group_summary.csv", index=False)
    std_book = make_standardized_book_per_book_summary(book)
    if not std_book.empty:
        std_book.to_csv(outdir / "ALL_standardized_book_per_book_summary.csv", index=False)
    decision = make_decision_flip_summary(book)
    if not decision.empty:
        decision.to_csv(outdir / "ALL_decision_flip_summary.csv", index=False)
    for th in args.coverage_thresholds:
        pct = int(round(th * 100))
        bgf = make_book_group_summary_coverage_filtered(book, th)
        if not bgf.empty:
            bgf.to_csv(outdir / f"ALL_book_group_summary_coverage{pct}.csv", index=False)
        sbf = make_standardized_book_per_book_summary(book, threshold=th)
        if not sbf.empty:
            sbf.to_csv(outdir / f"ALL_standardized_book_per_book_summary_coverage{pct}.csv", index=False)

    rho_base = make_rho_vs_baseline_summary(arm, book)
    if not rho_base.empty:
        rho_base.to_csv(outdir / "ALL_rho_vs_baseline_summary.csv", index=False)
    contrast = make_slice_selection_contrast(repair_summary)
    if not contrast.empty:
        contrast.to_csv(outdir / "ALL_slice_selection_contrast.csv", index=False)
    geom_double, geom_reg = pd.DataFrame(), pd.DataFrame()
    if not args.skip_geometry:
        if args.write_strike_raw and not strike.empty:
            geom_double, geom_reg = make_geometry_control_summary(strike)
        else:
            # Stream per-run strike files so geometry outputs are generated by default
            # without materializing a potentially huge ALL_strike_pairwise file.
            geom_double, geom_reg = collect_geometry_summaries(root)
    if not geom_double.empty:
        geom_double.to_csv(outdir / "ALL_geometry_double_sort_summary.csv", index=False)
    if not geom_reg.empty:
        geom_reg.to_csv(outdir / "ALL_geometry_regression_summary.csv", index=False)

    if not args.no_plots:
        plot_rho_vs_baseline(rho_base, outdir)
        plot_slice_selection(contrast, outdir)

    print(f"[aggregate] wrote outputs to {outdir}")
    for name, df in [("repair_summary", repair_summary), ("arm", arm), ("protocol", protocol), ("book", book), ("strike", strike), ("rho_vs_baseline", rho_base), ("contrast", contrast), ("geometry_double", geom_double), ("geometry_reg", geom_reg)]:
        print(f"  {name}: {len(df):,} rows")
    return 0




# =============================================================================
# VaR-style aggregation
# =============================================================================
# Aggregates var_style_* outputs from the repair analysis.

_main_base = main


def make_var_style_capital_summary(var_pair: pd.DataFrame) -> pd.DataFrame:
    if var_pair.empty:
        return pd.DataFrame()
    df = ensure_selection_cols(var_pair)
    keys = [c for c in ["run", "currency", "month", "slice_selection", "selected_slices_from_run", "book", "loss_type"] if c in df.columns]
    agg = {"n_pairs": ("book", "size")}
    for col in df.columns:
        if re.match(r"var_a[0-9P]+_diff_scaled", col):
            agg[f"p50_{col}"] = (col, "median")
            agg[f"p90_{col}"] = (col, q90)
        elif re.match(r"cvar_a[0-9P]+_diff_scaled", col):
            agg[f"p90_{col}"] = (col, q90)
        elif col.startswith("capital_flip_"):
            agg[f"{col}_rate"] = (col, "mean")
    return df.groupby(keys, dropna=False).agg(**agg).reset_index() if agg else pd.DataFrame()


def make_var_style_breach_summary(var_breach: pd.DataFrame) -> pd.DataFrame:
    if var_breach.empty:
        return pd.DataFrame()
    df = ensure_selection_cols(var_breach)
    keys = [c for c in ["run", "currency", "month", "slice_selection", "selected_slices_from_run", "book", "loss_type"] if c in df.columns]
    agg = {"n_breach_pairs": ("book", "size")}
    for col in df.columns:
        if col.startswith("breach_flip_") or col.startswith("breach_a_") or col.startswith("breach_b_"):
            agg[f"{col}_rate"] = (col, "mean")
    return df.groupby(keys, dropna=False).agg(**agg).reset_index() if agg else pd.DataFrame()


def make_var_style_arm_agg(var_arm: pd.DataFrame) -> pd.DataFrame:
    if var_arm.empty:
        return pd.DataFrame()
    df = ensure_selection_cols(var_arm)
    keys = [c for c in ["run", "currency", "month", "slice_selection", "selected_slices_from_run", "book", "loss_type", "arm"] if c in df.columns]
    agg = {"n_rows": ("book", "size"), "median_n_loss": ("n_loss", "median")}
    for col in df.columns:
        if re.match(r"var_a[0-9P]+$", col) or re.match(r"cvar_a[0-9P]+$", col):
            agg[f"median_{col}"] = (col, "median")
    return df.groupby(keys, dropna=False).agg(**agg).reset_index()


def _parse_agg_root_out(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--root", required=True)
    ap.add_argument("--output-dir", default=None)
    ns, _ = ap.parse_known_args(argv)
    root = Path(ns.root)
    return root, ensure_dir(ns.output_dir or (root / "_aggregate"))


def main(argv: Optional[Iterable[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    root, outdir = _parse_agg_root_out(raw)
    ret = _main_base(raw)
    if ret != 0:
        return ret

    var_arm = collect_csv(root, "var_style_arm_summary.csv")
    var_pair = collect_csv(root, "var_style_pairwise.csv")
    var_breach = collect_csv(root, "var_style_breach_pairwise.csv")
    var_run = collect_csv(root, "var_style_run_summary.csv")

    if not var_arm.empty:
        var_arm.to_csv(outdir / "ALL_var_style_arm_summary.csv", index=False)
        arm_agg = make_var_style_arm_agg(var_arm)
        if not arm_agg.empty:
            arm_agg.to_csv(outdir / "ALL_var_style_arm_by_run_book.csv", index=False)
    if not var_pair.empty:
        var_pair.to_csv(outdir / "ALL_var_style_pairwise.csv", index=False)
        cap = make_var_style_capital_summary(var_pair)
        if not cap.empty:
            cap.to_csv(outdir / "ALL_var_style_capital_summary.csv", index=False)
    if not var_breach.empty:
        var_breach.to_csv(outdir / "ALL_var_style_breach_pairwise.csv", index=False)
        breach = make_var_style_breach_summary(var_breach)
        if not breach.empty:
            breach.to_csv(outdir / "ALL_var_style_breach_summary.csv", index=False)
    if not var_run.empty:
        var_run.to_csv(outdir / "ALL_var_style_run_summary.csv", index=False)

    if any(not x.empty for x in [var_arm, var_pair, var_breach, var_run]):
        print(f"[var-style aggregate] arm={len(var_arm):,}, pair={len(var_pair):,}, breach={len(var_breach):,}, run={len(var_run):,}")
    else:
        print("[var-style aggregate] no var_style_* files found")
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
