"""Compute VaR-style relative differences and hedge-notional diagnostics.

The script reads aggregate outputs and reports (1) the absolute relative
difference ``omega = |K_m / K_L2 - 1|`` between VaR-style estimates and
(2) the gross spot notional associated with the risk-reversal hedge gap.
Synthetic self-checks validate arm ordering and units before data are loaded.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
from artifact_paths import DERIVED, RESULTS  # noqa: E402

CLASSICAL_AGG = RESULTS / "classical_18m" / "_aggregate"
NNPROJ_AGG = RESULTS / "nn_targeted" / "_aggregate"
PERSISTENCE_DIR = DERIVED / "hedge_persistence"
DEFAULT_OUTPUT = DERIVED / "var_style_diagnostics"

BOOK_ORDER = [
    "ATM_STRADDLE",
    "PUT_SPREAD_5_10",
    "PUT_FLY_0_5_10",
    "RISK_REVERSAL_5",
    "LOCAL_DIAGNOSTIC_FLY",
]
REGIME_ORDER = ["bottom", "random", "top"]


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def normalize_l2_pair(
    df: pd.DataFrame,
    target_arm: str,
    *,
    alpha: str = "0p95",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Keep L2-vs-target rows and orient VaR estimates as L2 and target."""
    var_a = f"var_a{alpha}_a"
    var_b = f"var_a{alpha}_b"
    required = [
        "run",
        "currency",
        "month",
        "slice_selection_from_run",
        "book",
        "loss_type",
        "arm_a",
        "arm_b",
        var_a,
        var_b,
    ]
    require_columns(df, required, "var_style_pairwise")
    pair_mask = (
        ((df["arm_a"] == "L2") & (df["arm_b"] == target_arm))
        | ((df["arm_b"] == "L2") & (df["arm_a"] == target_arm))
    )
    out = df.loc[pair_mask, required].copy()
    if out.empty:
        raise ValueError(f"No L2-vs-{target_arm} rows found.")
    l2_is_a = out["arm_a"] == "L2"
    out["comparison"] = f"L2_vs_{target_arm}"
    out["K_L2"] = np.where(l2_is_a, out[var_a], out[var_b])
    out["K_target"] = np.where(l2_is_a, out[var_b], out[var_a])
    out["selection_regime"] = out["slice_selection_from_run"].astype(str)
    out["abs_l2"] = out["K_L2"].abs()
    out["unstable_l2_denominator"] = out["abs_l2"] <= eps
    denom = out["abs_l2"].clip(lower=eps)
    out["omega_abs"] = (out["K_target"] - out["K_L2"]).abs() / denom
    out["omega_signed"] = (out["K_target"] - out["K_L2"]) / denom
    if (out["omega_abs"] < -1e-15).any():
        raise AssertionError("omega_abs cannot be negative.")
    return out


def summarize_relative_difference(long_df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "source",
        "comparison",
        "loss_type",
        "selection_regime",
        "book",
        "omega_abs",
        "unstable_l2_denominator",
    ]
    require_columns(long_df, required, "VaR-style relative-difference data")
    grouped = (
        long_df.groupby(["source", "comparison", "loss_type", "selection_regime", "book"], dropna=False)
        .agg(
            n=("omega_abs", "size"),
            n_unstable_l2=("unstable_l2_denominator", "sum"),
            median_omega=("omega_abs", "median"),
            p90_omega=("omega_abs", lambda x: float(np.quantile(x, 0.90))),
            max_omega=("omega_abs", "max"),
        )
        .reset_index()
    )
    return grouped


def plot_relative_difference_heatmaps(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Draw compact heatmaps without adding a seaborn dependency."""
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for (source, comparison, loss_type), sub in summary.groupby(["source", "comparison", "loss_type"]):
        pivot = sub.pivot(index="selection_regime", columns="book", values="p90_omega")
        pivot = pivot.reindex(index=REGIME_ORDER)
        ordered_books = [b for b in BOOK_ORDER if b in pivot.columns]
        pivot = pivot.reindex(columns=ordered_books)
        values = pivot.to_numpy(dtype=float)
        if values.size == 0 or np.all(np.isnan(values)):
            continue
        fig_w = max(6.5, 1.25 * len(ordered_books))
        fig, ax = plt.subplots(figsize=(fig_w, 3.4))
        vmax = float(np.nanquantile(values, 0.95))
        if not math.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(values)) if np.nanmax(values) > 0 else 1.0
        im = ax.imshow(values, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(ordered_books)))
        ax.set_xticklabels(ordered_books, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(REGIME_ORDER)))
        ax.set_yticklabels(REGIME_ORDER, fontsize=9)
        short_source = source.replace("classical_additional_clusters", "Classical").replace(
            "nn_targeted", "NN-Proj OOS"
        )
        short_comp = comparison.replace("SMOOTH_RHO_10", "Smooth10").replace("NNPROJ", "NN-Proj")
        ax.set_title(f"p90 absolute relative difference\n{short_source}: {short_comp}, {loss_type}", fontsize=10)
        ax.set_xlabel("Book")
        ax.set_ylabel("Slice selection")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"p90 $|K_m/K_{L2}-1|$")
        fig.tight_layout()
        stem = f"fig_var_relative_difference_{source}_{comparison}_{loss_type}".replace("/", "_")
        pdf_path = output_dir / f"{stem}.pdf"
        png_path = output_dir / f"{stem}.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.extend([pdf_path, png_path])
    return paths


def load_var_style_pairwise(root: Path) -> pd.DataFrame:
    path = root / "ALL_var_style_pairwise.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def compute_relative_difference_outputs(
    classical_root: Path,
    nnproj_root: Path,
    *,
    alpha: str,
    eps: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    classical = load_var_style_pairwise(classical_root)
    c_piece = normalize_l2_pair(classical, "SMOOTH_RHO_10", alpha=alpha, eps=eps)
    c_piece["source"] = "classical_additional_clusters"
    pieces.append(c_piece)

    nnproj = load_var_style_pairwise(nnproj_root)
    for target in ["SMOOTH_RHO_10", "NNPROJ"]:
        piece = normalize_l2_pair(nnproj, target, alpha=alpha, eps=eps)
        piece["source"] = "nn_targeted"
        pieces.append(piece)

    long_df = pd.concat(pieces, ignore_index=True)
    if not np.isfinite(long_df["omega_abs"]).all():
        bad = long_df.loc[~np.isfinite(long_df["omega_abs"])].head()
        raise AssertionError(f"Non-finite omega values detected:\n{bad}")
    summary = summarize_relative_difference(long_df)
    long_df.to_csv(output_dir / "var_relative_difference_long.csv", index=False)
    summary.to_csv(output_dir / "var_relative_difference_summary.csv", index=False)
    pivot = summary.pivot_table(
        index=["source", "comparison", "loss_type", "selection_regime"],
        columns="book",
        values="p90_omega",
        aggfunc="first",
    ).reset_index()
    pivot.to_csv(output_dir / "var_relative_difference_p90_pivot.csv", index=False)
    plot_relative_difference_heatmaps(summary, output_dir)
    return long_df, summary


def pair_filter(df: pd.DataFrame, arm_a: str, arm_b: str) -> pd.Series:
    return ((df["arm_a"] == arm_a) & (df["arm_b"] == arm_b)) | (
        (df["arm_a"] == arm_b) & (df["arm_b"] == arm_a)
    )


def load_break_even_anchor(persistence_dir: Path) -> pd.DataFrame:
    path = persistence_dir / "risk_reversal_hedge_persistence_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = [
        "coverage_filter",
        "n_runs_coverage_pass",
        "median_run_p90_abs_gap_bps",
        "median_run_p90_abs_gap_usd_per_1mm",
    ]
    require_columns(df, required, "hedge persistence summary")
    return df


def compute_direct_risk_reversal_gap(book_pairwise_path: Path, coverage_path: Path | None) -> pd.DataFrame:
    if not book_pairwise_path.exists():
        raise FileNotFoundError(book_pairwise_path)
    df = pd.read_csv(book_pairwise_path)
    required = [
        "run",
        "currency",
        "month",
        "slice_selection_from_run",
        "book",
        "arm_a",
        "arm_b",
        "hedge_diff_notional_scaled",
        "nextday_available",
    ]
    require_columns(df, required, "book_pairwise_dispersion")
    sub = df.loc[
        (df["book"] == "RISK_REVERSAL_5")
        & pair_filter(df, "L2", "SMOOTH_RHO_10")
        & (df["nextday_available"].astype(bool))
    ].copy()
    if sub.empty:
        raise ValueError("No risk-reversal L2-vs-SMOOTH_RHO_10 next-day rows found.")
    sub["abs_gap_fraction"] = sub["hedge_diff_notional_scaled"].abs()
    sub["abs_gap_bps"] = sub["abs_gap_fraction"] * 10000.0

    if coverage_path is not None and coverage_path.exists():
        cov = pd.read_csv(coverage_path)
        if {"run", "book_nextday_pair_rate"}.issubset(cov.columns):
            pass_runs = set(cov.loc[cov["book_nextday_pair_rate"] >= 0.90, "run"].astype(str))
            sub["coverage90_pass"] = sub["run"].astype(str).isin(pass_runs)
        else:
            sub["coverage90_pass"] = False
    else:
        sub["coverage90_pass"] = False

    rows = []
    for label, part in [
        ("all_available_rows", sub),
        ("coverage90_rows_if_available", sub.loc[sub["coverage90_pass"]]),
    ]:
        if part.empty:
            continue
        rows.append(
            {
                "direct_scope": label,
                "n_rows": len(part),
                "n_runs": part["run"].nunique(),
                "p90_gap_fraction_pooled_rows": float(np.quantile(part["abs_gap_fraction"], 0.90)),
                "p90_gap_bps_pooled_rows": float(np.quantile(part["abs_gap_bps"], 0.90)),
                "median_run_p90_gap_bps": float(
                    part.groupby("run")["abs_gap_bps"].quantile(0.90).median()
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_break_even_outputs(
    persistence_dir: Path,
    book_pairwise_path: Path,
    coverage_path: Path | None,
    *,
    output_dir: Path,
    thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor = load_break_even_anchor(persistence_dir)
    rows = []
    for _, row in anchor.iterrows():
        gap_bps = float(row["median_run_p90_abs_gap_bps"])
        gap_fraction = gap_bps / 10000.0
        if gap_fraction <= 0:
            raise AssertionError("Break-even gap must be positive.")
        for threshold in thresholds:
            rows.append(
                {
                    "scope": "coverage90_median_run_p90_anchor",
                    "coverage_filter": row["coverage_filter"],
                    "n_runs_coverage_pass": row["n_runs_coverage_pass"],
                    "gap_bps": gap_bps,
                    "usd_per_1mm": gap_fraction * 1_000_000.0,
                    "materiality_usd": threshold,
                    "break_even_gross_spot_notional_usd": threshold / gap_fraction,
                }
            )
    break_even = pd.DataFrame(rows)
    direct = compute_direct_risk_reversal_gap(book_pairwise_path, coverage_path)
    break_even.to_csv(output_dir / "break_even_notional_risk_reversal.csv", index=False)
    direct.to_csv(output_dir / "risk_reversal_gap_direct_scope_check.csv", index=False)
    return break_even, direct


def write_markdown_summary(
    output_dir: Path,
    relative_difference_summary: pd.DataFrame,
    break_even: pd.DataFrame,
    direct: pd.DataFrame,
) -> None:
    def fmt_usd(x: float) -> str:
        return f"${x:,.0f}"

    lines = [
        "# VaR-Style and Hedge-Notional Diagnostics",
        "",
        "## Absolute Relative Difference",
        "",
        "The diagnostic is `abs(K_target / K_L2 - 1)` using arm-level VaR-style estimates.",
        "",
    ]
    key = relative_difference_summary.loc[
        (relative_difference_summary["book"] == "RISK_REVERSAL_5")
        & (relative_difference_summary["selection_regime"] == "top")
        & (relative_difference_summary["loss_type"] == "common")
    ].copy()
    if not key.empty:
        lines.append("Top-selection risk-reversal p90 absolute relative differences under common loss:")
        lines.append("")
        for _, row in key.sort_values(["source", "comparison"]).iterrows():
            lines.append(
                f"- {row['source']} {row['comparison']}: p90 omega={row['p90_omega']:.4f}, "
                f"median={row['median_omega']:.4f}, n={int(row['n'])}."
            )
        lines.append("")

    lines.extend(
        [
            "## Break-even Notional",
            "",
            "Break-even notional uses the reported coverage-90 median run-level p90 "
            "L2--Smooth10 hedge-notional gap for the standardized risk reversal.",
            "",
        ]
    )
    anchor_rows = break_even.drop_duplicates(["gap_bps", "usd_per_1mm"])
    for _, row in anchor_rows.iterrows():
        lines.append(
            f"- Anchor gap: {row['gap_bps']:.4f} bps, "
            f"{fmt_usd(row['usd_per_1mm'])} per USD 1mm gross spot notional."
        )
    lines.append("")
    lines.append("| Materiality | Break-even gross spot notional |")
    lines.append("|---:|---:|")
    for _, row in break_even.iterrows():
        lines.append(
            f"| {fmt_usd(row['materiality_usd'])} | "
            f"{fmt_usd(row['break_even_gross_spot_notional_usd'])} |"
        )
    lines.append("")
    if not direct.empty:
        lines.append("Direct aggregate scope checks:")
        lines.append("")
        for _, row in direct.iterrows():
            lines.append(
                f"- {row['direct_scope']}: n_rows={int(row['n_rows'])}, n_runs={int(row['n_runs'])}, "
                f"pooled p90={row['p90_gap_bps_pooled_rows']:.4f} bps, "
                f"median run p90={row['median_run_p90_gap_bps']:.4f} bps."
            )
    (output_dir / "VAR_STYLE_AND_HEDGE_NOTIONAL_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_self_checks() -> None:
    toy = pd.DataFrame(
        [
            {
                "run": "r1",
                "currency": "BTC",
                "month": "2024-01",
                "slice_selection_from_run": "top",
                "book": "RISK_REVERSAL_5",
                "loss_type": "common",
                "arm_a": "L2",
                "arm_b": "SMOOTH_RHO_10",
                "var_a0p95_a": 100.0,
                "var_a0p95_b": 110.0,
            },
            {
                "run": "r2",
                "currency": "BTC",
                "month": "2024-01",
                "slice_selection_from_run": "top",
                "book": "RISK_REVERSAL_5",
                "loss_type": "common",
                "arm_a": "SMOOTH_RHO_10",
                "arm_b": "L2",
                "var_a0p95_a": 90.0,
                "var_a0p95_b": 100.0,
            },
        ]
    )
    out = normalize_l2_pair(toy, "SMOOTH_RHO_10")
    got = out["omega_abs"].round(10).tolist()
    if got != [0.1, 0.1]:
        raise AssertionError(f"Orientation self-check failed: {got}")
    summary = summarize_relative_difference(out.assign(source="toy"))
    if not math.isclose(float(summary["p90_omega"].iloc[0]), 0.1, abs_tol=1e-12):
        raise AssertionError("Relative-difference p90 self-check failed.")

    gap_bps = 21.253428
    gap_fraction = gap_bps / 10000.0
    be_10k = 10000.0 / gap_fraction
    if not (4_704_000 <= be_10k <= 4_706_000):
        raise AssertionError(f"Break-even self-check failed: {be_10k}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classical-aggregate", type=Path, default=CLASSICAL_AGG)
    parser.add_argument("--nnproj-aggregate", type=Path, default=NNPROJ_AGG)
    parser.add_argument("--persistence-dir", type=Path, default=PERSISTENCE_DIR)
    parser.add_argument(
        "--book-pairwise",
        type=Path,
        default=CLASSICAL_AGG / "ALL_book_pairwise_dispersion.csv",
    )
    parser.add_argument(
        "--coverage-file",
        type=Path,
        default=CLASSICAL_AGG / "ALL_run_book_nextday_coverage.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", default="0p95")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[10000.0, 25000.0, 50000.0, 100000.0, 250000.0],
    )
    parser.add_argument("--self-check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_self_checks()
    if args.self_check_only:
        print("Self-checks passed.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    relative_long, relative_summary = compute_relative_difference_outputs(
        args.classical_aggregate,
        args.nnproj_aggregate,
        alpha=args.alpha,
        eps=args.eps,
        output_dir=args.output_dir,
    )
    break_even, direct = compute_break_even_outputs(
        args.persistence_dir,
        args.book_pairwise,
        args.coverage_file,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
    )
    write_markdown_summary(args.output_dir, relative_summary, break_even, direct)
    print(f"Wrote outputs to {args.output_dir}")
    print(f"Relative-difference rows: {len(relative_long):,}; grouped rows: {len(relative_summary):,}")
    print(
        "Break-even anchor bps:",
        ", ".join(f"{x:.4f}" for x in break_even["gap_bps"].drop_duplicates()),
    )


if __name__ == "__main__":
    main()
