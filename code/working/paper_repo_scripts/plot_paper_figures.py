"""Generate paper figures from repair-risk aggregate CSV files.

The script reads the final aggregate outputs used by the manuscript and writes
vector PDF figures into the paper repository.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

WORKING_ROOT = Path(__file__).resolve().parents[1]
if str(WORKING_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKING_ROOT))
from artifact_paths import ARTIFACT_ROOT, FIGURES, RESULTS  # noqa: E402

DEFAULT_RESULTS_ROOT = RESULTS
DEFAULT_DIAGNOSTICS_ROOT = Path(
    os.environ.get("REPAIR_RISK_DIAGNOSTICS_ROOT", str(ARTIFACT_ROOT))
)
BLUE = "#0072B2"
ORANGE = "#E69F00"
RED = "#D55E00"
GRAY = "#4D4D4D"
LIGHT_GRAY = "#D8D8D8"
PAPER_SERIF = ["Linux Libertine O", "Times New Roman", "STIXGeneral", "DejaVu Serif"]
HEATMAP_BLUES = LinearSegmentedColormap.from_list(
    "repair_blues", ["#F7FBFF", "#DEEBF7", "#9ECAE1", "#4292C6", "#08519C"]
)


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _register_paper_fonts() -> None:
    font_dir = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs"
        / "MiKTeX"
        / "fonts"
        / "opentype"
        / "public"
        / "libertine"
    )
    if not font_dir.exists():
        return
    for font_path in font_dir.glob("LinLibertine_*.otf"):
        font_manager.fontManager.addfont(font_path)


def _median_by(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    work = df.copy()
    work[value_col] = _num(work[value_col])
    return (
        work.dropna(subset=[value_col])
        .groupby(group_cols, as_index=False)[value_col]
        .median()
    )


def _style() -> None:
    _register_paper_fonts()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": PAPER_SERIF,
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _finish_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def _classical_agg(results_root: Path) -> Path:
    preferred = results_root / "strict_controls_v38_18m_combined_20260619" / "_aggregate"
    if preferred.exists():
        return preferred
    return results_root / "strict_controls_v38" / "_aggregate"


def fig_rho_path(results_root: Path, out_dir: Path) -> None:
    df = pd.read_csv(_classical_agg(results_root) / "ALL_rho_vs_baseline_summary.csv")
    df["rho"] = _num(df["rho"])
    rho_order = [0.01, 0.1, 1.0, 10.0]
    baselines = ["L1", "L2"]
    colors = {"L1": BLUE, "L2": RED}
    markers = {"L1": "o", "L2": "s"}
    linestyles = {"L1": "-", "L2": "--"}
    metrics = [
        ("p90_gamma_diff_scaled", "p90 scaled spot-gamma gap"),
        ("p90_hedge_slippage_scaled", "p90 scaled hedge-gap impact"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.25), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        med = _median_by(df, ["baseline", "rho"], metric)
        for baseline in baselines:
            sub = med[med["baseline"] == baseline].set_index("rho").reindex(rho_order)
            ax.plot(
                range(len(rho_order)),
                sub[metric].to_numpy(),
                marker=markers[baseline],
                markersize=4.2,
                linewidth=1.8,
                linestyle=linestyles[baseline],
                color=colors[baseline],
                label=f"{baseline} baseline",
            )
        ax.set_title(ylabel)
        ax.set_xticks(range(len(rho_order)))
        ax.set_xticklabels(["0.01", "0.1", "1", "10"])
        ax.set_xlabel(r"smoothness weight $\rho$")
        _finish_axis(ax)
    axes[0].set_ylabel("median across evaluations")
    axes[1].legend(frameon=False, loc="upper left", handlelength=2.3)
    fig.subplots_adjust(wspace=0.28, bottom=0.23)
    _save(fig, out_dir / "fig_rho_path.pdf")


def fig_geometry_protocol(results_root: Path, out_dir: Path) -> None:
    agg = _classical_agg(results_root)
    proto = pd.read_csv(agg / "ALL_protocol_comparison_summary.csv")
    contrast = pd.read_csv(agg / "ALL_slice_selection_contrast.csv")

    proto["p90_gamma_diff_scaled"] = _num(proto["p90_gamma_diff_scaled"])
    pvals = (
        proto[proto["protocol_group"].isin(["old_sparse_danger", "revised_spread_h2_danger"])]
        .dropna(subset=["p90_gamma_diff_scaled"])
        .groupby("protocol_group")["p90_gamma_diff_scaled"]
        .median()
    )

    selection_order = ["bottom", "random", "top"]
    contrast["danger_p90_gamma_diff_scaled"] = _num(contrast["danger_p90_gamma_diff_scaled"])
    contrast["book_p90_hedge_slippage_scaled"] = _num(contrast["book_p90_hedge_slippage_scaled"])
    cmed = contrast.groupby("slice_selection")[
        ["danger_p90_gamma_diff_scaled", "book_p90_hedge_slippage_scaled"]
    ].median()

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35))
    axes[0].bar(
        ["sparsity", "RelSpread/h^2"],
        [pvals.get("old_sparse_danger"), pvals.get("revised_spread_h2_danger")],
        color=[GRAY, BLUE],
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].set_title("Protocol")
    axes[0].set_ylabel("median p90 scaled spot-gamma gap")
    _finish_axis(axes[0])

    x = range(len(selection_order))
    width = 0.35
    axes[1].bar(
        [i - width / 2 for i in x],
        [cmed.loc[s, "danger_p90_gamma_diff_scaled"] for s in selection_order],
        width=width,
        color=BLUE,
        edgecolor="black",
        linewidth=0.4,
        label="spot-gamma gap",
    )
    axes[1].bar(
        [i + width / 2 for i in x],
        [cmed.loc[s, "book_p90_hedge_slippage_scaled"] for s in selection_order],
        width=width,
        color=ORANGE,
        edgecolor="black",
        linewidth=0.4,
        hatch="//",
        label="hedge impact",
    )
    axes[1].set_title("Slice-selection controls")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(["low", "random", "high"])
    _finish_axis(axes[1])
    axes[1].legend(frameon=False)

    fig.subplots_adjust(wspace=0.36, bottom=0.24)
    _save(fig, out_dir / "fig_geometry_protocol.pdf")


def fig_feasible_range_scatter(diagnostics_root: Path, out_dir: Path) -> None:
    merged_path = (
        diagnostics_root
        / "feasible_curvature_range_18m_20260619"
        / "feasible_curvature_range_merged_pairwise.csv"
    )
    if not merged_path.exists():
        merged_path = (
            diagnostics_root
            / "feasible_curvature_range_20260613"
            / "feasible_curvature_range_merged_pairwise.csv"
        )
    df = pd.read_csv(merged_path)
    source_family = "classical_all_selected_roots"
    if source_family not in set(df["source_family"].dropna().astype(str)):
        source_family = "classical_all_12clusters"
    df = df[
        (df["source_family"] == source_family)
        & (df["pair_label"] == "L2_vs_SMOOTH_RHO_10")
        & (df["selection"] == "top")
        & (df["range_status"] == "ok")
    ].copy()
    df["feasible_cxx_range"] = _num(df["feasible_cxx_range"])
    df["abs_cxx_diff"] = _num(df["cxx_diff"]).abs()
    n_obs = len(df)
    spearman = df["feasible_cxx_range"].corr(df["abs_cxx_diff"], method="spearman")
    log_corr = np.log1p(df["feasible_cxx_range"].clip(lower=0)).corr(
        np.log1p(df["abs_cxx_diff"].clip(lower=0))
    )
    log_r2 = log_corr * log_corr
    plot = df.dropna(subset=["feasible_cxx_range", "abs_cxx_diff"])
    plot = plot[(plot["feasible_cxx_range"] > 0) & (plot["abs_cxx_diff"] > 0)].copy()

    x = plot["feasible_cxx_range"].to_numpy()
    y = np.clip(plot["abs_cxx_diff"].to_numpy(), 1e-6, None)
    x_lower = 1e-5
    y_lower = 1e-6
    upper = 1e2
    grid = np.geomspace(x_lower, upper, 200)

    fig, ax = plt.subplots(figsize=(3.28, 2.28))
    ax.scatter(x, y, s=5, color=BLUE, alpha=0.23, linewidths=0, rasterized=True)
    ax.plot(grid, grid, color=RED, linewidth=1.1, linestyle="--", label=r"$y=x$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lower, upper)
    ax.set_ylim(y_lower, upper)
    ax.set_xlabel(r"LP feasible $D_2$ range")
    ax.set_ylabel(r"observed absolute $D_2$ repair gap")
    ax.grid(True, which="major", color=LIGHT_GRAY, linewidth=0.45, alpha=0.75)
    ax.grid(True, which="minor", color=LIGHT_GRAY, linewidth=0.25, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)
    ax.text(
        0.04,
        0.96,
        f"$n_{{\\mathrm{{obs}}}}={n_obs:,}$\nSpearman={spearman:.4f}\nlog-log $R^2$={log_r2:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.2},
    )
    ax.legend(frameon=False, loc="lower right", handlelength=2.0)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.20, top=0.98)
    _save(fig, out_dir / "fig_feasible_range_scatter.pdf")


def fig_nnproj_pairs(results_root: Path, out_dir: Path) -> None:
    aggregate = results_root / "nn_targeted" / "_aggregate"
    if not aggregate.exists():
        aggregate = results_root / "nnproj_oos_v42" / "_aggregate"
    df = pd.read_csv(aggregate / "ALL_ml_arm_pair_summary.csv")
    strike = df[df["level"] == "strike"].copy()
    for col in ["p90_gamma_diff_scaled", "book_p90_hedge_slippage_scaled"]:
        strike[col] = _num(strike[col])

    pairs = [
        ("L2", "NNPROJ", "L2 vs NNPROJ"),
        ("NNPROJ", "SMOOTH_RHO_10", "NNPROJ vs Smooth10"),
    ]
    gamma_vals = []
    hedge_vals = []
    labels = []
    for arm_a, arm_b, label in pairs:
        sub = strike[(strike["arm_a"] == arm_a) & (strike["arm_b"] == arm_b)]
        labels.append(label)
        gamma_vals.append(sub["p90_gamma_diff_scaled"].median())
        hedge_vals.append(sub["book_p90_hedge_slippage_scaled"].median())

    metrics = ["Spot-gamma gap", "Hedge-gap impact"]
    values = {
        labels[0]: [gamma_vals[0], hedge_vals[0]],
        labels[1]: [gamma_vals[1], hedge_vals[1]],
    }

    fig, ax = plt.subplots(figsize=(3.28, 1.58))
    y = [0, 1]
    height = 0.32
    offsets = [-height / 2, height / 2]
    colors = [BLUE, RED]
    hatches = ["", "//"]
    for idx, label in enumerate(labels):
        bars = ax.barh(
            [pos + offsets[idx] for pos in y],
            values[label],
            height=height,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.4,
            hatch=hatches[idx],
            label=label.replace("NNPROJ", "NN-Proj").replace("Smooth10", r"Smooth$_{10}$"),
        )
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.006, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=7.2)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlabel("median p90 scaled metric")
    ax.set_xlim(0, max(max(gamma_vals), max(hedge_vals)) * 1.30)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.5,
        columnspacing=0.9,
        fontsize=7.0,
    )
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.25, top=0.84)
    _save(fig, out_dir / "fig_nnproj_pairs.pdf")


def fig_book_consequence(results_root: Path, out_dir: Path) -> None:
    agg = _classical_agg(results_root)
    std = pd.read_csv(agg / "ALL_standardized_book_per_book_summary_coverage90.csv")
    group = pd.read_csv(agg / "ALL_book_group_summary_coverage90.csv")

    books = [
        ("ATM_STRADDLE", "ATM"),
        ("PUT_SPREAD_5_10", "Put spread"),
        ("PUT_FLY_0_5_10", "Put butterfly"),
        ("RISK_REVERSAL_5", "Risk reversal"),
        ("LOCAL_DANGER_FLY", "Mechanism fly"),
    ]
    rows = []
    for book, label in books:
        source = std[std["book"] == book]
        if source.empty:
            source = group[group["book"] == book]
        rows.append(
            {
                "label": label,
                "hedge": _num(source["p90_hedge_slippage_scaled"]).median(),
                "gamma_pnl": _num(source["p90_gamma_pnl_proxy"]).median(),
            }
        )
    plot = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35))
    colors = [BLUE, BLUE, BLUE, BLUE, RED]
    hatches = ["", "", "", "", "//"]
    bars0 = axes[0].bar(plot["label"], plot["hedge"], color=colors, edgecolor="black", linewidth=0.4)
    for bar, hatch in zip(bars0, hatches):
        bar.set_hatch(hatch)
    axes[0].set_title("Hedge channel")
    axes[0].set_ylabel("median p90 hedge impact")
    _finish_axis(axes[0])
    bars1 = axes[1].bar(plot["label"], plot["gamma_pnl"], color=colors, edgecolor="black", linewidth=0.4)
    for bar, hatch in zip(bars1, hatches):
        bar.set_hatch(hatch)
    axes[1].set_title("Curvature-risk proxy")
    axes[1].set_ylabel("median p90 proxy")
    _finish_axis(axes[1])
    for ax in axes:
        ax.tick_params(axis="x", rotation=22)
    fig.subplots_adjust(wspace=0.38, bottom=0.28)
    _save(fig, out_dir / "fig_book_consequence.pdf")


def var_relative_difference_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    """Select the final classical common-scenario comparison when needed."""
    if {"source", "comparison", "loss_type"}.issubset(df.columns):
        return df[
            (df["source"] == "classical_v38_extra_clusters")
            & (df["comparison"] == "L2_vs_SMOOTH_RHO_10")
            & (df["loss_type"] == "common")
        ].copy()
    return df.copy()


def fig_var_relative_difference_classical_common(results_root: Path, out_dir: Path) -> None:
    compact = results_root / "figure6_common_scenario_relative_difference.csv"
    packaged = results_root / "full_strike_arb_v50_20260709" / compact.name
    if packaged.exists():
        df = pd.read_csv(packaged)
    elif compact.exists():
        df = pd.read_csv(compact)
    else:
        source = pd.read_csv(
            Path(os.environ.get("REPAIR_RISK_DERIVED_ROOT", str(ARTIFACT_ROOT / "derived")))
            / "capital_boundary"
            / "capital_boundary_width_summary.csv"
        )
        df = var_relative_difference_plot_data(source)
    ordinary_books = [
        ("ATM_STRADDLE", "ATM straddle"),
        ("PUT_SPREAD_5_10", "Put spread"),
        ("PUT_FLY_0_5_10", "Put butterfly"),
        ("RISK_REVERSAL_5", "Risk reversal"),
    ]
    selection_order = ["bottom", "random", "top"]
    expected = pd.MultiIndex.from_product(
        [selection_order, [book for book, _ in ordinary_books]],
        names=["selection_regime", "book"],
    )
    observed = pd.MultiIndex.from_frame(df[["selection_regime", "book"]])
    if (
        len(df) != len(expected)
        or not observed.is_unique
        or set(observed) != set(expected)
    ):
        raise ValueError("Figure 6 source must contain one row for each of the 12 selection-book cells")
    if not (_num(df["n"]) == 36).all():
        raise ValueError("Figure 6 source must contain 36 currency-month cells per row")
    df["p90_omega"] = _num(df["p90_omega"])
    pivot = (
        df.pivot(index="selection_regime", columns="book", values="p90_omega")
        .reindex(index=selection_order, columns=[book for book, _ in ordinary_books])
    )

    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    image = ax.imshow(pivot.to_numpy(), cmap=HEATMAP_BLUES, vmin=0.0, vmax=0.30, aspect="auto")
    ax.set_xticks(range(len(ordinary_books)))
    ax.set_xticklabels([label for _, label in ordinary_books], rotation=30, ha="right")
    ax.set_yticks(range(len(selection_order)))
    ax.set_yticklabels(["low", "random", "high"])
    ax.set_xlabel("Book")
    ax.set_ylabel("Selection group")
    ax.set_xticks([i - 0.5 for i in range(1, len(ordinary_books))], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, len(selection_order))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    for row_idx, selection in enumerate(selection_order):
        for col_idx, (book, _) in enumerate(ordinary_books):
            value = pivot.loc[selection, book]
            label_color = "white" if value >= 0.16 else "#111111"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=label_color, fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.055, pad=0.04)
    cbar.set_label(r"p90 absolute relative" "\n" r"difference $\omega$")
    cbar.outline.set_linewidth(0.5)
    fig.subplots_adjust(left=0.18, right=0.93, bottom=0.32, top=0.98)
    _save(fig, out_dir / "fig_var_relative_difference_classical_common.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--diagnostics-root", type=Path, default=DEFAULT_DIAGNOSTICS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()

    _style()
    fig_rho_path(args.results_root, args.output_dir)
    fig_geometry_protocol(args.results_root, args.output_dir)
    fig_feasible_range_scatter(args.diagnostics_root, args.output_dir)
    fig_nnproj_pairs(args.results_root, args.output_dir)
    fig_book_consequence(args.results_root, args.output_dir)
    fig_var_relative_difference_classical_common(args.results_root, args.output_dir)
    print(f"Wrote figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
