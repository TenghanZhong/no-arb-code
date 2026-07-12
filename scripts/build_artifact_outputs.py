#!/usr/bin/env python3
"""Rebuild artifact figures and a table source from compact outputs."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "figure_sources"
GENERATED = ROOT / "generated"
FIGURES = GENERATED / "figures"
TABLES = GENERATED / "tables"
EXPECTED = ROOT / "outputs" / "expected_figures"

BLUE = "#0072B2"
ORANGE = "#E69F00"
RED = "#D55E00"
LIGHT_GRAY = "#D8D8D8"
HEATMAP_BLUES = LinearSegmentedColormap.from_list(
    "repair_blues", ["#F7FBFF", "#DEEBF7", "#9ECAE1", "#4292C6", "#08519C"]
)


def style() -> None:
    font_dir = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs"
        / "MiKTeX"
        / "fonts"
        / "opentype"
        / "public"
        / "libertine"
    )
    if font_dir.exists():
        for font_path in font_dir.glob("LinLibertine_*.otf"):
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Linux Libertine O", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
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
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)


def figure1() -> str:
    script = ROOT / "code" / "analysis" / "diagnostics" / "draw_mechanism_overview.py"
    spec = importlib.util.spec_from_file_location("artifact_mechanism_figure", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT_PDF = FIGURES / "fig_mechanism_overview.pdf"
    module.OUT_PNG = FIGURES / "fig_mechanism_overview.png"
    module.main()
    return "regenerated from Python source"


def figure2() -> str:
    source = ROOT / "scripts" / "validation_protocol_tikz_singlecol.tex"
    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        build = GENERATED / "latex"
        build.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(build), str(source)],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        shutil.copy2(build / "validation_protocol_tikz_singlecol.pdf", FIGURES / "validation_protocol_tikz_singlecol.pdf")
        return "compiled from standalone TikZ source"
    shutil.copy2(EXPECTED / "validation_protocol_tikz_singlecol.pdf", FIGURES / "validation_protocol_tikz_singlecol.pdf")
    return "copied supplied vector rendering (pdflatex unavailable)"


def figure3() -> dict[str, object]:
    frame = pd.read_csv(SOURCE / "figure3_rho_path.csv")
    order = [0.01, 0.1, 1.0, 10.0]
    metrics = [
        ("p90_gamma_diff_scaled", "p90 scaled spot-gamma gap"),
        ("p90_hedge_slippage_scaled", "p90 scaled hedge-gap impact"),
    ]
    colors = {"L1": BLUE, "L2": RED}
    markers = {"L1": "o", "L2": "s"}
    lines = {"L1": "-", "L2": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.25), sharex=True)
    reported: dict[str, object] = {"rows": int(len(frame)), "medians": {}}
    for ax, (metric, title) in zip(axes, metrics):
        medians = frame.groupby(["baseline", "rho"])[metric].median()
        reported["medians"][metric] = {
            f"{baseline}:{rho:g}": float(medians.loc[(baseline, rho)])
            for baseline in ("L1", "L2")
            for rho in order
        }
        for baseline in ("L1", "L2"):
            values = [medians.loc[(baseline, rho)] for rho in order]
            ax.plot(
                range(len(order)), values, marker=markers[baseline], markersize=4.2,
                linewidth=1.8, linestyle=lines[baseline], color=colors[baseline],
                label=f"{baseline} baseline",
            )
        ax.set_title(title)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["0.01", "0.1", "1", "10"])
        ax.set_xlabel(r"smoothness weight $\rho$")
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.45, alpha=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("median across evaluations")
    axes[1].legend(frameon=False, loc="upper left", handlelength=2.3)
    fig.subplots_adjust(wspace=0.28, bottom=0.23)
    save(fig, "fig_rho_path.pdf")
    return reported


def figure4() -> dict[str, object]:
    frame = pd.read_csv(SOURCE / "figure4_lp_range.csv")
    x_series = pd.to_numeric(frame["feasible_cxx_range"], errors="coerce")
    y_series = pd.to_numeric(frame["cxx_diff"], errors="coerce").abs()
    spearman = float(x_series.corr(y_series, method="spearman"))
    log_r2 = float(np.log1p(x_series.clip(lower=0)).corr(np.log1p(y_series.clip(lower=0))) ** 2)
    plot = pd.DataFrame({"x": x_series, "y": y_series}).dropna()
    plot = plot[(plot["x"] > 0) & (plot["y"] > 0)]
    x = plot["x"].to_numpy()
    y = np.clip(plot["y"].to_numpy(), 1e-6, None)
    grid = np.geomspace(1e-5, 1e2, 200)
    fig, ax = plt.subplots(figsize=(3.28, 2.28))
    ax.scatter(x, y, s=5, color=BLUE, alpha=0.23, linewidths=0, rasterized=True)
    ax.plot(grid, grid, color=RED, linewidth=1.1, linestyle="--", label=r"$y=x$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-5, 1e2)
    ax.set_ylim(1e-6, 1e2)
    ax.set_xlabel(r"LP feasible $D_2$ range")
    ax.set_ylabel(r"observed absolute $D_2$ repair gap")
    ax.grid(True, which="major", color=LIGHT_GRAY, linewidth=0.45, alpha=0.75)
    ax.grid(True, which="minor", color=LIGHT_GRAY, linewidth=0.25, alpha=0.35)
    ax.text(
        0.04, 0.96,
        f"$n_{{\\mathrm{{obs}}}}={len(frame):,}$\nSpearman={spearman:.4f}\nlog-log $R^2={log_r2:.4f}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.2},
    )
    ax.legend(frameon=False, loc="lower right", handlelength=2.0)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.20, top=0.98)
    save(fig, "fig_feasible_range_scatter.pdf")
    return {"rows": int(len(frame)), "spearman": spearman, "log_log_r2": log_r2}


def figure5() -> dict[str, object]:
    frame = pd.read_csv(SOURCE / "figure5_nnproj_pairs.csv")
    strike = frame[frame["level"] == "strike"].copy()
    pairs = [
        ("L2", "NNPROJ", "L2 vs NN-Proj"),
        ("NNPROJ", "SMOOTH_RHO_10", r"NN-Proj vs Smooth$_{10}$"),
    ]
    values: dict[str, list[float]] = {}
    reported: dict[str, object] = {"rows": int(len(frame)), "pairs": {}}
    for arm_a, arm_b, label in pairs:
        selected = strike[(strike["arm_a"] == arm_a) & (strike["arm_b"] == arm_b)]
        pair_values = [
            float(selected["p90_gamma_diff_scaled"].median()),
            float(selected["book_p90_hedge_slippage_scaled"].median()),
        ]
        values[label] = pair_values
        reported["pairs"][f"{arm_a}__{arm_b}"] = {
            "evaluations": int(len(selected)),
            "median_p90_gamma_gap": pair_values[0],
            "median_p90_hedge_gap_impact": pair_values[1],
        }
    fig, ax = plt.subplots(figsize=(3.28, 1.58))
    y = np.array([0.0, 1.0])
    height = 0.32
    for index, (label, pair_values) in enumerate(values.items()):
        bars = ax.barh(
            y + (-height / 2 if index == 0 else height / 2), pair_values,
            height=height, color=(BLUE if index == 0 else RED), edgecolor="black",
            linewidth=0.4, hatch=("" if index == 0 else "//"), label=label,
        )
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.006, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=7.2)
    ax.set_yticks(y)
    ax.set_yticklabels(["Spot-gamma gap", "Hedge-gap impact"])
    ax.set_xlabel("median p90 scaled metric")
    ax.set_xlim(0, max(max(item) for item in values.values()) * 1.30)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)
    ax.legend(
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
        handlelength=1.5, columnspacing=0.9, fontsize=7.0,
    )
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.25, top=0.84)
    save(fig, "fig_nnproj_pairs.pdf")
    return reported


def figure6() -> dict[str, object]:
    frame = pd.read_csv(SOURCE / "figure6_var_relative_difference.csv")
    books = [
        ("ATM_STRADDLE", "ATM straddle"),
        ("PUT_SPREAD_5_10", "Put spread"),
        ("PUT_FLY_0_5_10", "Put butterfly"),
        ("RISK_REVERSAL_5", "Risk reversal"),
    ]
    order = ["bottom", "random", "top"]
    pivot = frame.pivot(index="selection_regime", columns="book", values="p90_omega").reindex(
        index=order, columns=[book for book, _ in books]
    )
    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    image = ax.imshow(pivot.to_numpy(), cmap=HEATMAP_BLUES, vmin=0.0, vmax=0.30, aspect="auto")
    ax.set_xticks(range(len(books)))
    ax.set_xticklabels([label for _, label in books], rotation=30, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["low", "random", "high"])
    ax.set_xlabel("Book")
    ax.set_ylabel("Selection group")
    ax.set_xticks([index - 0.5 for index in range(1, len(books))], minor=True)
    ax.set_yticks([index - 0.5 for index in range(1, len(order))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    for row_index, selection in enumerate(order):
        for column_index, (book, _) in enumerate(books):
            value = float(pivot.loc[selection, book])
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color=("white" if value >= 0.16 else "#111111"), fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.055, pad=0.04)
    cbar.set_label(r"p90 absolute relative" "\n" r"difference $\omega$")
    cbar.outline.set_linewidth(0.5)
    fig.subplots_adjust(left=0.18, right=0.93, bottom=0.32, top=0.98)
    save(fig, "fig_var_relative_difference_classical_common.pdf")
    top_rr = frame[(frame["selection_regime"] == "top") & (frame["book"] == "RISK_REVERSAL_5")].iloc[0]
    return {
        "rows": int(len(frame)),
        "top_risk_reversal_median": float(top_rr["median_omega"]),
        "top_risk_reversal_p90": float(top_rr["p90_omega"]),
    }


def table1() -> None:
    summary = json.loads((ROOT / "outputs" / "full_rerun_summary.json").read_text(encoding="utf-8"))
    rows = [
        ("Classical smoothness", "median evaluation p90 gamma gap", summary["classical"]["median_evaluation_p90_smooth10_gamma_gap"], "108 evaluations"),
        ("Classical control", "median evaluation p90 gamma gap", summary["classical"]["median_evaluation_p90_l1_l2_control"], "108 evaluations"),
        ("Calendar excess", "median paired excess", summary["classical"]["median_calendar_month_excess"], "18 calendar months"),
        ("SVI target", "median evaluation p90 gamma gap", summary["svi"]["median_l2_svi_p90_gamma_gap"], "108 evaluations"),
        ("Targeted NN", "median p90 gamma gap L2 vs NN-Proj", summary["targeted_nn"]["median_p90_gamma_gap_l2_nnproj"], "10 evaluations"),
        ("Five-seed replication", "median seed-pair p90 D2 gap", summary["seed_replication"]["median_common_support_seed_pair_p90_d2_gap"], "180 refits"),
        ("LP range", "Spearman rank correlation", summary["lp_range"]["spearman_range_vs_abs_cxx"], "51,189 observations"),
        ("Hedge notional", "risk-reversal basis points", summary["classical"]["risk_reversal_p90_hedge_notional_bps"], "42 coverage-qualified runs"),
        ("VaR-style common", "median / p90", f"{summary['classical']['var_style']['common']['median']:.10g} / {summary['classical']['var_style']['common']['p90']:.10g}", "432 cells"),
        ("VaR-style repair-specific", "median / p90", f"{summary['classical']['var_style']['pipeline']['median']:.10g} / {summary['classical']['var_style']['pipeline']['p90']:.10g}", "417 cells"),
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["analysis", "metric", "value", "support"]).to_csv(TABLES / "table1_source.csv", index=False)


def main() -> None:
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    statuses = {"figure1": figure1(), "figure2": figure2()}
    summary = {
        "design_figures": statuses,
        "figure3": figure3(),
        "figure4": figure4(),
        "figure5": figure5(),
        "figure6": figure6(),
    }
    table1()
    (GENERATED / "quick_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote artifact outputs to {GENERATED}")


if __name__ == "__main__":
    main()
