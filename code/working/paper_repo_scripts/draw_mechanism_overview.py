"""Draw the mechanism overview figure used as Figure 1 in the paper."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_PDF = Path("figures/fig_mechanism_overview.pdf")
OUT_PNG = Path("figures/fig_mechanism_overview.png")

BLUE = "#164A9B"
GREEN = "#1E6A34"
ORANGE = "#D65F00"
INK = "#202020"
ARROW = "#303030"
PANEL_EDGE = "#606060"
PANEL = "#FFFFFF"
BLUE_FILL = "#F4F8FF"
GREEN_FILL = "#F5FBF5"
ORANGE_FILL = "#FFF7F0"
PAPER_SERIF = ["Linux Libertine O", "Times New Roman", "STIXGeneral", "DejaVu Serif"]


def register_paper_fonts() -> None:
    """Make TeX's Libertine fonts visible to matplotlib when available."""
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


def add_box(ax, x, y, w, h, *, edge=BLUE, face=BLUE_FILL, lw=1.15, radius=0.018):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.014,rounding_size={radius}",
        edgecolor=edge,
        facecolor=face,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def add_arrow(ax, x0, y0, x1, y1, *, lw=1.25, scale=12):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=ARROW,
            shrinkA=7,
            shrinkB=7,
        )
    )


def panel_title(ax, x, y, text):
    ax.text(x, y, text, fontsize=13.4, fontweight="bold", va="center", color=INK)


def draw_panel_a(ax):
    add_box(ax, 0.024, 0.585, 0.952, 0.370, edge=PANEL_EDGE, face=PANEL, lw=0.75, radius=0.026)
    panel_title(ax, 0.045, 0.923, "A. Multiplicity within a fixed admissible set")

    add_box(ax, 0.055, 0.690, 0.200, 0.120, edge=BLUE, face=BLUE_FILL, lw=1.15, radius=0.022)
    ax.text(0.155, 0.750, "Fixed quote panel", ha="center", va="center", fontsize=10.8, color=BLUE, fontweight="bold")

    add_arrow(ax, 0.267, 0.750, 0.337, 0.750, lw=1.25, scale=13)

    add_box(ax, 0.350, 0.655, 0.300, 0.185, edge=GREEN, face=GREEN_FILL, lw=1.15, radius=0.024)
    ax.text(0.500, 0.795, r"Common admissible set $\mathcal{A}$", ha="center", va="center", fontsize=11.0, color=GREEN, fontweight="bold")
    ax.text(0.500, 0.718, "same price bounds and", ha="center", va="center", fontsize=9.6)
    ax.text(0.500, 0.680, "no-arbitrage constraints", ha="center", va="center", fontsize=9.6)

    add_arrow(ax, 0.662, 0.750, 0.722, 0.750, lw=1.25, scale=13)

    add_box(ax, 0.735, 0.655, 0.210, 0.185, edge=BLUE, face=BLUE_FILL, lw=1.15, radius=0.024)
    ax.text(0.840, 0.795, "Admissible repairs", ha="center", va="center", fontsize=11.0, color=BLUE, fontweight="bold")
    ax.text(0.840, 0.735, r"$c,\,\tilde c\in\mathcal{A}$", ha="center", va="center", fontsize=12.0, color=INK)
    ax.text(0.840, 0.680, "selected by different rules", ha="center", va="center", fontsize=9.2, color=INK)


def draw_panel_b(ax):
    add_box(ax, 0.024, 0.045, 0.952, 0.485, edge=PANEL_EDGE, face=PANEL, lw=0.75, radius=0.026)
    panel_title(ax, 0.045, 0.495, "B. Local amplification and downstream effects")

    add_box(ax, 0.050, 0.225, 0.175, 0.155, edge=BLUE, face=BLUE_FILL, lw=1.15, radius=0.020)
    ax.text(0.1375, 0.337, "Two admissible repairs", ha="center", va="center", fontsize=9.9, color=BLUE, fontweight="bold")
    ax.text(0.1375, 0.273, r"$c,\,\tilde c\in\mathcal{A}$", ha="center", va="center", fontsize=11.5)
    add_arrow(ax, 0.235, 0.302, 0.275, 0.302)

    add_box(ax, 0.285, 0.220, 0.220, 0.165, edge=BLUE, face=BLUE_FILL, lw=1.15, radius=0.020)
    ax.text(0.395, 0.340, "price-space separation", ha="center", va="center", fontsize=10.0, color=BLUE, fontweight="bold")
    ax.text(0.395, 0.270, r"$\|c-\tilde c\|_\infty\leq\varepsilon$", ha="center", va="center", fontsize=11.0)

    add_arrow(ax, 0.515, 0.330, 0.570, 0.375)
    add_arrow(ax, 0.515, 0.275, 0.570, 0.180)

    add_box(ax, 0.580, 0.305, 0.365, 0.135, edge=ORANGE, face=ORANGE_FILL, lw=1.15, radius=0.020)
    ax.text(0.7625, 0.400, "curvature gap", ha="center", va="center", fontsize=10.4, color=ORANGE, fontweight="bold")
    ax.text(
        0.7625,
        0.348,
        r"$|(D_2c)_i-(D_2\tilde c)_i|\leq 4\varepsilon/h^2$",
        ha="center",
        va="center",
        fontsize=10.8,
    )

    add_box(ax, 0.580, 0.105, 0.365, 0.145, edge=GREEN, face=GREEN_FILL, lw=1.15, radius=0.020)
    ax.text(0.7625, 0.210, "downstream diagnostics", ha="center", va="center", fontsize=10.4, color=GREEN, fontweight="bold")
    ax.text(0.7625, 0.160, "Black-marked Greeks and hedge quantities", ha="center", va="center", fontsize=9.1)
    ax.text(0.7625, 0.125, "VaR-style risk estimates", ha="center", va="center", fontsize=9.1)

def main() -> None:
    register_paper_fonts()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": PAPER_SERIF,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
        }
    )
    fig, ax = plt.subplots(figsize=(10.2, 4.60))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_panel_a(ax)
    draw_panel_b(ax)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_PNG, dpi=260, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {OUT_PDF.resolve()}")
    print(f"Wrote {OUT_PNG.resolve()}")


if __name__ == "__main__":
    main()
