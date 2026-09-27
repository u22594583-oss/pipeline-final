"""Shared Matplotlib style for the thesis figures produced by both notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent / "figures"

# Neutral ink for methodology figures and for contrasts that are not a single scheme.
NEUTRAL = "#333333"

# One fixed visual identity per monitoring scheme, reused in every figure.
# Colours are the Okabe-Ito colour-blind-safe palette. Line style carries the scheme
# family (solid = hybrid, dashed = conventional, dotted = ablation) and the marker
# identifies the scheme, so the figures stay readable in greyscale. The primary pair
# H(0.05) and M(0.05) is drawn heavier, with filled markers.
SCHEME_STYLE = {
    "H(0.05)": dict(color="#0072B2", linestyle="-", marker="o", linewidth=2.2,
                    markersize=6, zorder=4),
    "H(0.10)": dict(color="#56B4E9", linestyle="-", marker="^", linewidth=1.3,
                    markersize=5.5, markerfacecolor="white"),
    "H(0.20)": dict(color="#009E73", linestyle="-", marker="v", linewidth=1.3,
                    markersize=5.5, markerfacecolor="white"),
    "S(0.05)": dict(color="#E69F00", linestyle=":", marker="D", linewidth=1.5,
                    markersize=4.5, markerfacecolor="white"),
    "R(0.05)": dict(color="#CC79A7", linestyle=":", marker="X", linewidth=1.5,
                    markersize=6, markerfacecolor="white"),
    "M(0.05)": dict(color="#D55E00", linestyle="--", marker="s", linewidth=2.2,
                    markersize=5.5, zorder=4),
}

# Filled areas (bars, boxes) cannot use line style, so the benchmark is also hatched.
HATCH = {"M(0.05)": "///"}


def apply_style():
    """Set the thesis rcParams: white background, serif text, restrained axes."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": "0.88",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,  # embed TrueType so PDF text stays selectable in the thesis
    })


def save_figure(fig, name):
    """Save a final thesis figure as figures/<name>.pdf and a 300-dpi figures/<name>.png."""
    FIGURES.mkdir(exist_ok=True)
    for suffix, dpi in ((".pdf", None), (".png", 300)):
        fig.savefig(FIGURES / f"{name}{suffix}", dpi=dpi, bbox_inches="tight",
                    facecolor="white")
