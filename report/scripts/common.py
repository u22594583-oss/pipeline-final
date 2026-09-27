"""Shared paths, style and read-only loaders for the thesis figure scripts.

Canonical inputs are read from the authoritative repository
(the combined repository root) without modification; figures are
written to the shared figures/ folder and derived caches to scripts/cache/.
Nothing is refitted, retuned or recalibrated: the frozen models are only
applied, exactly as in the repository's own Phase II evaluation.
"""

import sys

sys.dont_write_bytecode = True  # never emit bytecode into the read-only source

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
FIGURES = REPO / "figures"
CACHE = ROOT / "scripts" / "cache"
# Outputs are limited to shared figures and the report-local cache.
assert FIGURES == REPO / "figures"
assert CACHE == ROOT / "scripts" / "cache"

TEXT_WIDTH_IN = 16.0 / 2.54  # A4 with 2.5 cm side margins (legacy template)

KEYS = ("H_0p05", "H_0p10", "H_0p20", "S_0p05", "R_0p05", "M_0p05")
LABEL = {
    "H_0p05": "H(0.05)", "H_0p10": "H(0.10)", "H_0p20": "H(0.20)",
    "S_0p05": "S(0.05)", "R_0p05": "R(0.05)", "M_0p05": "M(0.05)",
}
DELTAS = np.round(np.arange(1, 16) * 0.2, 1)
HORIZON = 2969  # reportable monitoring opportunities per replication
WINDOW = 32

# Scheme identity copied from the repository's thesis_plots.py (Okabe-Ito palette;
# solid = hybrid, dashed = benchmark, dotted = ablation), so the new figures match
# the canonical ones. Sizes are reduced for figures drawn at final print size.
SCHEME_STYLE = {
    "H_0p05": dict(color="#0072B2", linestyle="-", marker="o", linewidth=1.6,
                   markersize=3.8, zorder=4),
    "H_0p10": dict(color="#56B4E9", linestyle="-", marker="^", linewidth=1.0,
                   markersize=3.6, markerfacecolor="white"),
    "H_0p20": dict(color="#009E73", linestyle="-", marker="v", linewidth=1.0,
                   markersize=3.6, markerfacecolor="white"),
    "S_0p05": dict(color="#E69F00", linestyle=":", marker="D", linewidth=1.2,
                   markersize=3.0, markerfacecolor="white"),
    "R_0p05": dict(color="#CC79A7", linestyle=":", marker="X", linewidth=1.2,
                   markersize=3.8, markerfacecolor="white"),
    "M_0p05": dict(color="#D55E00", linestyle="--", marker="s", linewidth=1.6,
                   markersize=3.4, zorder=4),
}
NEUTRAL = "#333333"


def apply_style():
    """Thesis rcParams: serif text sized for figures drawn at final width."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": "0.88", "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 2.4,
        "pdf.fonttype": 42,
    })


def save(fig, name):
    """Write figures/<name>.pdf (vector) and a 300-dpi figures/<name>.png."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, dpi, meta in ((".pdf", None, {"CreationDate": None}), (".png", 300, None)):
        fig.savefig(FIGURES / f"{name}{suffix}", dpi=dpi, bbox_inches="tight",
                    pad_inches=0.02, metadata=meta)  # no timestamp: reruns are byte-identical
    plt.close(fig)
    print(f"wrote figures/{name}.pdf|.png")


def stem(delta):
    """D5 bank / result key for a shift magnitude, e.g. 0.2 -> 'delta_0p2'."""
    return "delta_" + f"{delta:.1f}".replace(".", "p")


def load_json(relative):
    return json.loads((REPO / relative).read_text())


def load_run_lengths():
    """Canonical per-replication run lengths and censoring flags (192 arrays)."""
    with np.load(REPO / "results" / "run_lengths.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def pipeline_module():
    """Import the authoritative pipeline.py read-only (no bytecode is written)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import pipeline  # noqa: E402
    return pipeline


def load_bank(relative, replications=None):
    """Load a canonical observation bank (D3, D4 or a D5 scenario) read-only."""
    with np.load(REPO / relative, allow_pickle=False) as archive:
        bank = archive["x"]
    return bank if replications is None else np.ascontiguousarray(bank[:replications])


def monitoring_vectors(name, relative, replications=None):
    """y_t = [s_t, log SPE_t] from the frozen CAE and OC-SVM, cached per bank.

    This is the repository's own monitoring_bank() applied to a canonical bank;
    the D3 cache is checked against the frozen D3 moments by the caller.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npz"
    if path.exists():
        with np.load(path) as archive:
            return archive["y"]
    P = pipeline_module()
    y = P.monitoring_bank(P.load_frozen(), load_bank(relative, replications))
    np.savez(path, y=y)
    return y
