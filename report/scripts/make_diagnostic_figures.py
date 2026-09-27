"""Methodology and component-diagnostic figures.

Sources (authoritative repository, read-only):
  data/generated/D2.npz, D3.npz, D5/delta_*.npz   canonical observation banks
  models/{preprocessing.npz, autoencoder.pt, ocsvm.npz, mewma.npz}   frozen artifacts
  reproduction-report-G8ud52MN/production.log      CAE best epoch / validation loss

The monitoring vectors are produced by the repository's own monitoring_bank()
with the frozen models; the D3 vectors are verified to reproduce the frozen D3
moments exactly before any figure is drawn. Figures and numbers built on D5 banks
are post-hoc descriptive diagnostics, not pre-specified outputs of the experiment.
"""

import json

import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from common import (CACHE, NEUTRAL, REPO, ROOT, SCHEME_STYLE, TEXT_WIDTH_IN, WINDOW,
                    apply_style, load_bank, monitoring_vectors, pipeline_module,
                    plt, save, stem)

S_COLOR = SCHEME_STYLE["S_0p05"]["color"]  # OC-SVM score, as in the S ablation
R_COLOR = SCHEME_STYLE["R_0p05"]["color"]  # log SPE, as in the R ablation
MAX_LAG = 100
NUMBERS = ROOT / "provenance" / "derived_diagnostics.json"


def d3_vectors():
    """Frozen-model D3 monitoring vectors, checked against models/mewma.npz."""
    P = pipeline_module()
    y = monitoring_vectors("D3_y", "data/generated/D3.npz")
    moments = P.pooled_moments(np.ascontiguousarray(y))
    hybrid, _ = P.load_moments()
    assert np.array_equal(moments.mean, hybrid.mean)
    assert np.array_equal(moments.covariance, hybrid.covariance)
    return y, hybrid


def acf(series, mean):
    """Pooled ACF about the chart's reference mean, and per-replication ACFs."""
    u = series - mean
    pooled = np.array([np.mean(u[:, : u.shape[1] - k] * u[:, k:]) for k in range(MAX_LAG + 1)])
    v = series - series.mean(axis=1, keepdims=True)
    per = np.stack([np.sum(v[:, : v.shape[1] - k] * v[:, k:], axis=1) for k in range(MAX_LAG + 1)], axis=1)
    return pooled / pooled[0], per / per[:, :1]


def ic_monitoring_vector(y, hybrid, raw, numbers):
    """Body figure: in-control joint distribution and serial dependence on D3."""
    s, ell = y[..., 0].ravel(), y[..., 1].ravel()
    corr = hybrid.covariance[0, 1] / np.sqrt(hybrid.covariance[0, 0] * hybrid.covariance[1, 1])
    numbers["d3_corr_s_ell"] = float(corr)
    numbers["d3_frac_s_positive"] = float(np.mean(s > 0))
    for name, x in (("s", s), ("ell", ell)):
        z = (x - x.mean()) / x.std()
        numbers[f"d3_skew_{name}"] = float(np.mean(z ** 3))
        numbers[f"d3_exkurt_{name}"] = float(np.mean(z ** 4) - 3)
    spe = np.exp(ell)
    numbers["d3_mean_spe_per_element"] = float(spe.mean() / (4 * WINDOW))
    z = (spe - spe.mean()) / spe.std()
    numbers["d3_skew_spe"] = float(np.mean(z ** 3))

    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 2.45))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[3.0, 0.55, 0.62, 4.1],
                  height_ratios=[0.62, 3.0], wspace=0.06, hspace=0.06)
    joint = fig.add_subplot(gs[1, 0])
    top = fig.add_subplot(gs[0, 0], sharex=joint)
    side = fig.add_subplot(gs[1, 1], sharey=joint)
    lag_ax = fig.add_subplot(gs[:, 3])

    joint.hexbin(s, ell, gridsize=55, bins="log", cmap="Greys", mincnt=1, linewidths=0.1)
    joint.axvline(0, color=S_COLOR, linestyle="--", linewidth=0.9)
    joint.text(0.25, ell.max() - 0.02, "OC-SVM\nboundary", color=S_COLOR, fontsize=6.8,
               va="top", ha="left")
    joint.plot(*hybrid.mean, marker="+", color="white", markersize=7, mew=1.6)
    joint.plot(*hybrid.mean, marker="+", color=NEUTRAL, markersize=6, mew=0.9)
    joint.text(0.03, 0.04, rf"$r={corr:.2f}$", transform=joint.transAxes, fontsize=7.5)
    joint.set_xlabel(r"OC-SVM score $s_t$")
    joint.set_ylabel(r"$\ell_t=\log\mathrm{SPE}_t$")
    joint.grid(False)
    top.hist(s, bins=90, color=S_COLOR, alpha=0.75)
    side.hist(ell, bins=90, color=R_COLOR, alpha=0.75, orientation="horizontal")
    for ax in (top, side):
        ax.set_axis_off()
    top.set_title("(a) Joint and marginal distributions (D3)", loc="left")

    raw_acfs = [acf(raw[..., c], raw[..., c].mean())[0] for c in range(raw.shape[2])]
    lags = np.arange(MAX_LAG + 1)
    for c, r in enumerate(raw_acfs):
        lag_ax.plot(lags, r, color="0.6", linewidth=0.8,
                    label=r"raw $x_{t,c}$, $c=1,\dots,4$" if c == 0 else None)
    for name, k, color, label in (("s", 0, S_COLOR, r"$s_t$"), ("ell", 1, R_COLOR, r"$\ell_t$")):
        pooled, per = acf(y[..., k], hybrid.mean[k])
        lag_ax.plot(lags, pooled, color=color, linewidth=1.5, label=label,
                    linestyle="-" if k == 0 else "--")
        numbers[f"d3_acf1_{name}"] = float(pooled[1])
        numbers[f"d3_acf32_{name}"] = float(pooled[32])
        # spread of single-trajectory estimates beyond the overlap (reported, not drawn)
        numbers[f"d3_acf_single_rep_p95_abs_lag40to100_{name}"] = float(
            np.percentile(np.abs(per[:, 40:]), 95))
    numbers["d3_acf1_raw_range"] = [float(min(r[1] for r in raw_acfs)), float(max(r[1] for r in raw_acfs))]
    numbers["d3_acf5_raw_max"] = float(max(r[5] for r in raw_acfs))
    lag_ax.axvline(WINDOW, color=NEUTRAL, linestyle=":", linewidth=0.9)
    lag_ax.text(WINDOW + 1.5, 0.52, "from lag 32, the two\nwindows share no\nobservations",
                fontsize=6.8, va="top", color=NEUTRAL)
    lag_ax.axhline(0, color=NEUTRAL, linewidth=0.6)
    lag_ax.set_xlim(0, MAX_LAG)
    lag_ax.set_ylim(-0.12, 1.02)
    lag_ax.set_xlabel("Lag (monitoring opportunities)")
    lag_ax.set_ylabel("Autocorrelation")
    lag_ax.set_title("(b) In-control autocorrelation (D3)", loc="left")
    lag_ax.legend(loc="upper right")
    save(fig, "diagnostic_ic_monitoring_vector")


def _box(ax, x, y, w, h, text, face="white", edge=NEUTRAL, size=7.2, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                facecolor=face, edgecolor=edge, linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight=weight, linespacing=1.25)


def _arrow(ax, start, end, color=NEUTRAL):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, shrinkA=0, shrinkB=0,
                                mutation_scale=7))


def pipeline_diagram():
    """Body figure: monitoring pipeline, benchmark and the five data roles."""
    fit = {"D1": "#D6E8F5", "D2": "#FBE7C6", "D3": "#D5EFE3", "D4": "#EDEDED", "D5": "#F7DCCB"}
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.set_axis_off()

    # Hybrid path (top row); box shading = the dataset that fits or calibrates the stage.
    ax.text(0.5, 42.3, r"Hybrid chart $H$ (ablations: $S$ uses $s_t$ only, $R$ uses $\ell_t$ only)",
            fontsize=7.2, va="center", style="italic")
    _box(ax, 0.5, 30, 10, 8, "Observed\n" r"$x_t\in\mathbb{R}^4$")
    _box(ax, 13, 30, 13, 8, "Standardise,\n" r"window $L=32$", face=fit["D1"])
    _box(ax, 28.5, 30, 12, 8, "CAE encoder\n" r"$h_t\in\mathbb{R}^8$", face=fit["D1"])
    _box(ax, 43, 34.6, 14, 3.8, r"OC-SVM $\to s_t$", face=fit["D2"])
    _box(ax, 43, 29.6, 14, 3.8, r"decoder $\to \ell_t$", face=fit["D1"])
    _box(ax, 59.5, 30, 11.5, 8, r"$y_t=(s_t,\ell_t)^{\mathsf{T}}$")
    _box(ax, 73.5, 30, 15.5, 8, r"MEWMA $T^2_j$;" "\n" r"signal: $T^2_j>h^*$", face=fit["D3"])
    for start, end in (((10.8, 34), (12.7, 34)), ((26.3, 34), (28.2, 34)),
                       ((40.8, 35), (42.7, 36.5)), ((40.8, 33), (42.7, 31.5)),
                       ((57.3, 36.5), (59.2, 35)), ((57.3, 31.5), (59.2, 33)),
                       ((71.3, 34), (73.2, 34)), ((71.3, 23), (73.2, 23))):
        _arrow(ax, start, end)

    # Benchmark path (middle row).
    _box(ax, 13, 21, 58, 4, r"Benchmark $M$: MEWMA on raw $x_t$ (silent run-in for $t\leq31$)")
    _box(ax, 73.5, 21, 15.5, 4, r"signal: $T^2_j>h^*$", face=fit["D3"])
    ax.plot([5.5, 5.5, 11.5], [29.7, 23, 23], color=NEUTRAL, linewidth=0.8)
    _arrow(ax, (11.4, 23), (12.7, 23))

    # Data roles (bottom row).
    roles = (("D1", r"$2{,}000\times4$", "standardisation,\nCAE training"),
             ("D2", r"$2{,}000\times4$", "OC-SVM fit\n" r"($\nu=0.03$, $\gamma$ rule)"),
             ("D3", r"$300\times3{,}000\times4$", r"moments $\hat\mu,\hat\Sigma$;" "\n"
              r"$h^*$ for ARL$_0=370$"),
             ("D4", r"$500\times3{,}000\times4$", "independent\nIC validation"),
             ("D5", r"$500\times3{,}000\times4$", r"per shift $\delta$:" "\nOOC evaluation"))
    for i, (name, shape, role) in enumerate(roles):
        _box(ax, 0.5 + i * 20, 1, 18.6, 12, f"{name}: {shape}\n{role}", face=fit[name], size=6.9)
    for x0, x1, label in ((0.5, 59.4, "Phase I: fit and calibrate, then freeze"),
                          (60.5, 99.4, "Phase II: frozen evaluation")):
        ax.plot([x0, x1], [16, 16], color=NEUTRAL, linewidth=0.7)
        ax.text((x0 + x1) / 2, 17.4, label, ha="center", fontsize=7.2, style="italic")
    save(fig, "methodology_pipeline_diagram")


def component_response(hybrid, numbers):
    """Appendix, post-hoc descriptive: IC versus OOC distributions of each component."""
    deltas = (0.4, 1.0, 3.0)
    reps = 100  # first 100 canonical D5 replications per shift, deterministic subset
    y_ic = monitoring_vectors("D3_y", "data/generated/D3.npz")
    y_oc = {d: monitoring_vectors(f"D5_{stem(d)}_first{reps}_y",
                                  f"data/generated/D5/{stem(d)}.npz", reps)
            for d in deltas}
    sd = np.sqrt(np.diag(hybrid.covariance))
    shades = ("#9ECAE1", "#4292C6", "#08306B")
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 1.95))
    for k, (ax, label) in enumerate(zip(axes, (r"standardised $s_t$", r"standardised $\ell_t$"))):
        z_ic = (y_ic[..., k].ravel() - hybrid.mean[k]) / sd[k]
        bins = np.linspace(-4, 7, 111)
        ax.hist(z_ic, bins=bins, density=True, color="0.8", label="in control (D3)")
        for d, shade in zip(deltas, shades):
            z = (y_oc[d][..., k].ravel() - hybrid.mean[k]) / sd[k]
            ax.hist(z, bins=bins, density=True, histtype="step", color=shade, linewidth=1.1,
                    label=rf"$\delta={d}$ (D5)")
            numbers[f"d5_std_mean_shift_{'s' if k == 0 else 'ell'}_{d}"] = float(z.mean())
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.set_yticks([])
        ax.grid(False)
    axes[0].legend(loc="upper right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "diagnostic_component_response")


def reconstruction_example(numbers):
    """Appendix: one standardised D3 window and its frozen-CAE reconstruction."""
    P = pipeline_module()
    frozen = P.load_frozen()
    x = load_bank("data/generated/D3.npz", 1)[0]
    windows = P.window_series(P.apply_preprocessing(x, frozen.mean, frozen.scale))
    window = windows[1000:1001]  # fixed, arbitrary in-control window
    with P.single_threaded():
        _, recon = P.infer_windows(frozen.autoencoder, window)
    spe = float(P.spe(window, recon)[0])
    numbers["example_window_spe"] = spe
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH_IN, 1.45), sharey=True)
    t = np.arange(1, WINDOW + 1)
    for c, ax in enumerate(axes):
        ax.plot(t, window[0, c], color=NEUTRAL, linewidth=0.9, label="standardised window")
        ax.plot(t, recon[0, c], color=SCHEME_STYLE["H_0p05"]["color"], linewidth=1.3,
                label="CAE reconstruction")
        ax.set_title(rf"channel {c + 1}", loc="left")
        ax.set_xlabel("position in window")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.6)
    save(fig, "diagnostic_reconstruction_example")


def d2_boundary_fraction(numbers):
    """In-sample share of D2 windows outside the OC-SVM boundary (nu-property check)."""
    P = pipeline_module()
    y = P.monitoring_series(P.load_frozen(), load_bank("data/generated/D2.npz"))
    numbers["d2_frac_s_positive"] = float(np.mean(y[:, 0] > 1e-9))
    numbers["d2_windows"] = int(y.shape[0])


def production_log(numbers):
    text = (REPO / "reproduction-report-G8ud52MN"
            / "production.log").read_text()
    line = next(line for line in text.splitlines() if "best epoch" in line)
    numbers["cae_production_log"] = line.strip()


def main():
    apply_style()
    numbers = {}
    y, hybrid = d3_vectors()
    numbers["d3_vectors_shape"] = list(y.shape)
    raw = np.ascontiguousarray(load_bank("data/generated/D3.npz")[:, WINDOW - 1:, :])
    ic_monitoring_vector(y, hybrid, raw, numbers)
    pipeline_diagram()
    component_response(hybrid, numbers)
    reconstruction_example(numbers)
    d2_boundary_fraction(numbers)
    production_log(numbers)
    NUMBERS.write_text(json.dumps(numbers, indent=1, sort_keys=True) + "\n")
    print(f"wrote {NUMBERS.relative_to(ROOT)}; cache in {CACHE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
