"""Results figures and LaTeX tables from the canonical Phase II outputs (no model is run).

Sources (authoritative repository, read-only):
  results/d5_performance_summary.json  ARL1, SE per chart and shift; paired H-M contrast
  results/d4_validation_summary.json   D4 in-control run-length summaries
  models/control_limits.json           calibrated limits h* and achieved D3 ARL0
  results/run_lengths.npz              per-replication run lengths (derived summaries below)
  figures/results_d4_arl0_validation.pdf, figures/results_primary_run_length_boxplots.pdf
                                       canonical figures reused unchanged in the appendix

Derived quantities (documented in provenance/final_figure_provenance.md): delays in
post-change observations (RL + 31, the canonical t_signal relation for tau = 1), the share
of replications signalling at the first opportunity, and paired H-S / H-R differences
computed exactly as the canonical paired H-M contrast (mean and ddof=1 SE / sqrt(n)).
"""

import json
import re

import numpy as np

from common import (DELTAS, KEYS, LABEL, NEUTRAL, REPO, ROOT, SCHEME_STYLE, FIGURES,
                    TEXT_WIDTH_IN, WINDOW, apply_style, load_json, load_run_lengths, plt,
                    save, stem)

TABLES = ROOT / "tables"
NUMBERS = ROOT / "provenance" / "derived_results.json"
INPUT = {"H": r"$y_t=(s_t,\ell_t)^{\mathsf T}$", "S": r"$s_t$", "R": r"$\ell_t$", "M": r"raw $x_t$"}



def ooc_summary():
    """ARL1 and SE per (chart, delta) and the paired contrast, from the JSON summary."""
    d5 = load_json("results/d5_performance_summary.json")
    arl, se = {}, {}
    for cell in d5["cells"]:
        key = f"{cell['chart_id']}_{cell['lambda']:.2f}".replace(".", "p")
        arl[key, round(cell["delta"], 1)] = cell["arl1_h"]
        se[key, round(cell["delta"], 1)] = cell["se_empirical"]
        assert cell["censor_count"] == 0  # every D5 run length is an observed signal
    contrast = {round(c["delta"], 1): (c["mean_difference"], c["se_paired"])
                for c in d5["primary_contrast"]}
    # Cross-check the summary against the stored per-replication run lengths.
    rl = load_run_lengths()
    for delta in DELTAS:
        for key in KEYS:
            assert np.isclose(rl[f"d5/{key}/{stem(delta)}/run_lengths"].mean(), arl[key, delta])
        diff = (rl[f"d5/H_0p05/{stem(delta)}/run_lengths"].astype(float)
                - rl[f"d5/M_0p05/{stem(delta)}/run_lengths"])
        assert np.isclose(diff.mean(), contrast[delta][0])
    return arl, se, contrast


def _series(ax, arl, keys, se=None, **overrides):
    for key in keys:
        style = {**SCHEME_STYLE[key], **overrides.get(key, {})}
        y = np.array([arl[key, d] for d in DELTAS])
        ax.plot(DELTAS, y, label=LABEL[key], **style)
        if se is not None:
            e = 2 * np.array([se[key, d] for d in DELTAS])
            ax.errorbar(DELTAS, y, yerr=e, fmt="none", ecolor=style["color"],
                        elinewidth=0.7, capsize=1.6, zorder=style.get("zorder", 2))


def _log_axis(ax):
    ax.set_yscale("log")
    ticks = [1, 3, 10, 30, 100, 300]
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.set_yticks([], minor=True)
    ax.set_ylim(0.8, 420)
    ax.set_ylabel(r"ARL$_1$ (opportunities, log scale)")


def ooc_panels(arl, se, contrast):
    """Body figure: primary comparison, paired contrast, ablation, lambda sensitivity."""
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 4.35), sharex=True)
    (a, b), (c, d) = axes

    _series(a, arl, ("H_0p05", "M_0p05"), se=se)
    a.set_title(r"(a) Primary comparison ($\pm2$ SE)", loc="left")

    mean = np.array([contrast[x][0] for x in DELTAS])
    err = 2 * np.array([contrast[x][1] for x in DELTAS])
    b.axhline(0, color=NEUTRAL, linewidth=0.7)
    b.errorbar(DELTAS, mean, yerr=err, color=NEUTRAL, marker="o", markersize=3.2,
               linewidth=1.0, elinewidth=0.8, capsize=1.8)
    b.set_ylabel(r"$\bar d(\delta)$ (opportunities)")
    b.set_title(r"(b) Paired difference $\mathrm{RL}_H-\mathrm{RL}_M$ ($\pm2$ SE)", loc="left")
    b.text(0.97, 0.93, r"$\bar d>0$: M(0.05) signals first", transform=b.transAxes,
           ha="right", va="top", fontsize=7.2, color=NEUTRAL)
    b.set_ylim(-45, 125)

    _series(c, arl, ("H_0p05", "S_0p05", "R_0p05"))
    c.set_title("(c) Ablation", loc="left")

    _series(d, arl, ("H_0p05", "H_0p10", "H_0p20"))
    _series(d, arl, ("M_0p05",), M_0p05=dict(alpha=0.35, zorder=1))
    d.set_title(r"(d) Smoothing sensitivity (M(0.05) shown faintly)", loc="left")

    for ax in (a, c, d):
        _log_axis(ax)
    a.legend(loc="lower left")
    c.legend(loc="lower left")
    d.legend(loc="upper right", ncol=2, columnspacing=1.0)
    for ax in (c, d):
        ax.set_xlabel(r"Shift magnitude $\delta$")
        ax.set_xticks(np.round(np.arange(0.2, 3.01, 0.4), 1))
    fig.tight_layout(h_pad=0.6, w_pad=1.2)
    save(fig, "results_ooc_arl_panels")


def _paired(a, b):
    """Mean and paired SE of a - b, the estimator used for the canonical H-M contrast."""
    d = a.astype(np.float64) - b.astype(np.float64)
    return d.mean(), d.std(ddof=1) / np.sqrt(d.size)


def _write(name, rows):
    TABLES.mkdir(parents=True, exist_ok=True)
    # typeset negative numbers with a true minus sign rather than a hyphen
    rows = [re.sub(r"(?<![\w$])-(?=\d)", "$-$", row) for row in rows]
    (TABLES / f"{name}.tex").write_text(
        "% generated by scripts/make_results_figures.py -- do not edit\n" + "\n".join(rows) + "\n")
    print(f"wrote tables/{name}.tex")


def tables_and_numbers(arl, se):
    """LaTeX table bodies plus the derived numbers quoted in the text."""
    d4 = {f"{c['chart_id']}_{c['lambda']:.2f}".replace(".", "p"): c
          for c in load_json("results/d4_validation_summary.json")["charts"]}
    limits = {f"{c['chart_id']}_{c['lambda']:.2f}".replace(".", "p"): c
              for c in load_json("models/control_limits.json")}
    rl = load_run_lengths()
    numbers = {}

    rows = []
    for key in KEYS:
        c, lim = d4[key], limits[key]
        assert c["censor_count"] == 0
        rows.append(rf"${LABEL[key]}$ & {INPUT[key[0]]} & {lim['h_star']:.2f} & "
                    rf"{lim['achieved_arl0']:.1f} & {c['arl0_validation_h']:.1f} "
                    rf"({c['se_empirical']:.1f}) & {c['sd_run_length']:.1f} & "
                    rf"{c['run_length_median']:.1f} \\")
        numbers[f"d4_z_{key}"] = (c["arl0_validation_h"] - 370) / c["se_empirical"]
        numbers[f"d3_calibration_censored_{key}"] = lim["censor_count"]
    _write("ic_validation", rows)

    rows = []
    for delta in DELTAS:
        cells = " & ".join(f"{arl[k, delta]:.1f} ({se[k, delta]:.1f})" for k in KEYS)
        rows.append(rf"{delta:.1f} & {cells} \\")
    _write("d5_arl1_full", rows)

    rows = []
    for delta in DELTAS:
        h = rl[f"d5/H_0p05/{stem(delta)}/run_lengths"]
        m = rl[f"d5/M_0p05/{stem(delta)}/run_lengths"]
        diff, dse = _paired(h, m)
        obs_h, obs_m = h.mean() + WINDOW - 1, m.mean() + WINDOW - 1  # t_signal = RL + 31
        p1_h, p1_m = np.mean(h == 1), np.mean(m == 1)
        rows.append(rf"{delta:.1f} & {diff:.1f} ({dse:.1f}) & {np.median(h):.1f} & "
                    rf"{np.median(m):.1f} & {p1_m:.3f} & {obs_h:.1f} & {obs_m:.1f} & "
                    rf"{obs_h / obs_m:.2f} & {h.mean() / m.mean():.1f} \\")
        numbers[f"primary_{delta}"] = dict(
            diff=diff, se=dse, t=diff / dse, mrl_h=float(np.median(h)), mrl_m=float(np.median(m)),
            p_first_h=float(p1_h), p_first_m=float(p1_m), obs_h=obs_h, obs_m=obs_m,
            ratio_obs=obs_h / obs_m, ratio_rl=h.mean() / m.mean(),
            p_h_faster=float(np.mean(h < m)))
    _write("primary_contrast", rows)

    rows = []
    for delta in DELTAS:
        h = rl[f"d5/H_0p05/{stem(delta)}/run_lengths"]
        hs = _paired(h, rl[f"d5/S_0p05/{stem(delta)}/run_lengths"])
        hr = _paired(h, rl[f"d5/R_0p05/{stem(delta)}/run_lengths"])
        rows.append(rf"{delta:.1f} & {hs[0]:.1f} ({hs[1]:.1f}) & {hr[0]:.1f} ({hr[1]:.1f}) \\")
        numbers[f"ablation_{delta}"] = dict(h_minus_s=hs[0], se_s=hs[1], h_minus_r=hr[0], se_r=hr[1])
        for lam in ("H_0p10", "H_0p20"):
            lk = _paired(rl[f"d5/{lam}/{stem(delta)}/run_lengths"], h)
            numbers[f"lambda_{lam}_minus_H005_{delta}"] = dict(diff=lk[0], se=lk[1])
    _write("ablation_contrast", rows)

    for lam in ("H_0p10", "H_0p20", "M_0p05"):
        numbers[f"d4_paired_{lam}_minus_H005"] = _paired(rl[f"d4/{lam}/run_lengths"],
                                                          rl["d4/H_0p05/run_lengths"])
    NUMBERS.write_text(json.dumps(numbers, indent=1, sort_keys=True, default=float) + "\n")
    print(f"wrote {NUMBERS.relative_to(ROOT)}")


def main():
    apply_style()
    arl, se, contrast = ooc_summary()
    ooc_panels(arl, se, contrast)
    tables_and_numbers(arl, se)


if __name__ == "__main__":
    main()
