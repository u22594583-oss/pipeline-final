# Final figure and table provenance

Authoritative source: `pipeline-thesis-independent` (original source; now at the combined repository root) (branch `simplify-pipeline-code`,
commit `e62bdeed620fca38350fe8e0021ee70ea7a0fad9`, clean working tree), read-only.
Regenerate from the combined repository root, after generating the required datasets:

    python report/scripts/make_all_figures.py

Nothing is refitted, retuned or recalibrated. "Frozen inference" means the repository's own
`monitoring_bank()` applied with the committed `models/` to a committed data bank; the D3 vectors are
asserted to reproduce `models/mewma.npz` (`mu_y`, `sigma_y`) bitwise before any figure is drawn.
Figure PDFs are byte-identical across reruns (no timestamps).

## Body

| Figure / table | Question answered | Source artifacts | Derived calculation | Script | Status |
|---|---|---|---|---|---|
| Fig. 1 `methodology_pipeline_diagram` | What is the pipeline and which dataset fits which stage? | `methodology.md` §§1–13, `pipeline.py` (verified facts) | none (schematic) | `make_diagnostic_figures.py` | explanatory |
| Fig. 2 `diagnostic_ic_monitoring_vector` | How do the learned components behave in control (Objective 3)? Why empirical calibration? | `data/generated/D3.npz`; `models/{preprocessing.npz,autoencoder.pt,ocsvm.npz,mewma.npz}` | frozen inference of y_t on all 300x2969 D3 windows; hexbin + marginals; pooled ACF about the frozen D3 mean (raw x_t: aligned rows t=32..3000) | `make_diagnostic_figures.py` | canonical-derived (descriptive) |
| Table 1 `tables/ic_validation.tex` | How well did each chart realise IC behaviour on D4 vs its D3 calibration? | `models/control_limits.json`, `results/d4_validation_summary.json` | rounding only | `make_results_figures.py` | canonical |
| Fig. 3 `results_ooc_arl_panels` | H vs M across the full grid; paired contrast; ablation; lambda sensitivity | `results/d5_performance_summary.json` (cells, primary_contrast); cross-checked against `results/run_lengths.npz` | (a),(c),(d) ARL1 with +/-2 SE in (a); (b) canonical paired mean difference +/-2 paired SE | `make_results_figures.py` | canonical (replotted; replaces the three linear-scale repository figures used in the first build) |

## Appendix

| Figure / table | Question answered | Source artifacts | Derived calculation | Script | Status |
|---|---|---|---|---|---|
| Table 2 (W2, typed in `appendix.tex`) | Fixed mixing matrix | `pipeline.py` `W2` | transpose; checked `== pipeline.W2.T` exactly | manual, verified | canonical |
| Table 3 (CAE layers, typed) | Architecture | `pipeline.py` `ConvAutoencoder1d` | per-layer parameter counts (sum 3180) checked by instantiating the class | manual, verified | canonical |
| Fig. 4 `diagnostic_reconstruction_example` | What does the CAE reconstruct? | `data/generated/D3.npz` (trajectory 1, window 1001); frozen models | one frozen forward pass; SPE = 96.65 | `make_diagnostic_figures.py` | canonical-derived (descriptive) |
| Fig. 5 `diagnostic_component_response` | How far do s_t and l_t move under a shift? | `data/generated/D5/delta_{0p4,1p0,3p0}.npz` (first 100 trajectories each), D3 | frozen inference; standardised by frozen D3 moments; histograms; mean shifts | `make_diagnostic_figures.py` | **post-hoc descriptive diagnostic** (labelled as such in caption and text) |
| Fig. 6 `results_d4_arl0_validation` | D4 IC ARL +/-1 SE vs target | repository figure (produced by `02_results.ipynb` from `results/`) | none (shared unchanged) | `make_results_figures.py` (shared asset) | canonical |
| Table 4 `tables/d5_arl1_full.tex` | Full ARL1 (SE) grid | `results/d5_performance_summary.json` | rounding | `make_results_figures.py` | canonical |
| Table 5 `tables/primary_contrast.tex` | Primary contrast in detail | `results/run_lengths.npz` | paired mean diff/SE (identical to canonical `primary_contrast`); medians; P(RL_M=1); delay in post-change observations = ARL1+31 (canonical `t_signal = RL + 31`); ratios | `make_results_figures.py` | canonical-derived |
| Table 6 `tables/ablation_contrast.tex` | Does H beat S and R pairwise? | `results/run_lengths.npz` | paired H-S, H-R mean and SE (same estimator as canonical H-M contrast) | `make_results_figures.py` | canonical-derived |
| Fig. 7 `results_primary_run_length_boxplots` | Run-length distribution shape, H vs M | repository figure (from `results/run_lengths.npz`) | none (shared unchanged) | `make_results_figures.py` (shared asset) | canonical |

## Candidates generated or inspected but not used

| Candidate | Decision |
|---|---|
| Repository `results_primary_hybrid_vs_classical_arl`, `results_hybrid_ablation_arl`, `results_lambda_sensitivity_arl` (first build) | superseded by Fig. 3: one float instead of three, log scale shows the large-shift gap (M ~ 1 vs H ~ 12.5) that the linear scale hid |
| Repository `results_arl_all_schemes`, `_sdrl_`, `_mrl_`, `_primary_run_length_metrics`, `_small_shift_zoom`, `_primary_paired_difference`, `_primary_run_length_ecdf` | redundant with Fig. 3, Tables 4-5 and Fig. 7 |
| Repository `methodology_d2_component_distributions`, `methodology_d2_monitoring_vector` | D2 is in-sample for the OC-SVM; replaced by the D3 (calibration-bank) view in Fig. 2 |
| Repository `methodology_d1_observed_channels` | decorative for the argument; raw dependence is shown by the ACF in Fig. 2(b) |
| Repository `methodology_finite_start_factor` | secondary; c_j is defined in Eq. (4) |
| Correlation / covariance heatmaps | not drawn: the hybrid correlation is a single number (r = -0.147 from the frozen D3 covariance), which the text reports; a 4x4 raw heatmap adds nothing to the argument |
| CAE training/validation loss curve | not available as a stored canonical artifact; only the production log's best epoch (33) and validation MSE (0.6622) exist, and retraining is not permitted |

## Derived numbers quoted in the text

`report/provenance/derived_results.json` (from `make_results_figures.py`) and
`report/provenance/derived_diagnostics.json` (from `make_diagnostic_figures.py`) hold every derived number
quoted in the report (D4 z-scores and paired D4 differences, P(RL=1), post-change-observation delays,
paired ablation and lambda contrasts, component correlation/skewness/ACF, OC-SVM outside-boundary
shares, CAE per-element reconstruction error, component mean shifts).
