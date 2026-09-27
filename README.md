# Pipeline thesis — final submission

- **Read the report:** [report/main.pdf](report/main.pdf)
- **Edit the report:** [report/main.tex](report/main.tex)
- **Implementation:** [pipeline.py](pipeline.py)
- **Methodology and results walkthroughs:** [01_pipeline.ipynb](01_pipeline.ipynb), [02_results.ipynb](02_results.ipynb)
- **Figure/table provenance:** [report/provenance/final_figure_provenance.md](report/provenance/final_figure_provenance.md)

## Build the report (Ubuntu/WSL)

From the repository root, run `cd report` followed by `latexmk`.
The existing configuration uses pdfLaTeX and BibTeX, writes temporary files to
`report/build/`, and copies the completed PDF to `report/main.pdf`.
Install a TeX distribution with latexmk and the packages used in main.tex if needed.
Saved figures and tables are included; compilation does not require Python or training.
All notebook and report figures share the root `figures/` directory.

## Verification and reproduction

Create a Python environment and install `requirements.txt` as described below.
Run `python -m pytest -q tests/test_pipeline.py` from the repository root.
Run notebooks from the repository root so their relative input paths resolve.

There are three distinct workflows:

1. **Inspect saved results:** use the tests and notebooks; no model training is needed.
2. **Re-evaluate frozen models:** generate data and call `pipeline.evaluate_phase2`,
   as shown below. This updates saved results, so use a separate copy when preserving the submission.
3. **Retrain and reproduce everything:** in a separate source-only copy with no
   `data/`, `models/` or `results/` directories, run `python reproduce_all.py`.
   On Linux/WSL, `bash reproduce_all.sh` also captures timing and environment evidence
   and requires that copy's `.venv`. Its clean-start guard deliberately refuses to
   overwrite the submitted artifacts. Do not delete the submitted artifacts to run it.

To regenerate report figures without retraining, first generate the synthetic banks
using `pipeline.generate_datasets("data/generated")`, then run
`python report/scripts/make_all_figures.py` from the repository root. This reads the
frozen models, results and original production log; it updates shared figures,
report tables and provenance JSON files. It is not needed for report compilation.
The two reused notebook figures are already in `figures/`; regenerate them through
`02_results.ipynb` if needed. Generated data and inference caches are ignored.

`reproduction-report-G8ud52MN/` preserves original runtime evidence, including
historical source paths and hashes. It is not evidence of a new training run here.
The original experiment commit remains documented in the report.

---

# Hybrid CAE–OC-SVM–MEWMA Process Monitoring

A hybrid multivariate statistical process monitoring framework that combines a
1D convolutional autoencoder, a One-Class Support Vector Machine, and a
multivariate exponentially weighted moving average (MEWMA) control chart into a
single monitoring scheme, evaluated on a simulated multivariate process against
one-component ablations and a classical MEWMA benchmark.

## Overview

A classical MEWMA monitors the raw process variables using the estimated
in-control mean and covariance structure. Machine-learning anomaly scores can
capture non-linear structure, but do not by themselves provide an SPC
control-limit calibration. This project evaluates whether combining the two
helps: a learned representation supplies the monitoring variables, and a
conventional MEWMA chart supplies the calibrated decision rule. All six charts
are calibrated on D3 to a common target in-control average run length of 370,
and the resulting hybrid scheme is compared against its own individual
components and against a classical MEWMA applied to the raw process data.

## Methodology

```text
synthetic multivariate process
  -> sliding windows
  -> convolutional autoencoder
  -> latent representation + squared prediction error (SPE)
  -> One-Class SVM score
  -> y_t = [s_t, log(SPE_t)]
  -> MEWMA
  -> T^2
  -> signal / run length
```

Overlapping windows of the process are encoded by the autoencoder. Each window
yields a One-Class SVM anomaly score `s_t` from the latent code and a
reconstruction error `log(SPE_t)`. These two quantities form the bivariate
monitoring vector `y_t`, which is fed to a MEWMA chart. A signal occurs the
first time the MEWMA `T^2` statistic exceeds the calibrated control limit; the
index of that signal is the run length. Full derivations and conventions are
documented separately in `methodology.md`.

## Experimental data

Five independent synthetic datasets separate fitting from evaluation:

| Dataset | Role |
| --- | --- |
| D1 | Autoencoder representation learning |
| D2 | One-Class SVM fitting |
| D3 | In-control moment estimation and control-limit calibration |
| D4 | Independent in-control validation |
| D5 | Out-of-control evaluation across 15 sustained shift magnitudes |

No dataset is used for more than one role. D4 and D5 are seen only at
evaluation time, by frozen artifacts.

## Monitoring schemes

Six pre-registered charts, all calibrated on D3 to a target in-control average
run length of 370:

| Chart | Description |
| --- | --- |
| H(.05) | Primary hybrid scheme, `y_t = [s_t, log(SPE_t)]` |
| H(.10), H(.20) | Hybrid scheme at larger smoothing constants (sensitivity) |
| S(.05) | One-Class SVM score only (ablation) |
| R(.05) | Reconstruction error only (ablation) |
| M(.05) | Classical MEWMA on the raw process variables (benchmark) |

## Repository structure

```text
pipeline.py            complete reference implementation
01_pipeline.ipynb      methodology walkthrough
02_results.ipynb       results analysis
thesis_plots.py        shared figure style and save_figure helper
figures/               thesis figures (PDF + 300-dpi PNG) written by the notebooks
models/                frozen fitted and calibrated artifacts
results/               committed Phase II results
tests/test_pipeline.py regression and smoke suite
methodology.md         detailed scientific methodology
requirements.txt       pinned dependencies
```

## Installation

```bash
git clone <repository-url>
cd pipeline-final

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Quick start

```bash
pytest -q tests/test_pipeline.py
jupyter execute 01_pipeline.ipynb
jupyter execute 02_results.ipynb
```

This uses the committed artifacts and results; no training is required.

## Notebooks

- `01_pipeline.ipynb` — a walkthrough of the methodology using the frozen
  pipeline, from data generation through to the monitoring statistic.
- `02_results.ipynb` — analysis of the committed Phase II results, including
  the in-control validation and the out-of-control comparisons.

## Reproducing the experiment

**Fast inspection.** Use the committed model artifacts and result files, run
the test suite and both notebooks. Nothing is refitted.

**Frozen-model re-evaluation.** Regenerate the synthetic datasets and rerun the Phase II
evaluation against the committed fitted artifacts:

```python
import pipeline

pipeline.generate_datasets("data/generated")
pipeline.evaluate_phase2("data/generated")
```

This reproduces `results/run_lengths.npz`,
`results/d4_validation_summary.json` and `results/d5_performance_summary.json`.

Data generation is seeded and the numerical routes are explicitly controlled,
but no claim of universal cross-platform bitwise identity is made. Deterministic
reproduction is defined relative to the documented software environment and
numerical conventions.

## Frozen artifacts

| File | Contents |
| --- | --- |
| `models/preprocessing.npz` | Per-variable centering and scaling statistics |
| `models/autoencoder.pt` | Trained convolutional autoencoder weights |
| `models/ocsvm.npz` | Fitted One-Class SVM state |
| `models/mewma.npz` | In-control means and covariances |
| `models/control_limits.json` | Calibrated control limits for the six charts |

These are the fitted and calibrated artifacts used for the reported experiment.
Evaluation loads them read-only.

## Main results

D4 provides independent in-control validation on data not used for fitting or
calibration. Achieved in-control average run lengths against the target of 370,
over 500 replications:

| Chart | ARL0 on D4 |
| --- | --- |
| H(.05) | 336.534 |
| H(.10) | 322.332 |
| H(.20) | 315.394 |
| S(.05) | 351.412 |
| R(.05) | 363.444 |
| M(.05) | 403.528 |

Across the fifteen D5 shift magnitudes:

- H(.05) has a lower out-of-control ARL than both S(.05) and R(.05) at every
  tested shift.
- The classical benchmark M(.05) has a lower out-of-control ARL than H(.05) for
  every shift of magnitude 0.4 and above.
- At the smallest tested shift (0.2), the paired H−M run-length difference is
  small relative to its paired standard error.

The hybrid scheme therefore improves on each of its individual machine-learning
components, but does not outperform the classical MEWMA benchmark under this
particular simulated process.

## Testing

`tests/test_pipeline.py` is the permanent regression and smoke suite. It guards
the frozen design constants, the committed artifacts, the MEWMA conventions,
and the committed result files.

## Reproducibility notes

- All stochastic components use explicit seeds and generators.
- Fitted and calibrated artifacts are frozen and committed; evaluation never
  refits them.
- Numerical routes, including native thread-pool usage, are explicitly
  controlled.
- Floating-point and software-platform differences may affect bitwise
  equivalence across environments.
- The artifacts and results reported in the thesis are the ones committed here.

## Thesis context

University of Pretoria, BCom (Hons) Statistics and Data Science — Honours
thesis research project.

## Citation

Citation details will be added once the thesis is finalised. A `CITATION.cff`
file can be added separately.

## License

No license has been selected yet.
