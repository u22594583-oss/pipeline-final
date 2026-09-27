# Methodology

This repository implements a hybrid convolutional-autoencoder / one-class-SVM / MEWMA
monitoring scheme and evaluates it on a simulated nonlinear multivariate process. Five
datasets assign separate roles to representation learning, one-class fitting, in-control
estimation and calibration, independent in-control validation, and out-of-control
evaluation. This document specifies the experiment and the conventions that determine
every reported run length.

---

## 1. Experimental design

The monitoring pipeline is a fixed chain:

```
raw multivariate process
  -> overlapping windows (length 32, stride 1)
  -> convolutional autoencoder
  -> latent representation h_t  and  reconstruction error SPE_t
  -> One-Class SVM anomaly score s_t
  -> hybrid monitoring vector y_t
  -> MEWMA recursion
  -> T^2 statistic
  -> signal / run length
```

The hybrid monitoring vector is

$$y_t = \begin{bmatrix} s_t \\ \log \mathrm{SPE}_t \end{bmatrix} \in \mathbb{R}^2,$$

where $s_t$ is the negated One-Class SVM decision function, so that larger $s_t$ means more
anomalous; $\mathrm{SPE}_t$ is the autoencoder reconstruction-error statistic of the window
ending at time $t$; and the logarithm is taken after the implemented numerical floor (§7).

Six monitoring schemes are compared, all fixed before evaluation: the hybrid chart at three
smoothing constants, its two one-component ablations, and a classical MEWMA applied
directly to the raw process variables (§11).

---

## 2. Synthetic multivariate process

### Latent dynamics

The latent state $z_t \in \mathbb{R}^{d_z}$, $d_z = 8$, follows a stable mean-adjusted
VAR(1) recursion

$$z_t = \mu_t + \Phi\,(z_{t-1} - \mu_{t-1}) + \eta_t, \qquad \eta_t \sim N(0, Q),$$

with $z_0 = 0$ and $\mu_t \equiv 0$ in control. The transition matrix $\Phi$ is $8 \times 8$
symmetric tridiagonal with $0.55$ on the diagonal, $0.15$ on the first upper and lower
off-diagonals, and zeros elsewhere. Its spectral radius is $\rho(\Phi) \approx
0.8319077862 < 1$, so the in-control latent process is stationary. The innovation
covariance is $Q = 0.20\,I_8$.

Each trajectory uses a burn-in of $B = 500$ latent steps, which are discarded. Burn-in
draws innovations only; no measurement noise is drawn for discarded steps.

### Observation model

The observed vector $x_t \in \mathbb{R}^{p}$, $p = 4$, is a static nonlinear map of the
latent state plus independent measurement noise:

$$x_t = W_2 \tanh(W_1 z_t + b_1) + b_2 + \varepsilon_t,
\qquad \varepsilon_t \sim N(0, \Sigma_\varepsilon),
\qquad \Sigma_\varepsilon = 0.05\, I_4 .$$

The observation-noise covariance is written $\Sigma_\varepsilon$ throughout, to avoid any
collision with the chart label $R(.05)$ of §11.

In the frozen specification $W_1 = I_8$, $b_1 = 0$ and $b_2 = 0$, so the map reduces to
$x_t = W_2 \tanh(z_t) + \varepsilon_t$. $W_2$ is a fixed $4 \times 8$ mixing matrix of full
row rank with approximately unit-norm rows. These are deterministic simulation parameters
stated as literals in `pipeline.py`; they are neither learned nor estimated.

Because $\tanh$ is applied elementwise before mixing, the observed process is a nonlinear
transformation of a linear latent dynamic process. This nonlinear observation mechanism
motivates evaluating a learned representation alongside the raw-data benchmark.

### Mean-shift fault

The fault is a sustained shift of the latent mean in a fixed direction

$$v = \tfrac{1}{\sqrt{2}}\,[\,1,\;1,\;0,\;0,\;0,\;0,\;0,\;0\,]^{\mathsf T},
\qquad
\mu_t = \begin{cases} \delta\,v, & t \ge \tau, \\ 0, & t < \tau, \end{cases}
\qquad \tau = 1 .$$

The index $t$ counts **retained**, post-burn-in observations, so $\tau = 1$ is the first
retained process observation. The burn-in itself is always in control. An out-of-control
trajectory is therefore post-change over its entire retained horizon.

The shift magnitudes form the grid $\delta \in \{0.2, 0.4, \dots, 3.0\}$ (15 values). No
other fault mechanism is used: there is no variance change, sensor fault, drift or
transient.

---

## 3. Dataset separation

The five datasets assign distinct roles to representation learning, one-class fitting,
in-control estimation and calibration, independent in-control validation, and
out-of-control evaluation.

| Dataset | Shape | Condition | Role |
| --- | --- | --- | --- |
| D1 | $(2000, 4)$ | in control | Autoencoder representation learning (with an internal train/validation split) |
| D2 | $(2000, 4)$ | in control | One-Class SVM fitting |
| D3 | $(300, 3000, 4)$ | in control | In-control monitoring-vector moments **and** control-limit calibration |
| D4 | $(500, 3000, 4)$ | in control | Independent in-control validation |
| D5 | 15 banks, each $(500, 3000, 4)$ | out of control | Out-of-control evaluation, one bank per $\delta$ |

D1–D4 are in control; D5 contains the sustained latent mean shifts, one bank per shift
magnitude. D3 and D4 replications are independent realisations of the same in-control
process.

All fitting and calibration is confined to D1, D2 and D3. D4 and D5 are never used to fit
the preprocessing statistics, train the autoencoder, fit the One-Class SVM, estimate the
in-control moments, or calibrate any control limit. They are not generic "test sets": D4
supplies an independent in-control run-length distribution, and D5 supplies out-of-control
run-length distributions at each fixed shift magnitude.

---

## 4. Random-number generation and numerical representation

The master simulation seed is $795$. All randomness is drawn from NumPy `Generator`
instances over the `PCG64DXSM` bit generator, seeded through independent deterministic
`SeedSequence` streams, one namespace per dataset:

| Dataset | D1 | D2 | D3 | D4 | D5 |
| --- | --- | --- | --- | --- | --- |
| Namespace | 1 | 2 | 3 | 4 | 5 |

Within D3, D4 and D5 each replication has its own stream, so a trajectory does not depend
on how many other trajectories were generated, or in what order. The exact entropy
construction and draw order are given in §15.

Generated process arrays are `float64`. Global mutable NumPy random state is never used,
and the latent process is never persisted — only observations leave the generator.

---

## 5. Window construction and preprocessing

### Windows

| Quantity | Value |
| --- | --- |
| window length $L$ | 32 |
| stride | 1 |
| first monitorable raw time index | $t = 32$ |
| tensor orientation | channels-first, $(W, C, L) = (W, 4, 32)$ |

Window $w$ (zero-based) covers raw rows $w, \dots, w+31$ and is labelled by its **last**
row, carrying the one-based raw time index $t = w + 32$. A series of $N$ rows yields
$N - 31$ windows; for the $N = 3000$ monitoring banks this is 2969 windows per replication.

Windows are formed within a replication only. The replication axis is never flattened
before windowing, so no window can contain observations from two replications.

### D1 split

The chronological D1 split is applied to raw rows, **before** windowing:

| Segment | Raw rows | Windows |
| --- | --- | --- |
| training | $[0, 1600)$ | 1569 |
| validation | $[1600, 2000)$ | 369 |

No window crosses the boundary: the 31 straddling windows are never constructed, so D1
yields $1569 + 369 = 1938$ windows rather than the 1969 an unsplit 2000-row series would
give. This keeps validation rows out of the fitted statistics and out of training.

### Preprocessing

Per-variable standardization is fitted on the D1 **training rows only**:

$$\tilde{x}_{t,c} = \frac{x_{t,c} - \hat m_c}{\hat\sigma_c}, \qquad c = 1,\dots,4,$$

with $\hat m_c$ and $\hat\sigma_c$ the mean and standard deviation of variable $c$ over rows
$[0,1600)$ of D1, computed with the population convention `ddof = 0` and stored in
`float64`. A degenerate scale aborts the fit rather than being clipped.

Standardization is applied to raw rows before windowing, which is equivalent to
standardizing each window. The fitted pair $(\hat m, \hat\sigma)$ is frozen after D1 and
applied unchanged to D2, D3, D4 and every D5 bank; it is never refitted.

---

## 6. Convolutional autoencoder

### Architecture

A symmetric 1D convolutional autoencoder maps a standardized window
$\tilde X_t \in \mathbb{R}^{4 \times 32}$ to a latent vector and back.

| Component | Specification |
| --- | --- |
| input | $(4, 32)$, channels-first |
| encoder conv 1 | `Conv1d(4 -> 8)`, kernel 3, stride 2, padding 1, ReLU |
| encoder conv 2 | `Conv1d(8 -> 16)`, kernel 3, stride 2, padding 1, ReLU |
| flatten | $16 \times 8 = 128$ |
| bottleneck | `Linear(128 -> 8)` |
| decoder linear | `Linear(8 -> 128)`, ReLU, unflatten to $(16, 8)$ |
| decoder conv 1 | `ConvTranspose1d(16 -> 8)`, kernel 3, stride 2, padding 1, output padding 1, ReLU |
| decoder conv 2 | `ConvTranspose1d(8 -> 4)`, kernel 3, stride 2, padding 1, output padding 1 |
| latent dimension $q$ | 8 |
| trainable parameters | 3180 |

The two stride-2 convolutions halve the time axis twice ($32 \to 16 \to 8$); the transposed
convolutions require `output_padding = 1` to invert the stride-2 floor division and return
to 32 samples. The latent dimension $q = 8$ is numerically equal to $d_z$, but the learned
bottleneck is a distinct quantity from the generating latent state and is never compared
to it.

### Training

| Setting | Value |
| --- | --- |
| training data | D1 training windows (1569) |
| validation data | D1 validation windows (369) |
| objective | mean-squared reconstruction error |
| optimiser | Adam |
| learning rate | $10^{-3}$ |
| batch size | 64 |
| maximum epochs | 100 |
| early-stopping patience | 10 |
| training seed | 3795 |
| precision | `float32` parameters and activations |

The training seed fixes the parameter initialisation and, through a separately seeded
shuffle generator, the batch order. Model selection is by validation loss: training stops
after 10 epochs without improvement and the best-validation checkpoint is restored, so the
frozen weights are generally not the final epoch's weights.

The trained network is frozen after D1 and is not retrained, fine-tuned or adapted for D2,
D3, D4 or D5.

### Latent representation

For each standardized window ending at raw time $t$, the encoder produces

$$h_t = \mathrm{Linear}\big(\mathrm{flatten}(\mathrm{conv}(\tilde X_t))\big) \in \mathbb{R}^{8}.$$

There is no pooling and no post-hoc aggregation: the flatten-then-linear projection is the
entire reduction. Features leave the network as `float32`.

---

## 7. Reconstruction-error statistic

Let $\hat X_t$ be the reconstruction of $\tilde X_t$, obtained by decoding the same $h_t$
produced for that window, and let $r_t = \tilde X_t - \hat X_t \in \mathbb{R}^{4 \times 32}$
be the reconstruction residual. Then

$$\mathrm{SPE}_t = \sum_{c=1}^{4} \sum_{\ell=1}^{32} r_{t,c,\ell}^{2}.$$

That is: the squared errors are **summed**, not averaged; the reduction runs over **both**
the channel axis and the time axis, i.e. over all $4 \times 32 = 128$ elements of the
window; and the `float32` reconstruction is cast up before subtraction, so the residual and
the summation are carried out in `float64`. $\mathrm{SPE}_t$ is a summed squared error, not
a residual variance and not a mean.

This is deliberately **not** the training objective of §6. The autoencoder is trained with a
mean-squared-error loss; the monitoring statistic is the *sum* of the squared residuals over
the window. The two quantities coexist in the pipeline, differ by the factor 128, and are
never interchanged.

The monitoring component is the floored logarithm

$$\log \mathrm{SPE}_t = \log\big(\max(\mathrm{SPE}_t,\, \epsilon)\big), \qquad
\epsilon = 10^{-12},$$

the floor preventing a vanishing reconstruction error from producing $-\infty$. The
logarithm places the reconstruction error on a scale comparable to the One-Class SVM score,
so the two can be monitored jointly by one covariance-standardized chart.

---

## 8. One-Class SVM

The one-class detector is fitted **only** on the D2 latent representations produced by the
frozen preprocessing and the frozen autoencoder.

| Quantity | Value |
| --- | --- |
| fitting data | D2 latent features |
| rows $n$ | 1969 |
| features $q$ | 8 |
| kernel | RBF |
| $\nu$ | 0.03 |
| $\gamma$ | $0.030522830729177747$ |
| support vectors | 78 |
| $\rho$ | $8.846560927608301$ |

No feature scaling, centering or whitening is applied between the bottleneck and the SVM:
the fitting matrix is exactly the frozen latent features, cast explicitly from `float32` to
`float64` at this boundary. Fitting and scoring are performed in `float64`.

The kernel width follows the median heuristic, evaluated exactly on the D2 fitting matrix:

$$\gamma = \frac{1}{\operatorname{median}_{i<j} \|h_i - h_j\|^2},$$

over all $\binom{1969}{2}$ unordered pairs, with no subsampling and no quantile
approximation. The resolved median squared pairwise distance is $32.762361029773956$, whose
reciprocal is the $\gamma$ above. $\gamma$ is resolved once during fitting and then frozen;
re-evaluating the rule on any other data would constitute a refit.

The monitoring score is the negated decision function,

$$s_t = -\operatorname{decision\_function}(h_t)
      = \rho - \sum_{j=1}^{78} \alpha_j \exp\!\big(-\gamma \|h_t - h_{(j)}\|^2\big),$$

with $h_{(j)}$ the support vectors and $\alpha_j$ the dual coefficients. The negation fixes
the orientation: the library decision function is positive inside the learned boundary,
whereas a monitoring statistic must increase as the process deteriorates, so **larger $s_t$
indicates greater abnormality**.

The frozen scoring state is exactly the support vectors, the dual coefficients, $\rho$ and
$\gamma$ — a complete mathematical description of $s_t$. After fitting on D2 the detector is
frozen and applied unchanged to D3, D4 and D5.

---

## 9. Hybrid monitoring vector and in-control moments

### Definition

$$y_t = \begin{bmatrix} s_t \\ \log \mathrm{SPE}_t \end{bmatrix}, \qquad \dim(y_t) = 2 .$$

The component order is fixed: component 0 is the One-Class SVM anomaly score, component 1
the log reconstruction error. Both come from one inference pass over the same window, so
they cannot be misaligned by an index. The ordering is load-bearing because the frozen
mean, the frozen covariance and every calibrated control limit are expressed in that order
and carry no labels; a transposed pair would silently reinterpret the chart rather than
fail. The two components are complementary: $s_t$ asks whether the encoded window resembles
the in-control latent cloud, $\log \mathrm{SPE}_t$ whether the autoencoder can represent the
window at all.

### Hybrid moments from D3

Each of the 300 D3 replications has 3000 raw observations and therefore $3000 - 31 = 2969$
reportable monitoring vectors, giving $300 \times 2969 = 890{,}700$ pooled monitoring rows.
D3 is pooled to estimate the in-control location and dispersion of $y_t$:

$$\hat\mu_y = \frac{1}{n}\sum_{k=1}^{n} y_k, \qquad
\hat\Sigma_y = \frac{1}{n-1}\sum_{k=1}^{n} (y_k - \hat\mu_y)(y_k - \hat\mu_y)^{\mathsf T},
\qquad n = 890{,}700,$$

the sample covariance with `ddof = 1`, pooled about the **global** mean rather than
replication by replication, so between-replication variation is retained. The reduction is
two-pass and centred; no shrinkage, regularization, ridge or robust variant is applied.

The frozen estimates are

$$\hat\mu_y = \begin{bmatrix} -3.7095320080083676 \\ 4.440224301241895 \end{bmatrix},
\qquad
\hat\Sigma_y = \begin{bmatrix}
4.9036459698528825 & -0.054034090934843335 \\
-0.054034090934843335 & 0.027454457937879286
\end{bmatrix},$$

with spectral condition number $\kappa_2(\hat\Sigma_y) \approx 182.614$. These are frozen D3
estimates of the in-control monitoring distribution, not universal population parameters.

The one-component ablations of §11 re-estimate nothing: the sample mean of component 0 *is*
$\hat\mu_y[0]$ and its sample variance *is* $\hat\Sigma_y[0,0]$, so they use the
corresponding sub-vector and sub-matrix.

### Benchmark moments from D3

The classical benchmark monitors the raw observed vector $x_t \in \mathbb{R}^4$ and so needs
its own in-control moments $\hat\mu_x \in \mathbb{R}^4$ and $\hat\Sigma_x \in
\mathbb{R}^{4\times4}$. They are estimated with the **same estimator** (pooled about the
global mean, `ddof = 1`, no shrinkage) from the **same frozen D3 bank**, using the *aligned*
raw observations — raw times $t = 32, \dots, 3000$ of each replication, i.e. the rows
corresponding to reportable decisions. The 31 run-in rows of §11 are excluded, so the
benchmark's moments, like the hybrid's, are estimated on reported decisions only.

$\hat\mu_x$ and $\hat\Sigma_x$ are frozen alongside $\hat\mu_y$ and $\hat\Sigma_y$ and are
used exclusively by $M(.05)$. Their exact frozen values are carried in `models/mewma.npz`
and are not reproduced here.

---

## 10. MEWMA monitoring statistic

For a monitoring vector sequence $u_j \in \mathbb{R}^m$ with in-control mean $\mu$ and
covariance $\Sigma$, the multivariate EWMA state is

$$Z_j = \lambda\, u_j + (1 - \lambda)\, Z_{j-1}, \qquad Z_0 = \mu .$$

Every replication is reset to $Z_0 = \mu$; state is never carried across replications.

The chart standardizes the state by the **standard finite-start covariance scaling**

$$\Sigma^{(\text{chart})}_{Z,j} = c_j \Sigma,
\qquad
c_j = \frac{\lambda}{2 - \lambda}\Big[1 - (1-\lambda)^{2j}\Big],
\qquad c_1 = \lambda^2,$$

with $j$ the one-based monitoring index. This is the chart's reference scaling, not a claim
about the process: the monitored sequences are serially dependent — strongly so, because
consecutive windows overlap in 31 of 32 rows — so $c_j \Sigma$ must **not** be read as the
exact unconditional stochastic covariance of $Z_j$. The control limits are instead
calibrated empirically on D3 using the actual dependent process (§12).

The finite-start factor is used exactly, never its asymptotic limit $\lambda/(2-\lambda)$.
For $\lambda = 0.05$ the limit is about $10.3$ times larger than $c_1 = \lambda^2$. Using
the asymptotic factor at startup would overstate the covariance scaling and therefore
understate the early $T^2$ statistic.

The charting statistic is

$$T_j^2 = (Z_j - \mu)^{\mathsf T}\,\big(c_j \Sigma\big)^{-1}\,(Z_j - \mu).$$

Numerically, $\Sigma$ is never inverted. The quadratic form is evaluated through a
Cholesky-based linear solve: one Cholesky factorization of the unscaled $\Sigma$ serves
every replication, index and $\lambda$, and the scalar $c_j$ divides the resulting
Mahalanobis distance afterwards. The one-dimensional ablations pass through the same route
as $1\times1$ systems rather than dividing by a variance directly, so all charts share one
arithmetic path.

### Signal rule and run length

A signal is declared when

$$T_j^2 > h,$$

**strictly**. Equality, $T_j^2 = h$, is *not* a signal. This matters because the calibrated
limit $h^\ast$ is itself an attained value of a running maximum (§12), so the boundary case
occurs with certainty at the limit that matters.

The run length is the one-based index $j$ of the **first reported monitoring position**
satisfying $T_j^2 > h$, over the reportable horizon of 2969 positions. If no reported
position exceeds $h$, the run length is recorded as the final reportable index (2969) and
the replication is flagged **censored** by a separate boolean. The flag is the definition,
not an inference from the value: a genuine crossing at $j = 2969$ and a replication that
never crossed both store 2969, and only the flag distinguishes them.

---

## 11. Monitoring schemes

Six charts are evaluated. All are fixed in advance; none was added, removed or retuned
after observing D4 or D5.

| Chart | Monitored vector | Dim. | $\lambda$ | Moments | Role |
| --- | --- | --- | --- | --- | --- |
| $H(.05)$ | $y_t = [s_t, \log \mathrm{SPE}_t]$ | 2 | 0.05 | $\hat\mu_y, \hat\Sigma_y$ | primary hybrid |
| $H(.10)$ | $y_t$ | 2 | 0.10 | $\hat\mu_y, \hat\Sigma_y$ | smoothing sensitivity |
| $H(.20)$ | $y_t$ | 2 | 0.20 | $\hat\mu_y, \hat\Sigma_y$ | smoothing sensitivity |
| $S(.05)$ | $s_t$ only | 1 | 0.05 | $\hat\mu_y[0], \hat\Sigma_y[0,0]$ | ablation |
| $R(.05)$ | $\log \mathrm{SPE}_t$ only | 1 | 0.05 | $\hat\mu_y[1], \hat\Sigma_y[1,1]$ | ablation |
| $M(.05)$ | raw $x_t \in \mathbb{R}^4$ | 4 | 0.05 | $\hat\mu_x, \hat\Sigma_x$ | classical MEWMA benchmark |

The scalar schemes $S(.05)$ and $R(.05)$ are not a different charting rule: they use the
MEWMA formulation of §10 in one dimension, with the same $Z_0 = \mu$ initialisation, the
same finite-start factor, the same strict signal rule and the same run-length convention.

### Raw-data benchmark alignment

$H$, $S$ and $R$ cannot report a decision until a complete window of $L = 32$ raw
observations exists, so their first reported decision corresponds to raw time $t = 32$. The
classical benchmark watches $x_t$ directly and could otherwise report from $t = 1$;
comparing the two without correction would compare different head starts. The benchmark is
therefore placed on the same reporting axis by this convention:

1. Raw observations $t = 1, \dots, 31$ **do** enter the MEWMA recursion and **do** update
   the benchmark's state, starting from $Z_0 = \hat\mu_x$. They are a silent state run-in,
   not discarded data.
2. **No signal may be reported at those times.** They are not alarm opportunities and
   contribute to no run length, whatever value the statistic would have taken.
3. The first reported benchmark statistic corresponds to raw time $t = 32$, which is the
   **32nd** update of the recursion.
4. The finite-start factor at that first report is therefore $c_{32}$, not $c_1 =
   \lambda^2$. Subsequent reports use $c_{33}, c_{34}, \dots, c_{3000}$.
5. Reported run-length position $j = 1$ corresponds to raw time $t = 32$ for every chart, so
   the benchmark's 2969 reported positions are index-aligned with the hybrid charts' 2969
   reported positions and the run lengths are directly comparable.

Consistently with point 1, the 31 run-in rows are excluded from $\hat\mu_x$ and
$\hat\Sigma_x$ (§9).

---

## 12. Control-limit calibration

Each of the six charts is calibrated **independently** against the same in-control target
$\mathrm{ARL}_0 = 370$.

Calibration uses D3: **the same 300-replication D3 bank supplies both the in-control
moments of §9 and the control-limit search described here.** D4 is the separate, independent
in-control bank used for validation (§13) and plays no part in calibration.

| Calibration setting | Value |
| --- | --- |
| replications | 300 |
| raw trajectory horizon | 3000 |
| reportable horizon (all six charts) | 2969 |

The search is exact rather than approximate. Let $M_{i,j} = \max_{k \le j} T^2_{i,k}$ be the
running maximum of replication $i$. Empirical $\mathrm{ARL}_0(h)$ is a monotone
non-decreasing step function of $h$ whose jumps occur exactly at attained values of the
running maxima, so the candidate set is the sorted unique attained values of $M$. Bisection
over that finite set — valid because raising an attained limit can only delay or remove a
signal — returns

$$h^\ast = \min\{\,h \in \text{attained values} : \widehat{\mathrm{ARL}}_0(h) \ge 370\,\},$$

and minimality is asserted by checking that the neighbouring smaller candidate falls short.
$h^\ast$ is thus a realised statistic of the D3 run-length distribution, not an
interpolation and not a tabulated or $\chi^2$ limit.

The frozen calibrated limits are:

| Chart | $h^\ast$ | Achieved D3 $\mathrm{ARL}_0$ | Calibration censors |
| --- | --- | --- | --- |
| $H(.05)$ | 111.08301690320326 | 371.04 | 0 |
| $H(.10)$ | 83.58784283669753 | 370.54 | 0 |
| $H(.20)$ | 51.43064153149941 | 372.43666666666667 | 1 |
| $S(.05)$ | 66.045190051217 | 370.63 | 0 |
| $R(.05)$ | 71.02839557889465 | 371.34 | 0 |
| $M(.05)$ | 46.5246825248281 | 372.4266666666667 | 0 |

Because $\widehat{\mathrm{ARL}}_0$ is a step function, the achieved value generally
overshoots 370 by an amount that depends on the gap between adjacent attained values. The
overshoot is reported as attained and never tuned away; the small deviations from 370 carry
no further interpretation. All six charts are calibrated against the same target using
their own D3 monitoring distributions.

---

## 13. Phase II evaluation

Phase II applies the frozen preprocessing, autoencoder, One-Class SVM, in-control moments
and the six control limits to banks that played no part in any fit or calibration. Nothing
is fitted, estimated, selected or tuned in this phase.

### D4 — independent in-control validation

| Setting | Value |
| --- | --- |
| replications | 500 |
| raw horizon per replication | 3000 |
| reportable positions per replication | 2969 |
| condition | in control |

Each chart is run once at its single frozen $h^\ast$, to observe the realised in-control
run-length behaviour on data independent of the bank that produced its moments and its
limit. The D3 achieved $\mathrm{ARL}_0$ of §12 is reported alongside the D4 value so the
calibration value and the independent validation value are never confused. Numerical D4
results appear in `README.md` and `02_results.ipynb`.

### D5 — out-of-control evaluation

| Setting | Value |
| --- | --- |
| shift magnitudes | $\delta = 0.2, 0.4, \dots, 3.0$ (15 values) |
| replications per $\delta$ | 500 |
| raw horizon per replication | 3000 |
| change point | $\tau = 1$ (first retained observation) |

The same frozen models, moments and limits are applied at every shift magnitude; there is no
retraining, re-estimation or recalibration per $\delta$. Because $\tau = 1$, the raw
observations $x_1, \dots, x_{31}$ that fill the first window are already post-change, and the
first monitoring opportunity $j = 1$ is raw time $t = 32$. Run lengths are counted in
monitoring opportunities, so a signal at reported position $\mathrm{RL}$ corresponds to raw
time $t_{\text{signal}} = \mathrm{RL} + 31$.

The evaluation therefore produces $15 \text{ shifts} \times 6 \text{ charts} = 90$
chart/scenario cells, each a distribution of 500 run lengths with its censoring flags.
Numerical D5 results appear in `README.md` and `02_results.ipynb`.

---

## 14. Run-length summaries

Per-replication run lengths and censoring flags are persisted; all reported summaries are
derived from them. For a cell of $n$ run lengths $\mathrm{RL}_1, \dots, \mathrm{RL}_n$:

| Statistic | Definition |
| --- | --- |
| $\mathrm{ARL}$ | $\bar{\mathrm{RL}} = \frac{1}{n}\sum_i \mathrm{RL}_i$ |
| $\mathrm{SDRL}$ | $\Big[\frac{1}{n-1}\sum_i (\mathrm{RL}_i - \bar{\mathrm{RL}})^2\Big]^{1/2}$ (sample convention, `ddof = 1`) |
| $\mathrm{MRL}$ | $\operatorname{median}(\mathrm{RL})$ |
| standard error | $\mathrm{SDRL}/\sqrt{n}$ |

For D4 the mean is the validated in-control $\mathrm{ARL}_0$; for a D5 cell it is the
out-of-control $\mathrm{ARL}_1$ at that shift magnitude.

### Censoring

Censoring is tracked as an explicit indicator, never inferred from the run-length value.
Two distinct events must not be conflated:

- a **genuine signal** on the final reportable index ($T^2_{2969} > h$): run length 2969,
  not censored;
- **no signal** by the final reportable index: run length recorded as 2969, censored.

Each cell reports its censor count and censor fraction. A cell whose censor fraction exceeds
$0.05$ is flagged *truncation-limited*, indicating an $\mathrm{ARL}$ bounded by the finite
horizon rather than estimated from it.

### Paired hybrid-versus-benchmark contrast

$H(.05)$ and $M(.05)$ monitor the same replications of the same D5 bank, so the comparison
is made replication by replication. At shift magnitude $\delta$, for replication $i$,

$$d_i(\delta) = \mathrm{RL}_{H,i}(\delta) - \mathrm{RL}_{M,i}(\delta),$$

with paired mean difference and paired standard error

$$\bar d(\delta) = \frac{1}{n}\sum_{i=1}^{n} d_i(\delta),
\qquad
\mathrm{SE}_{\text{paired}}(\delta) = \frac{s_d(\delta)}{\sqrt{n}},
\qquad n = 500,$$

where $s_d$ is the sample standard deviation of the differences with `ddof = 1`. Pairing
accounts for shared replication-level variation and estimates the run-length difference
directly within each common D5 replication. A negative $\bar d(\delta)$ means $H(.05)$
signalled sooner than $M(.05)$; the empirical values are reported in `README.md` and
`02_results.ipynb`.

---

## 15. Reproducibility conventions

- **Deterministic seeded generation.** Master seed 795, `PCG64DXSM`, no global random state.
  The `SeedSequence` entropy for one trajectory is `[1, 795]` for D1, `[2, 795]` for D2,
  `[3, replication, 795]` for D3, `[4, replication, 795]` for D4, and
  `[5, round(1000δ), replication, 795]` for D5. A stream is identified by what it is for,
  never by generation order, so a replication generated alone equals the same replication
  taken from a full bank. Within a trajectory the draw order is fixed: one standard-normal
  block of shape $(500 + N, 8)$ scaled by the innovation Cholesky factor, then one block of
  shape $(N, 4)$ scaled by the measurement factor.
- **Frozen artifacts.** Preprocessing statistics are frozen after D1, the autoencoder after
  D1 training, the One-Class SVM after D2, and the moments and control limits after D3.
  Phase II loads and applies them; it contains no fitting, estimation or calibration path.
- **Precision.** Process simulation, preprocessing statistics, the SPE reduction, the
  One-Class SVM fit and scoring, the moment reduction and all MEWMA arithmetic are
  `float64`. Network parameters and activations are `float32`, with an explicit
  `float32 -> float64` cast at the feature boundary.
- **Batching.** Autoencoder inference uses the canonical batch size of 64, and banks are
  evaluated one replication at a time — the partition the frozen results were produced with,
  which also makes a window spanning two replications structurally impossible.
- **Fixed numerical routes.** Pairwise distances for the median heuristic are accumulated as
  direct sums of squares rather than through the Gram expansion; the moment reduction is
  two-pass and centred rather than $E[yy^{\mathsf T}] - \mu\mu^{\mathsf T}$; the $T^2$ form
  is evaluated by a Cholesky-based solve rather than by forming an inverse. These choices
  are part of the specification because floating-point addition is not associative.
- **Single-threaded native execution.** Every runtime-detected native numerical thread pool
  is constrained to one thread across the frozen numerical path, and the constraint is
  verified to have taken effect, so no reduction is silently reassociated across threads.
- **Finite-start covariance.** All MEWMA statistics use $c_j$ exactly, never its asymptotic
  limit.
- **No Phase II refitting.** No preprocessing statistic, network weight, SVM parameter,
  moment or control limit is updated on D4 or D5.
- **Committed artifacts.** The artifacts under `models/` and the summaries under `results/`
  are the experiment reported by the thesis.

Exact bitwise equivalence across arbitrary hardware, operating systems, BLAS or PyTorch
builds, and dependency versions is not universally guaranteed. Reproducibility here is
defined relative to the documented software environment and the numerical conventions above.

---

## 16. Frozen experimental specification

| Group | Parameter | Value |
| --- | --- | --- |
| Process | latent dimension $d_z$ | 8 |
| | observed dimension $p$ | 4 |
| | burn-in $B$ | 500 |
| | $\Phi$ diagonal / off-diagonal | 0.55 / 0.15 |
| | $\rho(\Phi)$ | $\approx 0.8319077862$ |
| | innovation covariance $Q$ | $0.20\,I_8$ |
| | observation-noise covariance $\Sigma_\varepsilon$ | $0.05\,I_4$ |
| | observation map | $x_t = W_2\tanh(z_t) + \varepsilon_t$ |
| Fault | direction $v$ | $\tfrac{1}{\sqrt2}[1,1,0,0,0,0,0,0]^{\mathsf T}$ |
| | change point $\tau$ | 1 (first retained observation) |
| | magnitudes $\delta$ | $0.2$ to $3.0$ in steps of $0.2$ (15) |
| Windowing | window length $L$ / stride | 32 / 1 |
| | first monitorable raw time | $t = 32$ |
| | orientation | channels-first, $(W, 4, 32)$ |
| Preprocessing | per-variable standardization, `ddof = 0` | fitted on D1 rows $[0,1600)$ |
| CAE | latent dimension $q$ | 8 |
| | trainable parameters | 3180 |
| | learning rate / batch size | $10^{-3}$ / 64 |
| | max epochs / patience | 100 / 10 |
| | training seed | 3795 |
| SPE | reduction | sum over $4\times32$ window elements, `float64` |
| | log floor $\epsilon$ | $10^{-12}$ |
| OC-SVM | kernel / $\nu$ | RBF / 0.03 |
| | $\gamma$ | 0.030522830729177747 |
| | support vectors | 78 |
| Monitoring | hybrid dimension | 2 |
| | component order | $[s_t, \log \mathrm{SPE}_t]$ |
| | primary $\lambda$ | 0.05 |
| | sensitivity $\lambda$ | 0.10, 0.20 |
| | moment convention | pooled on D3, `ddof = 1` |
| | signal rule | $T_j^2 > h$, strict |
| | target $\mathrm{ARL}_0$ | 370 |
| | reportable horizon | 2969 |
| Evaluation | D4 replications | 500 |
| | D5 replications per $\delta$ | 500 |
| | D5 scenarios / cells | 15 / 90 |
| | RNG master seed | 795 (`PCG64DXSM`) |
