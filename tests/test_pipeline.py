"""Regression suite for the simplified thesis pipeline.

Guards the frozen scientific quantities of the completed experiment: the design
constants, the deterministic data generation, the frozen Stage 3/4/5 artifacts,
the MEWMA conventions, and the published Phase II results.

Nothing here fits, calibrates or re-runs Phase II. The expensive banks (D3, D4,
D5) are never generated; the committed result artifacts are read instead. The
whole file runs in a few seconds.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline as P  # noqa: E402  (needs the repository root on sys.path)


# --- session fixtures: generate/infer once -----------------------------------


@pytest.fixture(scope="session")
def d1():
    return P.generate_d1()


@pytest.fixture(scope="session")
def d2():
    return P.generate_d2()


@pytest.fixture(scope="session")
def frozen():
    return P.load_frozen()


@pytest.fixture(scope="session")
def d2_inference(frozen, d2):
    """Frozen CAE inference on D2: latent features and SPE."""
    return P.infer_series(frozen.autoencoder, d2, frozen.mean, frozen.scale)


@pytest.fixture(scope="session")
def d2_monitoring(frozen, d2):
    """The hybrid monitoring vectors y_t = [s_t, log SPE_t] on D2."""
    return P.monitoring_series(frozen, d2)


@pytest.fixture(scope="session")
def d4_summary():
    return json.loads((P.PHASE2_DIR / "d4_validation_summary.json").read_text())


@pytest.fixture(scope="session")
def d5_summary():
    return json.loads((P.PHASE2_DIR / "d5_performance_summary.json").read_text())


# --- 1. frozen design constants ----------------------------------------------


def test_frozen_design_constants():
    assert P.OBS_DIM == 4
    assert P.LATENT_DIM == 8
    assert P.WINDOW_LENGTH == 32
    assert P.CAE_LATENT_DIM == 8
    assert P.PRIMARY_LAMBDA == 0.05
    assert P.TARGET_ARL0 == 370
    assert P.SHIFT_GRID.shape == (15,)
    assert P.FIRST_REPRESENTED_TIME_INDEX == P.WINDOW_LENGTH


def test_chart_plan_is_the_six_pre_registered_charts():
    assert P.chart_plan() == (
        ("H", 0.05),
        ("H", 0.10),
        ("H", 0.20),
        ("S", 0.05),
        ("R", 0.05),
        ("M", 0.05),
    )
    # The benchmark is the only chart with a silent state run-in.
    assert [P.CHARTS[c]["update_index_offset"] for c, _ in P.chart_plan()] == [
        0, 0, 0, 0, 0, 31
    ]


# --- 2. deterministic data generation ----------------------------------------


def test_d1_and_d2_are_deterministic(d1, d2):
    assert d1.shape == (P.D1_LENGTH, P.OBS_DIM)
    assert d2.shape == (P.D2_LENGTH, P.OBS_DIM)
    assert d1.dtype == np.float64 and d2.dtype == np.float64
    assert np.isfinite(d1).all() and np.isfinite(d2).all()
    # Distinct RNG namespaces, and regeneration is bitwise identical.
    assert not np.array_equal(d1, d2)
    assert np.array_equal(d1, P.generate_d1())
    assert np.array_equal(d2, P.generate_d2())


# --- 3. D1 split, preprocessing and windowing --------------------------------


def test_d1_split_and_window_counts(d1):
    train, validation = P.split_d1(d1)
    assert train.shape == (1600, 4)
    assert validation.shape == (400, 4)
    # Split before windowing: the 31 straddling windows are never formed.
    assert P.window_series(train).shape == (1569, 4, 32)
    assert P.window_series(validation).shape == (369, 4, 32)


def test_frozen_preprocessing_matches_a_refit_on_d1_training_rows(d1):
    train, _ = P.split_d1(d1)
    mean, scale = P.fit_preprocessing(train)
    frozen_mean, frozen_scale = P.load_preprocessing()
    assert mean.shape == (4,) and scale.shape == (4,)
    assert np.isfinite(scale).all() and (scale > 0).all()
    assert np.array_equal(mean, frozen_mean)
    assert np.array_equal(scale, frozen_scale)


# --- 4. frozen autoencoder ---------------------------------------------------


def test_autoencoder_has_the_frozen_parameter_count(frozen):
    assert sum(p.numel() for p in frozen.autoencoder.parameters()) == 3180


def test_frozen_inference_on_d2(d2_inference):
    features, spe_values = d2_inference
    assert features.shape == (1969, P.CAE_LATENT_DIM)
    assert features.dtype == np.float32
    assert spe_values.shape == (1969,)
    assert spe_values.dtype == np.float64
    assert np.isfinite(features).all()
    assert np.isfinite(spe_values).all() and (spe_values > 0).all()


# --- 5. frozen OC-SVM --------------------------------------------------------


def test_frozen_ocsvm_state(frozen):
    ocsvm = frozen.ocsvm
    assert ocsvm.gamma == 0.030522830729177747
    assert ocsvm.rho == 8.846560927608301
    assert ocsvm.support_vectors.shape == (78, 8)
    assert ocsvm.dual_coefficients.shape == (78,)
    assert ocsvm.support_vectors.dtype == np.float64


def test_ocsvm_scores_d2_features(frozen, d2_inference):
    features, _ = d2_inference
    with P.single_threaded():
        scores = P.ocsvm_score(frozen.ocsvm, P.to_ocsvm_input(features))
    assert scores.shape == (1969,)
    assert scores.dtype == np.float64
    assert np.isfinite(scores).all()


# --- 6. hybrid monitoring vector ---------------------------------------------


def test_monitoring_vector_shape_and_column_order(frozen, d2_inference, d2_monitoring):
    features, spe_values = d2_inference
    assert d2_monitoring.shape == (1969, P.MONITORING_VECTOR_DIM)
    assert d2_monitoring.dtype == np.float64
    assert np.isfinite(d2_monitoring).all()
    # Column 0 is s_t, column 1 is log SPE_t. The order is frozen and nothing
    # downstream carries a label, so a swap would silently reinterpret every
    # calibrated limit.
    assert P.MONITORING_COMPONENTS == ("ocsvm_score", "log_spe")
    # Recomputed under the same single-threaded numerical boundary the frozen
    # path runs in, so the comparison can be exact.
    with P.single_threaded():
        scores = P.ocsvm_score(frozen.ocsvm, P.to_ocsvm_input(features))
    assert np.array_equal(d2_monitoring[:, 0], scores)
    assert np.array_equal(d2_monitoring[:, 1], P.log_spe(spe_values))


# --- 7. frozen Stage 5 moments and control limits ----------------------------


def test_frozen_in_control_moments():
    hybrid, benchmark = P.load_moments()
    assert hybrid.mean.shape == (2,) and hybrid.covariance.shape == (2, 2)
    assert benchmark.mean.shape == (4,) and benchmark.covariance.shape == (4, 4)
    assert hybrid.mean.tolist() == [-3.7095320080083676, 4.440224301241895]
    assert hybrid.covariance.tolist() == [
        [4.9036459698528825, -0.054034090934843335],
        [-0.054034090934843335, 0.027454457937879286],
    ]
    assert np.array_equal(hybrid.covariance, hybrid.covariance.T)
    assert np.array_equal(benchmark.covariance, benchmark.covariance.T)


def test_frozen_control_limits():
    limits = P.load_control_limits()
    assert limits == {
        "H_0p05": 111.08301690320326,
        "H_0p10": 83.58784283669753,
        "H_0p20": 51.43064153149941,
        "S_0p05": 66.045190051217,
        "R_0p05": 71.02839557889465,
        "M_0p05": 46.5246825248281,
    }
    # One limit per pre-registered chart, keyed by the frozen spelling.
    assert set(limits) == {P.limit_key(c, v) for c, v in P.chart_plan()}


# --- 8. MEWMA conventions (small synthetic arrays) ---------------------------


def test_finite_start_factor_at_the_first_reported_decision():
    lam = P.PRIMARY_LAMBDA
    # H/S/R report from j = 1, where the exact factor is lambda^2 -- about ten
    # times smaller than the asymptotic limit lambda / (2 - lambda).
    assert P.finite_start_factors(3, lam)[0] == lam * lam
    # The benchmark absorbs 31 silent updates, so its first reported factor is
    # c_32 and emphatically not lambda^2.
    first_index = P.CHARTS["M"]["update_index_offset"] + 1
    assert first_index == 32
    benchmark_first = P.finite_start_factors(3, lam, first_index)[0]
    assert benchmark_first == lam / (2.0 - lam) * (1.0 - (1.0 - lam) ** 64.0)
    assert benchmark_first > lam * lam


def test_signal_rule_is_strictly_greater_than_the_limit():
    running_max = np.array([[1.0, 5.0, 5.0, 5.0]])
    # T2 exactly at the limit does not signal; a hair above it does.
    lengths, censored = P.run_lengths(running_max, 5.0)
    assert lengths.tolist() == [4] and censored.tolist() == [True]
    lengths, censored = P.run_lengths(running_max, 4.999)
    assert lengths.tolist() == [2] and censored.tolist() == [False]


def test_run_length_indexing_and_censoring():
    running_max = np.array([
        [0.0, 0.0, 5.0, 5.0],   # first crossing at the third decision
        [0.0, 0.0, 0.0, 5.0],   # genuine crossing on the final decision
        [0.0, 0.0, 0.0, 0.0],   # never crossed: censored at the horizon
    ])
    lengths, censored = P.run_lengths(running_max, 1.0)
    # Run lengths are one-based monitoring indices.
    assert lengths.tolist() == [3, 4, 4]
    # A final-index crossing and a censored run store the same length, and only
    # the flag tells them apart.
    assert censored.tolist() == [False, False, True]


def test_mewma_state_resets_per_replication_and_recursion_is_exact():
    mean = np.zeros(2)
    values = np.zeros((2, 3, 2))
    values[0, 0] = [1.0, 2.0]
    lam = 0.5
    states = P.mewma_states(values, mean, lam)
    # m_1 = lambda * y_1 + (1 - lambda) * mu
    assert states[0, 0].tolist() == [0.5, 1.0]
    assert states[0, 1].tolist() == [0.25, 0.5]
    # The second replication starts again from mu, never from replication 0.
    assert np.array_equal(states[1], np.zeros((3, 2)))


# --- 9. committed Phase II result artifacts ----------------------------------


def test_persisted_run_lengths_back_the_summaries():
    keys = [P.limit_key(chart, value) for chart, value in P.chart_plan()]
    with np.load(P.PHASE2_DIR / "run_lengths.npz", allow_pickle=False) as archive:
        names = set(archive.files)
        # One run-length and one censoring array per chart cell: the six charts
        # on D4, and the six charts on each of the fifteen D5 scenarios.
        expected = {
            f"d4/{key}/{field}"
            for key in keys
            for field in ("run_lengths", "censored")
        }
        expected |= {
            f"d5/{key}/{P.scenario_name(delta)}/{field}"
            for key in keys
            for delta in P.SHIFT_GRID
            for field in ("run_lengths", "censored")
        }
        assert names == expected

        lengths = archive["d4/H_0p05/run_lengths"]
        censored = archive["d4/H_0p05/censored"]
        assert lengths.shape == (P.D4_REPLICATIONS,)
        assert lengths.dtype == np.int64 and censored.dtype == bool
        assert lengths.min() >= 1 and lengths.max() <= P.CALIBRATION_HORIZON
        # The persisted run lengths are what the published summary reports.
        assert float(lengths.mean()) == 336.534
        assert not censored.any()

        # The paired primary contrast is likewise recoverable from them.
        h = archive["d5/H_0p05/delta_0p2/run_lengths"].astype(np.float64)
        m = archive["d5/M_0p05/delta_0p2/run_lengths"].astype(np.float64)
        assert float((h - m).mean()) == -3.17


def test_d4_validation_arl0(d4_summary):
    charts = d4_summary["charts"]
    assert len(charts) == 6
    assert [(c["chart_id"], c["lambda"]) for c in charts] == list(P.chart_plan())
    assert {P.limit_key(c["chart_id"], c["lambda"]): c["arl0_validation_h"]
            for c in charts} == {
        "H_0p05": 336.534,
        "H_0p10": 322.332,
        "H_0p20": 315.394,
        "S_0p05": 351.412,
        "R_0p05": 363.444,
        "M_0p05": 403.528,
    }
    assert all(c["n_replications"] == P.D4_REPLICATIONS for c in charts)
    assert all(c["censor_count"] == 0 for c in charts)


def test_d5_grid_is_complete_and_uncensored(d5_summary):
    cells = d5_summary["cells"]
    assert len(cells) == 15 * 6
    assert sorted({c["delta"] for c in cells}) == P.SHIFT_GRID.tolist()
    assert {(c["chart_id"], c["lambda"]) for c in cells} == set(P.chart_plan())
    # Every D5 cell signalled within the horizon in every replication, so no
    # ARL1 here is truncation-limited.
    assert all(c["censor_count"] == 0 for c in cells)
    assert not any(c["truncation_limited"] for c in cells)


def test_primary_contrast_against_the_benchmark(d5_summary):
    contrast = d5_summary["primary_contrast"]
    assert len(contrast) == 15
    assert [c["delta"] for c in contrast] == P.SHIFT_GRID.tolist()
    at_smallest_shift = contrast[0]
    assert at_smallest_shift["delta"] == 0.2
    # RL_H - RL_M, paired by replication; negative means H signalled sooner.
    assert at_smallest_shift["mean_difference"] == -3.17
    assert at_smallest_shift["se_paired"] == 15.935695497277925
    assert all(c["n_replications"] == P.D5_REPLICATIONS for c in contrast)


def test_hybrid_beats_its_ablations_and_loses_to_the_benchmark_beyond_the_smallest_shift(
    d5_summary,
):
    arl1 = {
        (c["delta"], c["chart_id"], c["lambda"]): c["arl1_h"]
        for c in d5_summary["cells"]
    }
    for delta in P.SHIFT_GRID.tolist():
        hybrid = arl1[(delta, "H", 0.05)]
        # Both monitoring components together detect faster than either alone.
        assert hybrid < arl1[(delta, "S", 0.05)]
        assert hybrid < arl1[(delta, "R", 0.05)]
        # The conventional MEWMA on the raw process overtakes the hybrid chart
        # everywhere except at the smallest shift.
        if delta >= 0.4:
            assert arl1[(delta, "M", 0.05)] < hybrid
