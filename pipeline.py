"""Hybrid CAE / OC-SVM / MEWMA process-monitoring experiment."""

import copy
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from sklearn.svm import OneClassSVM
from threadpoolctl import threadpool_info, threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent


# Scientific constants

SEED = 795

LATENT_DIM = 8
BURN_IN = 500
INITIAL_STATE = np.zeros(LATENT_DIM)


PHI = (
    np.diag(np.full(LATENT_DIM, 0.55))
    + np.diag(np.full(LATENT_DIM - 1, 0.15), 1)
    + np.diag(np.full(LATENT_DIM - 1, 0.15), -1)
)


assert float(np.max(np.abs(np.linalg.eigvals(PHI)))) < 1.0


Q = 0.20 * np.eye(LATENT_DIM)


OBS_DIM = 4

W1 = np.eye(LATENT_DIM)
B1 = np.zeros(LATENT_DIM)


W2 = np.array(
    [
        [-0.60447198, -0.06150649, -0.37179578, -0.01161746,
         0.05885824, 0.52361117, 0.45120458, -0.10604239],
        [-0.04842081, 0.36085142, -0.83557846, -0.11135410,
         -0.18152197, 0.30422580, 0.15832824, 0.07924137],
        [-0.05518327, 0.45874929, -0.21939336, -0.17469105,
         -0.55464952, 0.30578469, -0.13652085, -0.53672635],
        [-0.04100004, -0.84305316, -0.40588165, -0.11509976,
         0.26627250, 0.07542546, 0.16952036, -0.06530930],
    ]
)

B2 = np.zeros(OBS_DIM)


R = 0.05 * np.eye(OBS_DIM)


D1_LENGTH = 2000
D2_LENGTH = 2000
D3_REPLICATIONS = 300
D3_HORIZON = 3000
D4_REPLICATIONS = 500
D4_HORIZON = 3000
D5_REPLICATIONS = 500
D5_HORIZON = 3000


FAULT_DIRECTION = np.array(
    [0.7071067811865476, 0.7071067811865476, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)

CHANGE_POINT = 1

SHIFT_GRID = np.array(
    [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6,
     1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
    dtype=np.float64,
)


def delta_code(delta):
    """Return the stable integer code for a shift magnitude."""
    return int(round(1000.0 * float(delta)))


STAGE3_SEED = 3795

WINDOW_LENGTH = 32
WINDOW_STRIDE = 1
TRAIN_FRACTION = 0.8


PREPROCESSING_METHOD = "per_channel_standardization"
PREPROCESSING_DDOF = 0
SCALE_THRESHOLD = 1e-12


INPUT_CHANNELS = 4


CAE_LATENT_DIM = 8
ENCODER_CHANNELS = (8, 16)
KERNEL_SIZE = 3
CONV_STRIDE = 2
PADDING = 1
OUTPUT_PADDING = 1


LOSS = "mse"
OPTIMIZER = "adam"
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
SHUFFLE = True
VALIDATION_SHUFFLE = False
NUM_WORKERS = 0


SPE_EPSILON = 1e-12


FITTING_DATASET = "D2"

FEATURE_DIM = 8


FIRST_REPRESENTED_TIME_INDEX = 32


FEATURE_SCALING = False

KERNEL = "rbf"
NU = 0.03


GAMMA_RULE = "inverse_median_pairwise_squared_distance"

SVM_TOL = 1e-3
SVM_SHRINKING = True
SVM_MAX_ITER = -1
SVM_CACHE_SIZE = 200
SVM_VERBOSE = False


MONITORING_DATASET = "D3"
MONITORING_VECTOR_DIM = 2
MONITORING_COMPONENTS = ("ocsvm_score", "log_spe")

SPE_FLOOR = 1e-12


MOMENTS_DDOF = 1
CONDITION_NUMBER_GUARD = 1e8


PRIMARY_LAMBDA = 0.05
HYBRID_SENSITIVITY_LAMBDAS = (0.10, 0.20)
BENCHMARK_LAMBDA = 0.05

TARGET_ARL0 = 370

CALIBRATION_HORIZON = 2969
TRUNCATION_FRACTION_THRESHOLD = 0.05


CHARTS = {
    "H": {
        "role": "primary",
        "source": "hybrid_monitoring_vector",
        "components": (0, 1),
        "dimension": 2,
        "update_index_offset": 0,
    },
    "S": {
        "role": "ablation",
        "source": "hybrid_monitoring_vector",
        "components": (0,),
        "dimension": 1,
        "update_index_offset": 0,
    },
    "R": {
        "role": "ablation",
        "source": "hybrid_monitoring_vector",
        "components": (1,),
        "dimension": 1,
        "update_index_offset": 0,
    },
    "M": {
        "role": "benchmark",
        "source": "aligned_raw_observations",
        "components": (0, 1, 2, 3),
        "dimension": 4,
        "update_index_offset": 31,
    },
}


AUTOCORRELATION_LAG_MAX = 64


DEVICE = "cpu"
TORCH_THREADS = 1
NATIVE_THREAD_LIMIT = 1
INFERENCE_BATCH_SIZE = 64


CUBLAS_WORKSPACE_CONFIG = ":4096:8"


PRECISION_NETWORK_PARAMETERS_AND_ACTIVATIONS = "float32"
PRECISION_PREPROCESSING_STATISTICS = "float64"
PRECISION_SPE_REDUCTION = "float64"
PRECISION_STAGE2_OBSERVATIONS = "float64"
PRECISION_FEATURE_CAST = "float64"
PRECISION_FIT_AND_SCORE = "float64"


# Data generation

INNOVATION_FACTOR = np.linalg.cholesky(Q)
MEASUREMENT_FACTOR = np.linalg.cholesky(R)


def apply_matrix(matrix, vectors):
    """Apply a matrix along the last axis using fixed accumulation order."""
    result = np.zeros(vectors.shape[:-1] + (matrix.shape[0],), dtype=np.float64)
    for column in range(matrix.shape[1]):
        result += vectors[..., column, None] * matrix[:, column]
    return result


def generator_for(dataset, replication=None, delta=None, seed=SEED):
    """Return the deterministic RNG stream for one trajectory."""
    namespace = ("D1", "D2", "D3", "D4", "D5").index(dataset) + 1
    if dataset in ("D1", "D2"):
        entropy = [namespace, seed]
    elif dataset in ("D3", "D4"):
        entropy = [namespace, replication, seed]
    else:
        entropy = [namespace, delta_code(delta), replication, seed]
    return np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entropy)))


def draw_trajectory_noise(generator, horizon):
    innovations = apply_matrix(
        INNOVATION_FACTOR,
        generator.standard_normal((BURN_IN + horizon, LATENT_DIM)),
    )
    measurement_noise = apply_matrix(
        MEASUREMENT_FACTOR,
        generator.standard_normal((horizon, OBS_DIM)),
    )
    return innovations, measurement_noise


def simulate(innovations, measurement_noise, shift=None):
    """Generate observed trajectories from pre-drawn process noise."""
    replications, total_steps, _ = innovations.shape
    horizon = total_steps - BURN_IN

    zero_mean = np.zeros(LATENT_DIM, dtype=np.float64)
    post_change_mean = (
        zero_mean if shift is None else np.asarray(shift, dtype=np.float64)
    )

    state = np.empty((replications, LATENT_DIM), dtype=np.float64)
    state[:] = INITIAL_STATE
    observations = np.empty((replications, horizon, OBS_DIM), dtype=np.float64)

    for step in range(total_steps):


        index = step + 1 - BURN_IN
        current = post_change_mean if index >= CHANGE_POINT else zero_mean
        previous = post_change_mean if index - 1 >= CHANGE_POINT else zero_mean
        state = current + apply_matrix(PHI, state - previous) + innovations[:, step]
        if index >= 1:
            hidden = np.tanh(apply_matrix(W1, state) + B1)
            observations[:, index - 1] = (
                apply_matrix(W2, hidden) + B2 + measurement_noise[:, index - 1]
            )

    return observations


def simulate_bank(dataset, replications, horizon, delta=None):
    innovations = np.empty(
        (replications, BURN_IN + horizon, LATENT_DIM), dtype=np.float64
    )
    measurement_noise = np.empty((replications, horizon, OBS_DIM), dtype=np.float64)
    for replication in range(replications):
        generator = generator_for(dataset, replication=replication, delta=delta)
        innovations[replication], measurement_noise[replication] = (
            draw_trajectory_noise(generator, horizon)
        )

    shift = None if delta is None else float(delta) * FAULT_DIRECTION
    return simulate(innovations, measurement_noise, shift=shift)


def generate_d1():
    return simulate_bank("D1", 1, D1_LENGTH)[0]


def generate_d2():
    return simulate_bank("D2", 1, D2_LENGTH)[0]


def generate_d3():
    return simulate_bank("D3", D3_REPLICATIONS, D3_HORIZON)


def generate_d4():
    return simulate_bank("D4", D4_REPLICATIONS, D4_HORIZON)


def generate_d5_scenario(delta):
    return simulate_bank("D5", D5_REPLICATIONS, D5_HORIZON, delta=delta)


def scenario_name(delta):
    whole, thousandths = divmod(delta_code(delta), 1000)
    text = f"{whole}.{thousandths:03d}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return "delta_" + text.replace(".", "p")


def generate_datasets(directory):
    """Generate D1-D5 under ``directory``."""
    directory = Path(directory)
    (directory / "D5").mkdir(parents=True, exist_ok=True)
    for name, generate in (
        ("D1", generate_d1),
        ("D2", generate_d2),
        ("D3", generate_d3),
        ("D4", generate_d4),
    ):
        np.savez(directory / f"{name}.npz", x=generate())
    for delta in SHIFT_GRID:
        np.savez(
            directory / "D5" / f"{scenario_name(delta)}.npz",
            x=generate_d5_scenario(delta),
        )
    return directory


# Windowing and preprocessing

def split_d1(observations):
    """Chronologically split D1 into training and validation rows."""
    split = int(observations.shape[0] * TRAIN_FRACTION)
    return observations[:split], observations[split:]


def fit_preprocessing(train_observations):
    """Fit per-channel standardization on D1 training rows."""
    mean = train_observations.mean(axis=0, dtype=np.float64)
    scale = train_observations.std(axis=0, ddof=PREPROCESSING_DDOF, dtype=np.float64)


    assert np.all(np.isfinite(scale)) and np.all(scale > SCALE_THRESHOLD)
    return mean, scale


def apply_preprocessing(observations, mean, scale):
    return (observations - mean) / scale


def window_series(series):
    """Return stride-1 windows as ``(W, C, L)``."""
    view = np.lib.stride_tricks.sliding_window_view(series, WINDOW_LENGTH, axis=0)
    return view[::WINDOW_STRIDE]


def window_bank(bank):
    """Window each replication independently."""
    view = np.lib.stride_tricks.sliding_window_view(bank, WINDOW_LENGTH, axis=1)
    return view[:, ::WINDOW_STRIDE]


# Convolutional autoencoder

AUTOENCODER_PATH = PROJECT_ROOT / "models" / "autoencoder.pt"


def configure_determinism(seed=STAGE3_SEED):
    """Configure deterministic CPU training."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))


def _conv_output_length(length):
    return (length + 2 * PADDING - KERNEL_SIZE) // CONV_STRIDE + 1


ENCODED_LENGTH = _conv_output_length(_conv_output_length(WINDOW_LENGTH))
FLATTENED_DIM = ENCODER_CHANNELS[1] * ENCODED_LENGTH


class ConvAutoencoder1d(nn.Module):
    """Fixed 1D convolutional autoencoder."""

    def __init__(self):
        super().__init__()
        first, second = ENCODER_CHANNELS
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, first, KERNEL_SIZE, CONV_STRIDE, PADDING),
            nn.ReLU(),
            nn.Conv1d(first, second, KERNEL_SIZE, CONV_STRIDE, PADDING),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.encoder_linear = nn.Linear(FLATTENED_DIM, CAE_LATENT_DIM)
        self.decoder_linear = nn.Sequential(
            nn.Linear(CAE_LATENT_DIM, FLATTENED_DIM),
            nn.ReLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (second, ENCODED_LENGTH)),
            nn.ConvTranspose1d(
                second, first, KERNEL_SIZE, CONV_STRIDE, PADDING,
                output_padding=OUTPUT_PADDING,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                first, INPUT_CHANNELS, KERNEL_SIZE, CONV_STRIDE, PADDING,
                output_padding=OUTPUT_PADDING,
            ),
        )

    def encode(self, windows):
        return self.encoder_linear(self.encoder_conv(windows))

    def decode(self, features):
        return self.decoder_conv(self.decoder_linear(features))

    def forward(self, windows):
        return self.decode(self.encode(windows))


def load_autoencoder(path=AUTOENCODER_PATH):
    model = ConvAutoencoder1d()
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def to_network_input(windows):
    return torch.as_tensor(np.ascontiguousarray(windows), dtype=torch.float32)


def _epoch_loss(model, loader, criterion, optimizer=None):
    model.train() if optimizer else model.eval()
    total = 0.0
    count = 0
    with torch.enable_grad() if optimizer else torch.no_grad():
        for (batch,) in loader:
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            if optimizer:
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * batch.shape[0]
            count += batch.shape[0]
    return total / count


def train_autoencoder(train_windows, validation_windows):
    """Train on D1 and restore the best validation checkpoint."""
    configure_determinism()
    model = ConvAutoencoder1d()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(STAGE3_SEED)
    train_loader = DataLoader(
        TensorDataset(to_network_input(train_windows)),
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        generator=shuffle_generator,
        num_workers=NUM_WORKERS,
    )
    validation_loader = DataLoader(
        TensorDataset(to_network_input(validation_windows)),
        batch_size=BATCH_SIZE,
        shuffle=VALIDATION_SHUFFLE,
        num_workers=NUM_WORKERS,
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = None
    since_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = _epoch_loss(model, train_loader, criterion, optimizer)
        validation_loss = _epoch_loss(model, validation_loader, criterion)
        history.append((epoch, train_loss, validation_loss))
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "epochs": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
    }


# CAE inference

PREPROCESSING_PATH = PROJECT_ROOT / "models" / "preprocessing.npz"


@contextmanager
def single_threaded():
    """Constrain Torch and native numerical pools to one thread."""
    previous_torch_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(TORCH_THREADS)
        with threadpool_limits(limits=NATIVE_THREAD_LIMIT):
            assert torch.get_num_threads() == TORCH_THREADS
            assert all(
                pool["num_threads"] == NATIVE_THREAD_LIMIT
                for pool in threadpool_info()
            )
            yield
    finally:
        torch.set_num_threads(previous_torch_threads)


def load_preprocessing(path=PREPROCESSING_PATH):
    with np.load(path) as archive:
        return archive["mean"], archive["scale"]


def infer_windows(model, windows, batch_size=INFERENCE_BATCH_SIZE):
    features = []
    reconstructions = []
    with torch.no_grad():
        for start in range(0, windows.shape[0], batch_size):
            batch = to_network_input(windows[start : start + batch_size])
            encoded = model.encode(batch)
            features.append(encoded.numpy())
            reconstructions.append(model.decode(encoded).numpy())
    return np.concatenate(features), np.concatenate(reconstructions)


def spe(windows, reconstructions):
    """Return summed squared reconstruction error per window in float64."""
    residual = np.asarray(windows, dtype=np.float64) - reconstructions.astype(np.float64)
    return np.sum(residual * residual, axis=(-2, -1), dtype=np.float64)


def log_spe(spe_values):
    return np.log(np.maximum(spe_values, SPE_EPSILON))


def infer_series(model, observations, mean, scale, batch_size=INFERENCE_BATCH_SIZE):
    """Return CAE features and SPE for one series."""
    with single_threaded():
        windows = window_series(apply_preprocessing(observations, mean, scale))
        features, reconstructions = infer_windows(model, windows, batch_size)
        return features, spe(windows, reconstructions)


def infer_bank(model, bank, mean, scale, batch_size=INFERENCE_BATCH_SIZE):
    """Apply CAE inference replication by replication."""
    with single_threaded():
        evaluated = [
            infer_series(model, bank[b], mean, scale, batch_size)
            for b in range(bank.shape[0])
        ]
    return (
        np.stack([features for features, _ in evaluated]),
        np.stack([spe_values for _, spe_values in evaluated]),
    )


# One-Class SVM

OCSVM_PATH = PROJECT_ROOT / "models" / "ocsvm.npz"


class OCSVM(NamedTuple):

    support_vectors: np.ndarray
    dual_coefficients: np.ndarray
    rho: float
    gamma: float


def to_ocsvm_input(features):
    assert features.dtype == np.float32, features.dtype
    return features.astype(np.float64)


def exact_gamma(features):
    """Return inverse-median RBF gamma and median squared distance."""
    n_rows = features.shape[0]
    pairwise = np.empty(n_rows * (n_rows - 1) // 2, dtype=np.float64)
    position = 0
    for i in range(n_rows - 1):
        delta = features[i + 1 :] - features[i]
        block = (delta * delta).sum(axis=1)
        pairwise[position : position + block.size] = block
        position += block.size
    assert position == pairwise.size
    median = float(np.median(pairwise))
    assert median > 0.0 and np.isfinite(median)
    return 1.0 / median, median


def fit_ocsvm(features):
    """Fit the OC-SVM and return its frozen scoring state."""
    with single_threaded():
        gamma, _ = exact_gamma(features)
        estimator = OneClassSVM(
            kernel=KERNEL,
            nu=NU,
            gamma=gamma,
            tol=SVM_TOL,
            shrinking=SVM_SHRINKING,
            cache_size=SVM_CACHE_SIZE,
            max_iter=SVM_MAX_ITER,
            verbose=SVM_VERBOSE,
        )
        estimator.fit(features, sample_weight=None)

    assert estimator.fit_status_ == 0
    assert estimator.offset_.shape == (1,) and estimator.offset_.dtype == np.float64
    assert estimator.intercept_.shape == (1,) and estimator.intercept_.dtype == np.float64
    rho = estimator.offset_.item()
    assert rho == -estimator.intercept_.item()


    # Support vectors are copied fitting rows; preserve this bitwise check.
    assert np.array_equal(features[estimator.support_], estimator.support_vectors_)
    return OCSVM(
        support_vectors=estimator.support_vectors_,
        dual_coefficients=estimator.dual_coef_[0],
        rho=rho,
        gamma=gamma,
    )


def load_ocsvm(path=OCSVM_PATH):
    with np.load(path, allow_pickle=False) as archive:
        support_vectors = archive["support_vectors_"]
        dual_coefficients = archive["dual_coef_"]
        intercept = archive["intercept_"]
        gamma = archive["gamma"]
    assert intercept.shape == (1,) and intercept.dtype == np.float64
    assert dual_coefficients.ndim == 2 and dual_coefficients.shape[0] == 1
    assert support_vectors.shape[1] == FEATURE_DIM
    return OCSVM(
        support_vectors=support_vectors,
        dual_coefficients=dual_coefficients[0],
        rho=-intercept.item(),
        gamma=gamma.item(),
    )


def ocsvm_score(model, features):
    """Compute the explicit RBF anomaly score; larger is more abnormal."""
    with single_threaded():
        support = model.support_vectors
        squared = (
            (features * features).sum(axis=1)[:, None]
            + (support * support).sum(axis=1)[None, :]
            - 2.0 * (features @ support.T)
        )
        kernel = np.exp(-model.gamma * np.maximum(squared, 0.0))
        return model.rho - kernel @ model.dual_coefficients


# Monitoring vectors

class Frozen(NamedTuple):

    mean: np.ndarray
    scale: np.ndarray
    autoencoder: nn.Module
    ocsvm: OCSVM


def load_frozen():
    mean, scale = load_preprocessing()
    return Frozen(mean, scale, load_autoencoder(), load_ocsvm())


def monitoring_series(frozen, observations, batch_size=INFERENCE_BATCH_SIZE):
    """Build ``[OC-SVM score, log SPE]`` vectors for one series."""
    features, spe_values = infer_series(
        frozen.autoencoder, observations, frozen.mean, frozen.scale, batch_size
    )
    scores = ocsvm_score(frozen.ocsvm, to_ocsvm_input(features))
    return np.column_stack((scores, log_spe(spe_values)))


def monitoring_bank(frozen, bank, batch_size=INFERENCE_BATCH_SIZE):
    with single_threaded():
        return np.stack([
            monitoring_series(frozen, bank[b], batch_size)
            for b in range(bank.shape[0])
        ])


# In-control moments

class Moments(NamedTuple):

    mean: np.ndarray
    covariance: np.ndarray


def condition_number(covariance):
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert eigenvalues[0] > 0.0, eigenvalues
    return float(eigenvalues[-1] / eigenvalues[0])


def pooled_moments(values, ddof=MOMENTS_DDOF):
    """Estimate pooled mean and sample covariance over all rows."""
    assert values.dtype == np.float64, values.dtype


    # C order fixes the pooled reduction order.
    assert values.flags["C_CONTIGUOUS"]
    flat = values.reshape(-1, values.shape[-1])
    n, dimension = flat.shape
    divisor = n - ddof
    assert divisor > 0, (n, ddof)

    mean = flat.mean(axis=0)
    centered = flat - mean
    covariance = np.empty((dimension, dimension), dtype=np.float64)
    for row in range(dimension):
        for col in range(row, dimension):
            value = np.sum(centered[:, row] * centered[:, col]) / divisor
            covariance[row, col] = value
            covariance[col, row] = value

    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(covariance))
    assert np.array_equal(covariance, covariance.T)


    np.linalg.cholesky(covariance)
    assert condition_number(covariance) <= CONDITION_NUMBER_GUARD
    return Moments(mean=mean, covariance=covariance)


def benchmark_bank(bank):
    """Return raw observations aligned to the reporting horizon."""
    return np.ascontiguousarray(bank[:, WINDOW_LENGTH - 1 :, :])


# MEWMA and calibration

LIMITS_PATH = PROJECT_ROOT / "models" / "control_limits.json"
MEWMA_PATH = PROJECT_ROOT / "models" / "mewma.npz"


class CalibratedLimit(NamedTuple):

    chart_id: str
    lambda_value: float
    h_star: float
    achieved_arl0: float
    censor_count: int


def chart_plan():
    return (
        ("H", PRIMARY_LAMBDA),
        *(("H", value) for value in HYBRID_SENSITIVITY_LAMBDAS),
        ("S", PRIMARY_LAMBDA),
        ("R", PRIMARY_LAMBDA),
        ("M", BENCHMARK_LAMBDA),
    )


def limit_key(chart_id, lambda_value):
    return f"{chart_id}_{lambda_value:.2f}".replace(".", "p")


def benchmark_run_in(bank):
    """Return the benchmark's 31 silent warmup observations."""
    return np.ascontiguousarray(bank[:, : WINDOW_LENGTH - 1, :])


def finite_start_factors(n_windows, lambda_value, first_update_index=1):
    """Return exact finite-start MEWMA covariance factors."""
    j = np.arange(
        first_update_index, first_update_index + n_windows, dtype=np.float64
    )
    factors = (
        lambda_value
        / (2.0 - lambda_value)
        * (1.0 - (1.0 - lambda_value) ** (2.0 * j))
    )
    # Preserve c1 = lambda² exactly; the algebraic form rounds differently.
    if first_update_index == 1:
        factors[0] = lambda_value * lambda_value
    return factors


def mewma_states(values, mean, lambda_value, warmup=None):
    """Compute MEWMA state paths with optional silent warmup."""
    assert values.dtype == np.float64 and values.flags["C_CONTIGUOUS"]
    assert mean.shape == (values.shape[2],) and mean.dtype == np.float64
    states = np.empty_like(values)
    one_minus = 1.0 - lambda_value
    previous = np.broadcast_to(mean, (values.shape[0], values.shape[2])).copy()
    if warmup is not None:
        for index in range(warmup.shape[1]):
            previous = lambda_value * warmup[:, index] + one_minus * previous
    for window in range(values.shape[1]):
        previous = lambda_value * values[:, window] + one_minus * previous
        states[:, window] = previous
    return states


def mewma_t2(values, mean, covariance, lambda_value, warmup=None,
             first_update_index=1):
    """Compute MEWMA T-squared paths using the frozen numerical route."""
    dimension = values.shape[2]
    assert covariance.shape == (dimension, dimension)
    assert np.array_equal(covariance, covariance.T)
    assert condition_number(covariance) <= CONDITION_NUMBER_GUARD
    assert warmup is None or warmup.shape[1] == first_update_index - 1

    with single_threaded():
        cholesky = np.linalg.cholesky(covariance)
        states = mewma_states(values, mean, lambda_value, warmup)
        centered = (states - mean).reshape(-1, dimension)
        solved = np.linalg.solve(cholesky, centered.T).T
        mahalanobis = np.sum(solved * solved, axis=1).reshape(values.shape[:2])
        factors = finite_start_factors(
            values.shape[1], lambda_value, first_update_index
        )
        t2 = mahalanobis / factors[np.newaxis, :]


    assert np.all(np.isfinite(t2)) and np.all(t2 >= 0.0)
    return t2


def run_lengths(running_max, h):
    """Return first strict crossings and explicit censoring flags."""
    # Signalling is strict; a final-index crossing is distinct from censoring.
    exceeded = running_max > h
    censored = ~exceeded[:, -1]
    lengths = np.where(censored, running_max.shape[1], exceeded.argmax(axis=1) + 1)
    return lengths, censored


def calibrate_limit(t2, target_arl0=TARGET_ARL0):
    """Find the smallest attained limit reaching the target ARL0."""
    with single_threaded():
        running_max = np.maximum.accumulate(t2, axis=1)
        candidates = np.unique(running_max)

        def arl0(index):
            return float(run_lengths(running_max, candidates[index])[0].mean())

        low, high, selected = 0, candidates.size - 1, None
        while low <= high:
            middle = (low + high) // 2
            if arl0(middle) >= target_arl0:
                selected, high = middle, middle - 1
            else:
                low = middle + 1
        assert selected is not None, "no attained value reaches the target ARL0"
        assert selected == 0 or arl0(selected - 1) < target_arl0

        lengths, censored = run_lengths(running_max, candidates[selected])
    return float(candidates[selected]), float(lengths.mean()), int(censored.sum())


def calibrate_charts(y, x_m, run_in, hybrid, benchmark):
    """Calibrate all six chart limits on D3."""
    limits = {}
    for chart_id, lambda_value in chart_plan():
        chart = CHARTS[chart_id]
        components = list(chart["components"])
        if chart["source"] == "hybrid_monitoring_vector":


            values = np.ascontiguousarray(y[..., components])
            mean = np.ascontiguousarray(hybrid.mean[components])
            covariance = np.ascontiguousarray(
                hybrid.covariance[np.ix_(components, components)]
            )
            warmup = None
        else:
            values, mean, covariance = x_m, benchmark.mean, benchmark.covariance
            warmup = run_in
        t2 = mewma_t2(
            values, mean, covariance, lambda_value, warmup,
            chart["update_index_offset"] + 1,
        )
        h_star, achieved, censor_count = calibrate_limit(t2)
        limits[limit_key(chart_id, lambda_value)] = CalibratedLimit(
            chart_id=chart_id,
            lambda_value=lambda_value,
            h_star=h_star,
            achieved_arl0=achieved,
            censor_count=censor_count,
        )
    return limits


def load_moments(path=MEWMA_PATH):
    with np.load(path, allow_pickle=False) as archive:
        return (
            Moments(mean=archive["mu_y"], covariance=archive["sigma_y"]),
            Moments(mean=archive["mu_x"], covariance=archive["sigma_x"]),
        )


def load_control_limits(path=LIMITS_PATH):
    return {
        limit_key(limit["chart_id"], limit["lambda"]): limit["h_star"]
        for limit in json.loads(path.read_text())
    }


# Phase II evaluation

PHASE2_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PHASE2_DIR / "cache"

CACHE_INPUTS = (
    Path(__file__).resolve(),
    PREPROCESSING_PATH,
    AUTOENCODER_PATH,
    OCSVM_PATH,
    MEWMA_PATH,
    LIMITS_PATH,
)

D5_INTERPRETATION = {
    "change_point_tau": CHANGE_POINT,
    "run_length_units": "monitoring-opportunity j units",
    "t_signal_relation": "t_signal = RL + 31",
    "first_monitoring_opportunity_original_time_index": WINDOW_LENGTH,
    "primary_contrast_definition": "RL_H - RL_M",
    "primary_contrast_direction": (
        "negative mean_difference means H(0.05) signals faster than M(0.05)"
    ),
    "qualification": (
        "The sustained latent mean shift begins at tau=1, the first retained "
        "observation. Therefore x_1 through x_31 used before the first "
        "permitted monitoring decision are already post-change; the first "
        "monitoring opportunity is j=1 at original time t=32."
    ),
}


def cache_token():
    digest = hashlib.sha256()
    for path in CACHE_INPUTS:
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def achieved_arl0_by_chart(path=LIMITS_PATH):
    return {
        limit_key(limit["chart_id"], limit["lambda"]): limit["achieved_arl0"]
        for limit in json.loads(path.read_text())
    }


def evaluate_bank(frozen, hybrid, benchmark, limits, bank):
    """Evaluate all six frozen charts on one bank."""
    assert bank.ndim == 3 and bank.shape[1:] == (D4_HORIZON, OBS_DIM)
    y = monitoring_bank(frozen, bank)
    x_m = benchmark_bank(bank)
    run_in = benchmark_run_in(bank)

    cells = {}
    for chart_id, lambda_value in chart_plan():
        chart = CHARTS[chart_id]
        components = list(chart["components"])
        if chart["source"] == "hybrid_monitoring_vector":
            values = np.ascontiguousarray(y[..., components])
            mean = np.ascontiguousarray(hybrid.mean[components])
            covariance = np.ascontiguousarray(
                hybrid.covariance[np.ix_(components, components)]
            )
            warmup = None
        else:
            values, mean, covariance = x_m, benchmark.mean, benchmark.covariance
            warmup = run_in
        t2 = mewma_t2(
            values, mean, covariance, lambda_value, warmup,
            chart["update_index_offset"] + 1,
        )
        key = limit_key(chart_id, lambda_value)
        lengths, censored = run_lengths(
            np.maximum.accumulate(t2, axis=1), limits[key]
        )
        lengths = lengths.astype(np.int64)
        assert np.all(lengths >= 1) and np.all(lengths <= CALIBRATION_HORIZON)
        assert np.all(lengths[censored] == CALIBRATION_HORIZON)
        cells[key] = (lengths, censored)
    return cells


def cached_bank_cells(name, compute, directory=CACHE_DIR):
    """Load a valid cached result or compute and cache it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.npz"
    token = cache_token()
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["token"]) == token:
                return {
                    key: (archive[f"{key}/run_lengths"], archive[f"{key}/censored"])
                    for key in (limit_key(c, v) for c, v in chart_plan())
                }
    cells = compute()
    np.savez(
        path,
        token=np.array(token),
        **{
            f"{key}/{field}": array
            for key, pair in cells.items()
            for field, array in zip(("run_lengths", "censored"), pair)
        },
    )
    return cells


def cell_summary(lengths, censored):
    """Summarize one run-length cell."""
    values = lengths.astype(np.float64)
    n = values.size
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    censor_count = int(censored.sum())
    censor_fraction = censor_count / n
    return {
        "se_empirical": sd / np.sqrt(n),
        "sd_run_length": sd,
        "censor_count": censor_count,
        "censor_fraction": censor_fraction,


        "truncation_limited": censor_fraction > TRUNCATION_FRACTION_THRESHOLD,
        "run_length_median": float(np.median(values)),
        "n_replications": n,
        "n_windows": CALIBRATION_HORIZON,
    }


def d4_entry(chart_id, lambda_value, lengths, censored, achieved_arl0):
    values = lengths.astype(np.float64)
    arl0 = float(values.mean())
    return {
        "chart_id": chart_id,
        "lambda": float(lambda_value),
        "arl0_validation_h": arl0,
        **cell_summary(lengths, censored),
        "run_length_min": float(lengths.min()),
        "run_length_max": float(lengths.max()),
        "target_arl0": TARGET_ARL0,
        "deviation_from_target": arl0 - TARGET_ARL0,
        "ratio_to_target": arl0 / TARGET_ARL0,
        "stage5_d3_achieved_arl0": achieved_arl0,
    }


def d5_entry(chart_id, lambda_value, delta, lengths, censored):
    return {
        "chart_id": chart_id,
        "lambda": float(lambda_value),
        "delta": float(delta),
        "arl1_h": float(lengths.astype(np.float64).mean()),
        **cell_summary(lengths, censored),
    }


def paired_contrast(h_lengths, m_lengths):
    """Compute paired H(.05) - M(.05) run-length differences."""
    diff = h_lengths.astype(np.float64) - m_lengths.astype(np.float64)
    n = diff.size
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    return {
        "mean_difference": float(diff.mean()),
        "se_paired": sd / np.sqrt(n),
        "n_replications": n,
    }


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def evaluate_phase2(data_dir, out_dir=PHASE2_DIR):
    """Evaluate D4 and all D5 scenarios and write final results."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    frozen = load_frozen()
    hybrid, benchmark = load_moments()
    limits = load_control_limits()
    achieved = achieved_arl0_by_chart()
    keys = [limit_key(chart, value) for chart, value in chart_plan()]

    def bank_cells(name, path):
        def compute():
            with np.load(path, allow_pickle=False) as archive:
                return evaluate_bank(
                    frozen, hybrid, benchmark, limits, archive["x"]
                )
        return cached_bank_cells(name, compute)

    d4 = bank_cells("D4", data_dir / "D4.npz")
    d5 = {
        scenario_name(delta): bank_cells(
            scenario_name(delta), data_dir / "D5" / f"{scenario_name(delta)}.npz"
        )
        for delta in SHIFT_GRID
    }

    arrays = {
        f"d4/{key}/{field}": array
        for key in keys
        for field, array in zip(("run_lengths", "censored"), d4[key])
    }
    arrays.update({
        f"d5/{key}/{stem}/{field}": array
        for key in keys
        for stem in d5
        for field, array in zip(("run_lengths", "censored"), d5[stem][key])
    })

    d4_summary = {
        "charts": [
            d4_entry(*chart, *d4[key], achieved[key])
            for chart, key in zip(chart_plan(), keys)
        ]
    }
    d5_summary = {
        "interpretation": D5_INTERPRETATION,
        "cells": [
            d5_entry(*chart, delta, *d5[scenario_name(delta)][key])
            for delta in SHIFT_GRID
            for chart, key in zip(chart_plan(), keys)
        ],
        "primary_contrast": [
            {
                "delta": float(delta),
                **paired_contrast(
                    d5[scenario_name(delta)]["H_0p05"][0],
                    d5[scenario_name(delta)]["M_0p05"][0],
                ),
            }
            for delta in SHIFT_GRID
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "run_lengths.npz", **arrays)
    write_json(out_dir / "d4_validation_summary.json", d4_summary)
    write_json(out_dir / "d5_performance_summary.json", d5_summary)
    return d4_summary, d5_summary
