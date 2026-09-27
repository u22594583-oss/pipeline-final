from pathlib import Path
import os

import numpy as np
import torch

import pipeline as P


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
DATA = DATA_ROOT / "generated"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def require_clean_start():
    """Refuse to overwrite an existing production run."""
    for path in (DATA_ROOT, MODELS, RESULTS):
        if path.exists() or path.is_symlink():
            raise RuntimeError(
                f"Refusing to overwrite existing production output: {path}"
            )


def save_control_limits(limits):
    """Write the canonical control-limit artifact."""
    rows = []

    for chart_id, lambda_value in P.chart_plan():
        key = P.limit_key(chart_id, lambda_value)
        limit = limits[key]
        chart = P.CHARTS[chart_id]

        censor_fraction = limit.censor_count / P.D3_REPLICATIONS

        if chart_id == "H" and lambda_value != P.PRIMARY_LAMBDA:
            role = "hybrid_sensitivity"
        else:
            role = chart["role"]

        rows.append(
            {
                "achieved_arl0": limit.achieved_arl0,
                "censor_count": limit.censor_count,
                "censor_fraction": censor_fraction,
                "chart_id": chart_id,
                "components": list(chart["components"]),
                "dimension": chart["dimension"],
                "h_star": limit.h_star,
                "lambda": float(lambda_value),
                "n_replications": P.D3_REPLICATIONS,
                "n_windows": P.CALIBRATION_HORIZON,
                "role": role,
                "signal_rule": "T2 > h",
                "source": chart["source"],
                "target_arl0": P.TARGET_ARL0,
                "truncation_limited": (
                    censor_fraction > P.TRUNCATION_FRACTION_THRESHOLD
                ),
                "update_index_offset": chart["update_index_offset"],
            }
        )

    P.write_json(MODELS / "control_limits.json", rows)


def main():
    os.chdir(ROOT)
    require_clean_start()

    DATA.mkdir(parents=True)
    MODELS.mkdir()
    RESULTS.mkdir()

    print("1/6 Generating D1-D5...", flush=True)
    P.generate_datasets(DATA)

    print("2/6 Fitting preprocessing and training CAE...", flush=True)
    with np.load(DATA / "D1.npz", allow_pickle=False) as archive:
        d1 = archive["x"]

    train, validation = P.split_d1(d1)
    mean, scale = P.fit_preprocessing(train)

    train_windows = P.window_series(
        P.apply_preprocessing(train, mean, scale)
    )
    validation_windows = P.window_series(
        P.apply_preprocessing(validation, mean, scale)
    )

    autoencoder, history = P.train_autoencoder(
        train_windows,
        validation_windows,
    )

    np.savez(
        MODELS / "preprocessing.npz",
        mean=mean,
        scale=scale,
    )
    torch.save(
        autoencoder.state_dict(),
        MODELS / "autoencoder.pt",
    )

    print(
        f"    best epoch={history['best_epoch']}, "
        f"validation loss={history['best_validation_loss']}",
        flush=True,
    )

    print("3/6 Fitting OC-SVM...", flush=True)
    with np.load(DATA / "D2.npz", allow_pickle=False) as archive:
        d2 = archive["x"]

    features, _ = P.infer_series(
        autoencoder,
        d2,
        mean,
        scale,
    )
    ocsvm = P.fit_ocsvm(
        P.to_ocsvm_input(features)
    )

    np.savez(
        MODELS / "ocsvm.npz",
        support_vectors_=ocsvm.support_vectors,
        dual_coef_=ocsvm.dual_coefficients[None, :],
        intercept_=np.array([-ocsvm.rho], dtype=np.float64),
        gamma=np.array(ocsvm.gamma, dtype=np.float64),
    )

    print(
        f"    gamma={ocsvm.gamma}, "
        f"rho={ocsvm.rho}, "
        f"support vectors={ocsvm.support_vectors.shape[0]}",
        flush=True,
    )

    print("4/6 Estimating D3 moments...", flush=True)
    frozen = P.Frozen(
        mean=mean,
        scale=scale,
        autoencoder=autoencoder,
        ocsvm=ocsvm,
    )

    with np.load(DATA / "D3.npz", allow_pickle=False) as archive:
        d3 = archive["x"]

    y = P.monitoring_bank(frozen, d3)
    x_m = P.benchmark_bank(d3)
    run_in = P.benchmark_run_in(d3)

    hybrid = P.pooled_moments(y)
    benchmark = P.pooled_moments(x_m)

    np.savez(
        MODELS / "mewma.npz",
        mu_y=hybrid.mean,
        sigma_y=hybrid.covariance,
        mu_x=benchmark.mean,
        sigma_x=benchmark.covariance,
    )

    print("5/6 Calibrating control limits...", flush=True)
    limits = P.calibrate_charts(
        y,
        x_m,
        run_in,
        hybrid,
        benchmark,
    )

    save_control_limits(limits)

    for key, value in limits.items():
        print(
            f"    {key}: "
            f"h*={value.h_star}, "
            f"ARL0={value.achieved_arl0}, "
            f"censored={value.censor_count}",
            flush=True,
        )

    # Release the large D3 arrays before Phase II.
    del d3, y, x_m, run_in

    print("6/6 Running D4/D5 Phase II evaluation...", flush=True)
    P.evaluate_phase2(
        DATA,
        out_dir=RESULTS,
    )

    print()
    print("FULL PRODUCTION RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
