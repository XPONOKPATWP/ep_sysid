"""
training_pipeline.py

Training, cross-validation, inference, and model comparison pipelines
for ARMAX, MLP, and SVR system identification.
Supports SISO, MISO, SIMO, and MIMO multivariable architectures.
"""

import json
import pickle
import os
import re
import time
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from mlp_simulator import simulate_mlp_system, simulate_mlp_free_run, get_mlp_parameter_count
from armax_simulator import simulate_armax_system, simulate_armax_free_run, get_armax_parameter_count
from svr_simulator import simulate_svr_system, simulate_svr_free_run, get_svr_parameter_count

from multivariable_data_parser import (
    load_cached_uci_benchmarks,
    get_dataset_summary_info,
    slice_signals_by_system_mode
)
from ep_algorithm_core import execute_evolutionary_programming, _estimate_legacy_checkpoint_time
from validation_metrics import generate_validation_report, compute_deep_single_model_diagnostics
from config_handler import load_hyperparameters, get_parameter_label, atomic_replace_retry
from diagnostic_plotter import (
    generate_convergence_plot,
    generate_prediction_tracking_plot,
    generate_residual_diagnostics_plot,
    generate_single_model_deep_diagnostics_plots,
    generate_cross_validation_summary,
    generate_multi_model_comparison_plots,
    generate_simulation_vs_prediction_plot,
    generate_bode_diagram
)


def enforce_reproducibility_seed(seed_value=42):
    # Fix random state so 10-fold CV splits and mutations are reproducible
    np.random.seed(seed_value)
    random.seed(seed_value)


def _fit_scaler_nd(signal_array):
    # sklearn StandardScaler requires 2D shape (n_samples, n_features)
    arr = np.asarray(signal_array, dtype=float)
    mat = arr.reshape(-1, 1) if arr.ndim == 1 else arr
    return StandardScaler().fit(mat)


def _transform_nd(scaler, signal_array):
    arr = np.asarray(signal_array, dtype=float)
    was_1d = (arr.ndim == 1)
    mat = arr.reshape(-1, 1) if was_1d else arr
    scaled = scaler.transform(mat)
    return scaled.flatten() if (was_1d and scaled.shape[1] == 1) else scaled


def _inverse_transform_nd(scaler, scaled_array):
    arr = np.asarray(scaled_array, dtype=float)
    was_1d = (arr.ndim == 1)
    mat = arr.reshape(-1, 1) if was_1d else arr
    unscaled = scaler.inverse_transform(mat)
    return unscaled.flatten() if (was_1d and unscaled.shape[1] == 1) else unscaled


def _build_parameter_names(model_type, na, nb, nc=2, hidden_units=None, n_u=1, n_y=1):
    # Builds readable parameter names for SISO, MISO, SIMO, and MIMO models
    if hidden_units is None:
        hidden_units = [5]
    if model_type == "ARMAX":
        names = []
        for out_ch in range(n_y):
            y_prefix = f"y{out_ch + 1}_" if n_y > 1 else ""
            names.extend([f"{y_prefix}a{i + 1}" for i in range(na)])
            for in_ch in range(n_u):
                u_suffix = f"_u{in_ch + 1}" if n_u > 1 else ""
                names.extend([f"{y_prefix}b{j + 1}{u_suffix}" for j in range(nb)])
            names.extend([f"{y_prefix}c{k + 1}" for k in range(nc)])
        return names
    elif model_type == "SVR":
        return ["C", "epsilon", "gamma"]
    else:
        total_params = get_mlp_parameter_count(na, nb, hidden_units, n_u=n_u, n_y=n_y)
        return [f"w_{i + 1}" for i in range(total_params)]


def _format_model_architecture_and_filter(model_type, config_dict, n_u=1, n_y=1):
    # Builds architecture summary string and filters out irrelevant parameters for reports
    system_mode = str(config_dict.get("system_mode", "SISO")).upper()
    na = int(config_dict.get("na", 2))
    nb = int(config_dict.get("nb", 2))
    nk = int(config_dict.get("nk", 0))
    io_tag = f"{system_mode} ({n_u}In->{n_y}Out)"

    excluded_keys = {"checkpoint_file", "mutation_mu", "n_u", "n_y"}
    if system_mode in ("MISO", "MIMO"):
        excluded_keys.add("input_col_idx")
    if system_mode in ("SIMO", "MIMO"):
        excluded_keys.add("output_col_idx")

    if model_type == "ARMAX":
        nc = int(config_dict.get("nc", 2))
        arch_summary = f"model=ARMAX [{io_tag}], na={na}, nb={nb}, nc={nc}, nk={nk}"
        excluded_keys.add("mlp_hidden_nodes")
    elif model_type == "SVR":
        arch_summary = f"model=SVR [{io_tag}], na={na}, nb={nb}, nk={nk} (RBF Kernel: [C, epsilon, gamma])"
        excluded_keys.update(["nc", "mlp_hidden_nodes"])
    else:
        hidden_units = config_dict.get("mlp_hidden_nodes", [5])
        arch_summary = f"model=MLP [{io_tag}], na={na}, nb={nb}, nk={nk}, hidden_nodes={hidden_units}"
        excluded_keys.add("nc")

    filtered_config = {k: v for k, v in config_dict.items() if k not in excluded_keys}
    return arch_summary, filtered_config


def _build_model_artifact_basename(output_dir, model_type, dataset_name):
    # Formats the model filename as <ModelType>_<Dataset>_<UUID>_model
    folder_name = os.path.basename(os.path.normpath(output_dir))
    parts = folder_name.split("_")
    if len(parts) >= 2 and re.fullmatch(r"[0-9a-fA-F]{6}", parts[-1]):
        unique_id = parts[-1]
        return f"{model_type}_{dataset_name}_{unique_id}_model"
    return f"{model_type}_{dataset_name}_model"


def _parse_serialized_model(model_filepath):
    # Loads a saved model (.pkl or .json) and reconstructs its scalers and parameters
    file_extension = os.path.splitext(model_filepath)[1].lower()

    if file_extension == ".json":
        with open(model_filepath, "r", encoding="utf-8", errors="replace") as file:
            parsed_data = json.load(file)

        required_keys = {"structure", "best_fold_parameters"}
        if not isinstance(parsed_data, dict) or not required_keys.issubset(parsed_data.keys()):
            raise ValueError("Invalid JSON model file. Please select a *_model.json file.")

        architecture = parsed_data["structure"]
        model_type = architecture.get("model_type", "ARMAX")
        na, nb = int(architecture["na"]), int(architecture["nb"])
        nc = int(architecture.get("nc", 2))
        hidden_units = architecture.get("mlp_hidden_nodes", [5])
        n_u = int(architecture.get("n_u", 1))
        n_y = int(architecture.get("n_y", 1))

        best_weights_dict = parsed_data["best_fold_parameters"]
        mean_weights_dict = parsed_data.get("mean_10fold_parameters", best_weights_dict)

        parameter_names = list(best_weights_dict.keys())
        if not parameter_names:
            parameter_names = _build_parameter_names(model_type, na, nb, nc, hidden_units, n_u=n_u, n_y=n_y)

        best_theta_vector = [float(best_weights_dict[k]) for k in parameter_names]
        mean_theta_vector = [float(mean_weights_dict.get(k, best_weights_dict[k])) for k in parameter_names]

        u_scaler, y_scaler = None, None
        if "scaler_u" in parsed_data and "scaler_y" in parsed_data:
            u_mean = np.atleast_1d(np.asarray(parsed_data["scaler_u"]["mean"], dtype=float))
            u_std = np.atleast_1d(np.asarray(parsed_data["scaler_u"]["std"], dtype=float))
            u_std = np.where(u_std <= 1e-12, 1.0, u_std)
            u_scaler = StandardScaler()
            u_scaler.mean_ = u_mean
            u_scaler.scale_ = u_std
            u_scaler.var_ = u_std ** 2
            u_scaler.n_features_in_ = len(u_mean)

            y_mean = np.atleast_1d(np.asarray(parsed_data["scaler_y"]["mean"], dtype=float))
            y_std = np.atleast_1d(np.asarray(parsed_data["scaler_y"]["std"], dtype=float))
            y_std = np.where(y_std <= 1e-12, 1.0, y_std)
            y_scaler = StandardScaler()
            y_scaler.mean_ = y_mean
            y_scaler.scale_ = y_std
            y_scaler.var_ = y_std ** 2
            y_scaler.n_features_in_ = len(y_mean)

        svr_u_train = np.asarray(parsed_data["svr_u_train"], dtype=float) if parsed_data.get("svr_u_train") is not None else None
        svr_y_train = np.asarray(parsed_data["svr_y_train"], dtype=float) if parsed_data.get("svr_y_train") is not None else None

        return {
            "dataset": parsed_data.get("dataset", "Unknown_Source"),
            "structure": architecture,
            "param_names": parameter_names,
            "best_fold_index": int(parsed_data.get("best_fold", 1)),
            "best_fold_theta": best_theta_vector,
            "mean_theta_10fold": mean_theta_vector,
            "scaler_u": u_scaler,
            "scaler_y": y_scaler,
            "svr_u_train": svr_u_train,
            "svr_y_train": svr_y_train
        }

    else:
        with open(model_filepath, "rb") as file:
            model_payload = pickle.load(file)

        if isinstance(model_payload, dict) and "population" in model_payload and "generation" in model_payload:
            raise ValueError("Selected file is a training checkpoint, not a finished model. Please select a '*_model.pkl' file.")

        required_keys = {"structure", "best_fold_theta", "scaler_u", "scaler_y"}
        if not isinstance(model_payload, dict) or not required_keys.issubset(model_payload.keys()):
            raise ValueError("Invalid PKL model format.")
        return model_payload


def execute_inference_pipeline(model_filepath, u_raw, y_raw, target_dataset_name, output_dir="results_inference", progress_callback=None):
    # Evaluates a saved model on the target dataset, saves predictions.csv, and generates the plots and report
    execution_start = time.perf_counter()

    if progress_callback:
        progress_callback(f"Loading model from {os.path.basename(model_filepath)}...")

    model_payload = _parse_serialized_model(model_filepath)
    os.makedirs(output_dir, exist_ok=True)

    architecture = model_payload["structure"]
    model_type = architecture.get("model_type", "ARMAX")
    system_mode = architecture.get("system_mode", "SISO")
    input_col_idx = int(architecture.get("input_col_idx", 0))
    output_col_idx = int(architecture.get("output_col_idx", 0))
    na, nb, nk = int(architecture["na"]), int(architecture["nb"]), int(architecture["nk"])
    hidden_units = architecture.get("mlp_hidden_nodes", [5])
    nc = int(architecture.get("nc", 2))
    parameter_names = model_payload.get("param_names", [])

    # Slice target dataset to match the model's saved system_mode and channel indices
    u_signal, y_signal = slice_signals_by_system_mode(
        u_raw, y_raw, system_mode=system_mode, input_col_idx=input_col_idx, output_col_idx=output_col_idx
    )
    n_u = 1 if u_signal.ndim == 1 else u_signal.shape[1]
    n_y = 1 if y_signal.ndim == 1 else y_signal.shape[1]

    source_dataset = model_payload.get("dataset", "Unknown_Source")
    winning_fold_index = int(model_payload.get("best_fold_index", 1))
    best_theta_weights = np.asarray(model_payload["best_fold_theta"], dtype=float)

    is_cross_domain = (source_dataset != target_dataset_name)
    u_scaler = model_payload.get("scaler_u")
    y_scaler = model_payload.get("scaler_y")

    # Use saved training scalers for same-dataset Source Checks unless channel counts differ
    scaler_u_mismatch = (u_scaler is None) or (getattr(u_scaler, "n_features_in_", 1) != n_u)
    scaler_y_mismatch = (y_scaler is None) or (getattr(y_scaler, "n_features_in_", 1) != n_y)
    if scaler_u_mismatch or scaler_y_mismatch or is_cross_domain:
        u_scaler = _fit_scaler_nd(u_signal)
        y_scaler = _fit_scaler_nd(y_signal)

    if progress_callback:
        mode_str = "Cross-Dataset Inference" if is_cross_domain else "Source Dataset Check"
        progress_callback(f"Running {mode_str} ({model_type}-{system_mode}) on {target_dataset_name}...")

    u_scaled = _transform_nd(u_scaler, u_signal)
    y_scaled = _transform_nd(y_scaler, y_signal)

    arch_summary, _ = _format_model_architecture_and_filter(model_type, architecture, n_u=n_u, n_y=n_y)

    simulated_y_scaled = None
    if model_type == "ARMAX":
        total_parameters = get_armax_parameter_count(na, nb, nc, n_u=n_u, n_y=n_y)
        predicted_y_scaled, max_delay_lag = simulate_armax_system(
            u_scaled, y_scaled, best_theta_weights, na, nb, nc, nk
        )
        simulated_y_scaled, _ = simulate_armax_free_run(
            u_scaled, y_scaled, best_theta_weights, na, nb, nc, nk
        )
    elif model_type == "SVR":
        total_parameters = get_svr_parameter_count(na, nb, n_u=n_u, n_y=n_y)
        svr_u_train = model_payload.get("svr_u_train")
        svr_y_train = model_payload.get("svr_y_train")
        predicted_y_scaled, max_delay_lag = simulate_svr_system(
            u_scaled, y_scaled, best_theta_weights, na, nb, nk,
            u_train=svr_u_train, y_train=svr_y_train, cross_domain=is_cross_domain
        )
        simulated_y_scaled, _ = simulate_svr_free_run(
            u_scaled, y_scaled, best_theta_weights, na, nb, nk,
            u_train=svr_u_train, y_train=svr_y_train, cross_domain=is_cross_domain
        )
    else:
        total_parameters = get_mlp_parameter_count(na, nb, hidden_units, n_u=n_u, n_y=n_y)
        # When testing an MLP on a different dataset, clip extreme z-scores to [-3, 3] so out-of-domain
        # spikes don't saturate the tanh hidden units, then re-center the output before inverse scaling.
        if is_cross_domain:
            u_in = np.clip(u_scaled, -3.0, 3.0)
            y_in = np.clip(y_scaled, -3.0, 3.0)
        else:
            u_in, y_in = u_scaled, y_scaled
        predicted_y_scaled, max_delay_lag = simulate_mlp_system(
            u_in, y_in, best_theta_weights, na, nb, nk, hidden_units
        )
        simulated_y_scaled, _ = simulate_mlp_free_run(
            u_in, y_in, best_theta_weights, na, nb, nk, hidden_units
        )
        if is_cross_domain:
            pred_valid = predicted_y_scaled[max_delay_lag:]
            pred_std = np.std(pred_valid, axis=0, keepdims=True)
            pred_mean = np.mean(pred_valid, axis=0, keepdims=True)
            predicted_y_scaled[max_delay_lag:] = np.where(pred_std > 1e-6, (pred_valid - pred_mean) / np.maximum(pred_std, 1e-6), pred_valid - pred_mean)
            if simulated_y_scaled is not None:
                sim_valid = simulated_y_scaled[max_delay_lag:]
                if np.isfinite(sim_valid).all():
                    sim_std = np.std(sim_valid, axis=0, keepdims=True)
                    sim_mean = np.mean(sim_valid, axis=0, keepdims=True)
                    simulated_y_scaled[max_delay_lag:] = np.where(sim_std > 1e-6, (sim_valid - sim_mean) / np.maximum(sim_std, 1e-6), sim_valid - sim_mean)

    if not np.isfinite(predicted_y_scaled).all():
        raise RuntimeError("Model produced unstable predictions on this dataset.")

    u_aligned_unscaled = u_signal[max_delay_lag:]
    y_actual_unscaled = _inverse_transform_nd(y_scaler, y_scaled[max_delay_lag:])
    y_pred_unscaled = _inverse_transform_nd(y_scaler, predicted_y_scaled[max_delay_lag:])
    y_sim_unscaled = _inverse_transform_nd(y_scaler, simulated_y_scaled[max_delay_lag:]) if simulated_y_scaled is not None else None

    best_validation_scores = generate_validation_report(
        y_actual_unscaled, y_pred_unscaled, model_parameter_count=total_parameters,
        simulated_outputs=y_sim_unscaled
    )
    deep_diag = compute_deep_single_model_diagnostics(
        u_aligned_unscaled, y_actual_unscaled, y_pred_unscaled,
        model_type=model_type, theta_weights=best_theta_weights, na=na, nb=nb, nc=nc
    )

    chart_suffix = f" ({model_type} {'Cross-Domain Inference' if is_cross_domain else 'Source Check'})"
    generate_prediction_tracking_plot(
        y_actual_unscaled, y_pred_unscaled, target_dataset_name, winning_fold_index,
        output_dir=output_dir, title_suffix=chart_suffix
    )
    generate_residual_diagnostics_plot(
        y_actual_unscaled, y_pred_unscaled, target_dataset_name, winning_fold_index,
        output_dir=output_dir, title_suffix=chart_suffix
    )
    generate_single_model_deep_diagnostics_plots(
        u_aligned_unscaled, y_actual_unscaled, y_pred_unscaled, target_dataset_name, winning_fold_index,
        output_dir=output_dir, title_suffix=chart_suffix
    )
    if y_sim_unscaled is not None:
        generate_simulation_vs_prediction_plot(
            y_actual_unscaled, y_pred_unscaled, y_sim_unscaled,
            target_dataset_name, winning_fold_index, output_dir=output_dir, title_suffix=chart_suffix
        )
    if model_type == "ARMAX":
        generate_bode_diagram(
            best_theta_weights, na, nb, nc, nk,
            target_dataset_name, winning_fold_index, output_dir=output_dir, title_suffix=chart_suffix,
            n_u=n_u, n_y=n_y
        )

    # Export predictions.csv supporting both 1D and 2D multivariable signals
    u_2d = u_aligned_unscaled.reshape(-1, 1) if u_aligned_unscaled.ndim == 1 else u_aligned_unscaled
    y_act_2d = y_actual_unscaled.reshape(-1, 1) if y_actual_unscaled.ndim == 1 else y_actual_unscaled
    y_prd_2d = y_pred_unscaled.reshape(-1, 1) if y_pred_unscaled.ndim == 1 else y_pred_unscaled
    res_2d = y_act_2d - y_prd_2d

    time_indices = np.arange(max_delay_lag, len(y_signal)).reshape(-1, 1)
    csv_data = np.hstack((time_indices, u_2d, y_act_2d, y_prd_2d, res_2d))
    u_headers = [f"Input_u{i+1}" for i in range(u_2d.shape[1])] if u_2d.shape[1] > 1 else ["Input_u"]
    y_headers = [f"Actual_y{i+1}" for i in range(y_act_2d.shape[1])] if y_act_2d.shape[1] > 1 else ["Actual_y"]
    yh_headers = [f"Predicted_y_hat{i+1}" for i in range(y_prd_2d.shape[1])] if y_prd_2d.shape[1] > 1 else ["Predicted_y_hat"]
    e_headers = [f"Error_e{i+1}" for i in range(res_2d.shape[1])] if res_2d.shape[1] > 1 else ["Error"]

    csv_header = ",".join(["Time_Step"] + u_headers + y_headers + yh_headers + e_headers)
    csv_fmt = ["%d"] + ["%.6f"] * (csv_data.shape[1] - 1)
    csv_filepath = os.path.join(output_dir, "predictions.csv")
    np.savetxt(csv_filepath, csv_data, delimiter=",", header=csv_header, comments="", fmt=csv_fmt)

    # Primary output channel extrema analysis (channel 0 for SIMO/MIMO)
    y_primary = y_act_2d[:, 0]
    yhat_primary = y_prd_2d[:, 0]
    max_idx = int(np.argmax(y_primary))
    min_idx = int(np.argmin(y_primary))
    true_max_val = float(y_primary[max_idx])
    true_min_val = float(y_primary[min_idx])
    pred_at_max = float(yhat_primary[max_idx])
    pred_at_min = float(yhat_primary[min_idx])
    win_max = float(np.max(yhat_primary[max(0, max_idx - 20): min(len(yhat_primary), max_idx + 21)]))
    win_min = float(np.min(yhat_primary[max(0, min_idx - 20): min(len(yhat_primary), min_idx + 21)]))

    execution_elapsed = time.perf_counter() - execution_start

    report_filepath = os.path.join(output_dir, "inference_report.txt")
    with open(report_filepath, "w", encoding="utf-8") as file:
        file.write("=== Single-Model Deep Evaluation & Diagnostic Report ===\n")
        file.write(f"Evaluation Mode      : {'Cross-Dataset Inference (Domain Adapted)' if is_cross_domain else 'Source Dataset Check (Native Scalers)'}\n")
        file.write(f"Model File           : {os.path.abspath(model_filepath)}\n")
        file.write(f"Model Type           : {model_type} ({system_mode})\n")
        file.write(f"Trained On           : {source_dataset} (Fold {winning_fold_index:02d})\n")
        file.write(f"Evaluated On         : {target_dataset_name} (N = {len(y_actual_unscaled)} aligned samples, Burn-in Lag = {max_delay_lag})\n")
        file.write(f"Architecture         : {arch_summary}\n")
        file.write(f"Time Taken           : {execution_elapsed:.4f} seconds\n")

        file.write("\n--- 1. Best-Fold Identified Parameters ---\n")
        for name, val in zip(parameter_names, best_theta_weights):
            file.write(f"{name}: {val:>8.4f}\n")

        file.write("\n--- 2. Standard Validation Metrics ---\n")
        file.write(f"MSE                  : {best_validation_scores['MSE']:.4f}\n")
        file.write(f"RMSE                 : {best_validation_scores['RMSE']:.4f}\n")
        file.write(f"MAPE                 : {best_validation_scores['MAPE']:.2f}%\n")
        file.write(f"R-Squared (R²)       : {best_validation_scores['R-Squared']:.4f}\n")
        file.write(f"AIC                  : {best_validation_scores['AIC']:.4f}\n")
        if "AICc" in best_validation_scores:
            file.write(f"AICc (Small-Sample)  : {best_validation_scores['AICc']:.4f}\n")
        file.write(f"BIC                  : {best_validation_scores['BIC']:.4f}\n")
        if "FPE" in best_validation_scores:
            file.write(f"Final Pred Err (FPE) : {best_validation_scores['FPE']:.6f}\n")
        file.write(f"White-Noise ACF (Ree): {best_validation_scores['White_Noise_Pct']:.1f}% of lags within 95% bounds\n")
        file.write(f"Input-Error CCF (Rue): {deep_diag['CCF_Pass_Pct']:.1f}% of lags within 95% bounds\n")

        file.write("\n--- 3. Control Theory & Engineering Benchmarks ---\n")
        file.write(f"NRMSE (MATLAB FIT%)  : {deep_diag['NRMSE_Fit_Pct']:.2f}%\n")
        if "Sim_Fit_Pct" in best_validation_scores:
            file.write(f"Autonomous Sim Fit   : {best_validation_scores['Sim_Fit_Pct']:.2f}% (Free-Run Simulation)\n")
        file.write(f"Variance Accounted   : {deep_diag['VAF_Pct']:.2f}% (VAF)\n")
        file.write(f"Pearson Correlation r: {deep_diag['Pearson_R']:.4f}\n")
        file.write(f"Signal-to-Noise (SNR): {deep_diag['SNR_dB']:.2f} dB\n")
        file.write(f"Naive Persistence RMSE: {deep_diag['Naive_RMSE']:.4f} (Baseline y_hat(t) = y(t-1))\n")
        file.write(f"Theil's U1 / U2      : U1 = {deep_diag['Theil_U1']:.4f} | U2 = {deep_diag['Theil_U2']:.4f} (< 1.0 beats Naive)\n")
        file.write(f"Skill Over Naive     : {deep_diag['Skill_Over_Naive_Pct']:+.2f}%\n")
        if deep_diag["Max_AR_Pole_Mag"] is not None:
            stab_lbl = "STABLE (Inside Unit Circle)" if deep_diag["Is_AR_Stable"] else "UNSTABLE (Pole >= 1.0)"
            file.write(f"AR Max Pole Magnitude: |z|_max = {deep_diag['Max_AR_Pole_Mag']:.4f} -> {stab_lbl}\n")

        file.write("\n--- 4. Error Quantiles & Residual Distribution Shape ---\n")
        file.write(f"Mean Absolute Error  : {deep_diag['MAE']:.4f} (MAE)\n")
        file.write(f"Median Abs Error     : {deep_diag['MedAE']:.4f} (P50)\n")
        file.write(f"90th Percentile |e|  : {deep_diag['P90_AE']:.4f} (P90)\n")
        file.write(f"95th Percentile |e|  : {deep_diag['P95_AE']:.4f} (P95)\n")
        file.write(f"99th Percentile |e|  : {deep_diag['P99_AE']:.4f} (P99)\n")
        file.write(f"Max Absolute Error   : {deep_diag['Max_AE']:.4f} (L-Infinity Norm)\n")
        file.write(f"Residual Mean Bias   : {deep_diag['Mean_Bias']:+.4f} (Std = {deep_diag['Std_Err']:.4f})\n")
        file.write(f"Residual Skewness    : {deep_diag['Skewness']:+.4f} (0.0 = Symmetric)\n")
        file.write(f"Excess Kurtosis      : {deep_diag['Excess_Kurtosis']:+.4f} (0.0 = Normal Gaussian Tails)\n")

        file.write("\n--- 5. Extrema Reach Analysis (Maxima & Minima) ---\n")
        file.write(f"Global Max (t={max_idx:<5d}) : Actual = {true_max_val:>10.4f} | Pred @ Max = {pred_at_max:>10.4f} | Window Reach = {win_max:>10.4f} | Gap (Δy) = {pred_at_max - true_max_val:>+10.4f}\n")
        file.write(f"Global Min (t={min_idx:<5d}) : Actual = {true_min_val:>10.4f} | Pred @ Min = {pred_at_min:>10.4f} | Window Reach = {win_min:>10.4f} | Gap (Δy) = {pred_at_min - true_min_val:>+10.4f}\n")

        file.write("\n--- 6. Operating Regime (Quartile) Error Breakdown ---\n")
        reg_hdr = f"{'Operating Regime':<24} | {'Samples':>7} | {'Actual Range':<18} | {'RMSE':>10} | {'MAE':>10} | {'Mean Bias':>10}"
        file.write(reg_hdr + "\n" + "-" * len(reg_hdr) + "\n")
        for r in deep_diag["Regime_Table"]:
            file.write(
                f"{r['name']:<24} | {r['count']:>7d} | {r['y_range']:<18} | "
                f"{r['rmse']:>10.4f} | {r['mae']:>10.4f} | {r['bias']:>+10.4f}\n"
            )

        file.write("\n--- 7. Automated System Identification Diagnostic Verdict ---\n")
        r2_val = best_validation_scores["R-Squared"]
        v_r2 = "[PASS]" if r2_val >= 0.85 else ("[WARN]" if r2_val >= 0.50 else "[FAIL]")
        v_acf = "[PASS]" if deep_diag["ACF_Pass_Pct"] >= 80.0 else ("[WARN]" if deep_diag["ACF_Pass_Pct"] >= 50.0 else "[FAIL]")
        v_ccf = "[PASS]" if deep_diag["CCF_Pass_Pct"] >= 80.0 else ("[WARN]" if deep_diag["CCF_Pass_Pct"] >= 50.0 else "[FAIL]")
        unbiased_ok = abs(deep_diag["Mean_Bias"]) <= max(1e-4, 0.05 * deep_diag["Target_Std"])
        v_bias = "[PASS]" if unbiased_ok else "[WARN]"
        v_skill = "[PASS]" if deep_diag["Theil_U2"] < 1.0 else "[WARN]"

        file.write(f"• Variance Capture Check   : {v_r2} R² = {r2_val:.4f}, VAF = {deep_diag['VAF_Pct']:.2f}%\n")
        file.write(f"• Residual Whiteness (Ree) : {v_acf} {deep_diag['ACF_Pass_Pct']:.1f}% of ACF lags within 95% confidence bounds\n")
        file.write(f"• Input Independence (Rue) : {v_ccf} {deep_diag['CCF_Pass_Pct']:.1f}% of CCF lags within 95% confidence bounds\n")
        file.write(f"• Unbiased Estimator Check : {v_bias} Mean Error = {deep_diag['Mean_Bias']:+.4f} (Threshold <= 5% of target SD)\n")
        file.write(f"• Naive Baseline Comparison: {v_skill} Theil's U2 = {deep_diag['Theil_U2']:.4f} ({deep_diag['Skill_Over_Naive_Pct']:+.2f}% skill vs. random walk)\n")

    print(f"\n[*] Single-model deep evaluation complete (10 plots + 7-section report). Saved to: {output_dir}")
    return best_validation_scores, model_payload


def generate_latex_results_table(fold_telemetry, parameter_names, mean_theta_array, std_theta_array,
                                output_dir, dataset_name, model_type, system_mode):
    """
    Generates a LaTeX table summarizing the 10-fold cross-validation performance.
    """
    tex_path = os.path.join(output_dir, "table_results.tex")
    clean_ds = dataset_name.replace("_", r"\_")

    num_folds_run = len(fold_telemetry.get("MSE", [])) or 10
    lines = [
        r"% Auto-generated by Evolutionary System Identification Pipeline",
        r"% Direct inclusion in LaTeX: \input{" + os.path.basename(tex_path) + "}",
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{num_folds_run}-Fold Cross-Validation Performance for {model_type} ({system_mode}) on {clean_ds}}}",
        rf"\label{{tab:results_{dataset_name.lower()}_{model_type.lower()}}}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{MSE} & \textbf{RMSE} & \textbf{MAPE (\%)} & \textbf{$R^2$} & \textbf{AIC} & \textbf{BIC} & \textbf{ACF Pass (\%)} & \textbf{Time (s)} \\",
        r"\midrule"
    ]

    mean_mse = np.mean(fold_telemetry["MSE"])
    std_mse = np.std(fold_telemetry["MSE"])
    mean_rmse = np.mean(fold_telemetry["RMSE"])
    std_rmse = np.std(fold_telemetry["RMSE"])
    mean_mape = np.mean(fold_telemetry["MAPE"])
    std_mape = np.std(fold_telemetry["MAPE"])
    mean_r2 = np.mean(fold_telemetry["R-Squared"])
    std_r2 = np.std(fold_telemetry["R-Squared"])
    mean_aic = np.mean(fold_telemetry["AIC"])
    std_aic = np.std(fold_telemetry["AIC"])
    mean_bic = np.mean(fold_telemetry["BIC"])
    std_bic = np.std(fold_telemetry["BIC"])
    mean_wn = np.mean(fold_telemetry["White_Noise_Pct"])
    std_wn = np.std(fold_telemetry["White_Noise_Pct"])
    mean_time = np.mean(fold_telemetry["Execution_Time_Sec"])
    std_time = np.std(fold_telemetry["Execution_Time_Sec"])

    best_idx = int(np.argmin(fold_telemetry["MSE"]))

    lines.append(
        rf"Mean ($\mu$) & {mean_mse:.4f} & {mean_rmse:.4f} & {mean_mape:.2f} & {mean_r2:.4f} & {mean_aic:.1f} & {mean_bic:.1f} & {mean_wn:.1f} & {mean_time:.2f} \\"
    )
    lines.append(
        rf"Std ($\sigma$) & $\pm${std_mse:.4f} & $\pm${std_rmse:.4f} & $\pm${std_mape:.2f} & $\pm${std_r2:.4f} & $\pm${std_aic:.1f} & $\pm${std_bic:.1f} & $\pm${std_wn:.1f} & $\pm${std_time:.2f} \\"
    )
    lines.append(
        rf"Best Fold & \textbf{{{fold_telemetry['MSE'][best_idx]:.4f}}} & \textbf{{{fold_telemetry['RMSE'][best_idx]:.4f}}} & {fold_telemetry['MAPE'][best_idx]:.2f} & \textbf{{{fold_telemetry['R-Squared'][best_idx]:.4f}}} & {fold_telemetry['AIC'][best_idx]:.1f} & {fold_telemetry['BIC'][best_idx]:.1f} & {fold_telemetry['White_Noise_Pct'][best_idx]:.1f} & {fold_telemetry['Execution_Time_Sec'][best_idx]:.2f} \\"
    )

    if "Sim_Fit_Pct" in fold_telemetry and len(fold_telemetry["Sim_Fit_Pct"]) > 0:
        mean_sim = np.mean(fold_telemetry["Sim_Fit_Pct"])
        std_sim = np.std(fold_telemetry["Sim_Fit_Pct"])
        best_sim = fold_telemetry["Sim_Fit_Pct"][best_idx]
        lines.append(r"\midrule")
        lines.append(rf"Simulation Fit (\%) & \multicolumn{{8}}{{c}}{{Mean: {mean_sim:.2f}\% $\pm$ {std_sim:.2f}\% \quad (Best Fold: \textbf{{{best_sim:.2f}\%}})}} \\")

    if "AICc" in fold_telemetry and len(fold_telemetry["AICc"]) > 0:
        mean_aicc = np.mean(fold_telemetry["AICc"])
        std_aicc = np.std(fold_telemetry["AICc"])
        best_aicc = fold_telemetry["AICc"][best_idx]
        lines.append(r"\midrule")
        lines.append(rf"AICc & \multicolumn{{8}}{{c}}{{Mean: {mean_aicc:.2f} $\pm$ {std_aicc:.2f} \quad (Best Fold: \textbf{{{best_aicc:.2f}}})}} \\")

    if "FPE" in fold_telemetry and len(fold_telemetry["FPE"]) > 0:
        mean_fpe = np.mean(fold_telemetry["FPE"])
        std_fpe = np.std(fold_telemetry["FPE"])
        best_fpe = fold_telemetry["FPE"][best_idx]
        lines.append(rf"FPE & \multicolumn{{8}}{{c}}{{Mean: {mean_fpe:.4f} $\pm$ {std_fpe:.4f} \quad (Best Fold: \textbf{{{best_fpe:.4f}}})}} \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        ""
    ])

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[*] Exported LaTeX results table -> {os.path.basename(tex_path)}")
    return tex_path


def generate_transfer_function_latex(model_type, best_theta, parameter_names, na, nb, nc, nk, output_dir, dataset_name, n_u=1, n_y=1):
    """
    Generates LaTeX equations for the identified model transfer functions.
    Supports SISO, MISO, SIMO, and MIMO systems.
    """
    eq_path = os.path.join(output_dir, "model_equations.tex")
    clean_ds = dataset_name.replace("_", r"\_")

    lines = [
        r"% Identified System Model Equations",
        r"% Dataset: " + clean_ds + f", Model: {model_type} ({n_u} In -> {n_y} Out)",
        r"\begin{equation}",
        rf"\label{{eq:identified_model_{model_type.lower()}}}"
    ]

    theta = list(best_theta)
    if model_type == "ARMAX":
        params_per_output = na + (nb * n_u) + nc
        lines.append(r"\begin{aligned}")

        for out_idx in range(n_y):
            offset = out_idx * params_per_output
            theta_ar = theta[offset: offset + na]
            theta_b_all = theta[offset + na: offset + na + (nb * n_u)]
            theta_ma = theta[offset + na + (nb * n_u): offset + params_per_output]

            a_terms = ["1"]
            for i, val in enumerate(theta_ar, start=1):
                sign = "+" if val >= 0 else "-"
                a_terms.append(f"{sign} {abs(val):.4f} z^{{-{i}}}")
            denom_str = " ".join(a_terms)

            c_terms = ["1"]
            for k_idx, val in enumerate(theta_ma, start=1):
                sign = "+" if val >= 0 else "-"
                c_terms.append(f"{sign} {abs(val):.4f} z^{{-{k_idx}}}")
            ma_str = " ".join(c_terms)

            y_sym = f"y_{{{out_idx + 1}}}(t)" if n_y > 1 else "y(t)"
            e_sym = f"e_{{{out_idx + 1}}}(t)" if n_y > 1 else "e(t)"

            if n_u == 1:
                b_terms = []
                for j, val in enumerate(theta_b_all):
                    lag = nk + j
                    sign = "+" if val >= 0 else "-"
                    if not b_terms and val >= 0:
                        b_terms.append(f"{val:.4f} z^{{-{lag}}}")
                    else:
                        b_terms.append(f"{sign} {abs(val):.4f} z^{{-{lag}}}")
                num_str = " ".join(b_terms) if b_terms else "0"
                lines.append(rf"{y_sym} &= \frac{{{num_str}}}{{{denom_str}}} u(t) + \frac{{{ma_str}}}{{{denom_str}}} {e_sym} \\")
            else:
                g_terms = []
                for in_idx in range(n_u):
                    b_sub = theta_b_all[in_idx * nb: (in_idx + 1) * nb]
                    b_terms = []
                    for j, val in enumerate(b_sub):
                        lag = nk + j
                        sign = "+" if val >= 0 else "-"
                        if not b_terms and val >= 0:
                            b_terms.append(f"{val:.4f} z^{{-{lag}}}")
                        else:
                            b_terms.append(f"{sign} {abs(val):.4f} z^{{-{lag}}}")
                    num_str = " ".join(b_terms) if b_terms else "0"
                    g_terms.append(rf"\frac{{{num_str}}}{{{denom_str}}} u_{{{in_idx + 1}}}(t)")
                rhs = " + ".join(g_terms) + rf" + \frac{{{ma_str}}}{{{denom_str}}} {e_sym}"
                lines.append(rf"{y_sym} &= {rhs} \\")

        lines.append(r"\end{aligned}")
        lines.append(r"\end{equation}")
    elif model_type == "SVR":
        c_val, eps_val, gamma_val = theta[0], theta[1], theta[2]
        lines.extend([
            rf"\text{{SVR-RBF:}}\quad C = {c_val:.4f}, \quad \epsilon = {eps_val:.4f}, \quad \gamma = {gamma_val:.4f}",
            r"\end{equation}"
        ])
    else:
        lines.extend([
            rf"\text{{MLP NARX [{n_u}\text{{In}}\to{n_y}\text{{Out}}]}}:\quad n_a = {na}, \quad n_b = {nb}, \quad n_k = {nk}, \quad \text{{Parameters: }} {len(theta)}",
            r"\end{equation}"
        ])

    with open(eq_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[*] Exported LaTeX model equations -> {os.path.basename(eq_path)}")
    return eq_path


def execute_kfold_training_pipeline(u_raw, y_raw, dataset_name, hyperparameters, output_dir="plots", progress_callback=None, stop_event=None):
    # Runs 10-fold cross-validation training for SISO, MISO, SIMO, or MIMO and exports all artifacts
    seed_val = hyperparameters.get("random_seed", 42)
    if seed_val is not None:
        enforce_reproducibility_seed(int(seed_val))
    pipeline_start_time = time.perf_counter()

    os.makedirs(output_dir, exist_ok=True)

    active_config = hyperparameters.copy()
    model_type = active_config.get("model_type", "ARMAX")
    system_mode = str(active_config.get("system_mode", "SISO")).upper()
    input_col_idx = int(active_config.get("input_col_idx", 0))
    output_col_idx = int(active_config.get("output_col_idx", 0))

    # Route raw dataset matrices into 1D or 2D signals according to SISO / MISO / SIMO / MIMO
    u_signal, y_signal = slice_signals_by_system_mode(
        u_raw, y_raw, system_mode=system_mode, input_col_idx=input_col_idx, output_col_idx=output_col_idx
    )
    n_u = 1 if u_signal.ndim == 1 else u_signal.shape[1]
    n_y = 1 if y_signal.ndim == 1 else y_signal.shape[1]
    active_config["n_u"] = n_u
    active_config["n_y"] = n_y

    na, nb, nk = int(active_config["na"]), int(active_config["nb"]), int(active_config["nk"])
    nc = int(active_config.get("nc", 2)) if model_type == "ARMAX" else 0

    if (na + nb) <= 0:
        raise ValueError("Invalid architecture: At least one AR lag (na >= 1) or Input lag (nb >= 1) is required.")

    max_delay_lag_check = max(1, na, nb + nk - 1, nc)
    k_folds = max(2, int(active_config.get("k_folds", 10)))
    min_test_fold_size = len(y_signal) // k_folds
    if max_delay_lag_check >= (min_test_fold_size - 10):
        max_allowed = max(1, min_test_fold_size - 15)
        raise ValueError(
            f"Lag configuration (max lag = {max_delay_lag_check}) is too large for '{dataset_name}' "
            f"({k_folds}-fold test split has only {min_test_fold_size} samples). "
            f"Please use lags <= {max_allowed} for this dataset."
        )

    hidden_units = active_config.get("mlp_hidden_nodes", [5])
    max_generations = active_config["num_gen"]

    arch_summary, model_specific_config = _format_model_architecture_and_filter(model_type, active_config, n_u=n_u, n_y=n_y)

    metadata_filepath = os.path.join(output_dir, "run_metadata.json")
    tmp_meta = metadata_filepath + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as meta_file:
        json.dump({
            "dataset_name": dataset_name,
            "system_mode": system_mode,
            "hyperparameters": model_specific_config
        }, meta_file, indent=4)
    atomic_replace_retry(tmp_meta, metadata_filepath)

    print(f"\n{'=' * 60}")
    print(f"Training on Dataset: {dataset_name} [{system_mode}: {n_u}In -> {n_y}Out]")
    print(f"Output Directory: {output_dir}")
    print(f"Structure: {arch_summary}")
    print(f"{'=' * 60}")

    # For dynamic time-series, shuffle must be False to preserve chronological dependencies
    cross_val_splits = KFold(n_splits=k_folds, shuffle=False)

    fold_telemetry = {
        "MSE": [], "RMSE": [], "MAPE": [], "R-Squared": [],
        "AIC": [], "AICc": [], "BIC": [], "FPE": [], "White_Noise_Pct": [],
        "Sim_Fit_Pct": [],
        "Generations": [], "Execution_Time_Sec": []
    }
    fold_theta_matrices = []
    fold_scaler_objects = []
    fold_train_scaled_tuples = []
    fold_ep_logbooks = []
    fold_prediction_tuples = []
    console_report_lines = []

    current_fold = 1

    for fold_idx, (train_indices, test_indices) in enumerate(cross_val_splits.split(y_signal)):
        fold_start = time.perf_counter()
        current_fold = fold_idx + 1

        if stop_event and stop_event.is_set():
            break

        if progress_callback:
            progress_callback(f"Running {dataset_name} ({model_type}-{system_mode}): Fold {current_fold}/{k_folds}...")

        # Fit scalers strictly on train_indices to avoid data leakage into the test fold
        u_scaler = _fit_scaler_nd(u_signal[train_indices])
        y_scaler = _fit_scaler_nd(y_signal[train_indices])

        u_train_scaled = _transform_nd(u_scaler, u_signal[train_indices])
        y_train_scaled = _transform_nd(y_scaler, y_signal[train_indices])
        u_test_scaled = _transform_nd(u_scaler, u_signal[test_indices])
        y_test_scaled = _transform_nd(y_scaler, y_signal[test_indices])

        active_config["checkpoint_file"] = os.path.join(output_dir, f"checkpoint_fold_{fold_idx}.pkl")

        def _on_ep_gen(gen_num, max_gens, best_val, el_time):
            if progress_callback:
                try:
                    progress_callback({
                        "type": "generation",
                        "job_key": output_dir,
                        "model_type": model_type,
                        "system_mode": system_mode,
                        "dataset_name": dataset_name,
                        "output_dir": output_dir,
                        "fold": current_fold,
                        "k_folds": k_folds,
                        "generation": gen_num,
                        "max_gen": max_gens,
                        "best_score": best_val,
                        "elapsed": el_time
                    })
                except Exception:
                    pass

        optimal_theta, ep_logbook = execute_evolutionary_programming(
            u_train_scaled, y_train_scaled, active_config, stop_event=stop_event,
            generation_callback=_on_ep_gen
        )

        if stop_event and stop_event.is_set():
            break

        if optimal_theta is None or not ep_logbook:
            break

        if model_type == "SVR":
            total_lags_svr = max(1, na * n_y + nb * n_u)
            max_gamma_svr = 5.0 if n_u == 1 else (4.0 / float(total_lags_svr))
            optimal_theta = [
                min(max(1e-3, abs(optimal_theta[0])), 50.0),
                min(max(1e-4, abs(optimal_theta[1])), 2.0),
                min(max(1e-4, abs(optimal_theta[2])), max_gamma_svr)
            ]

        fold_theta_matrices.append(optimal_theta)
        fold_scaler_objects.append((u_scaler, y_scaler))
        fold_train_scaled_tuples.append((u_train_scaled, y_train_scaled))
        fold_ep_logbooks.append(ep_logbook)

        final_generation_reached = ep_logbook.select("gen")[-1]
        wall_fold_time = time.perf_counter() - fold_start
        saved_ckpt_time = float(getattr(ep_logbook, "elapsed_time", 0.0))
        fold_elapsed_time = saved_ckpt_time if saved_ckpt_time > 0.0 else wall_fold_time

        fold_telemetry["Generations"].append(final_generation_reached)
        fold_telemetry["Execution_Time_Sec"].append(fold_elapsed_time)
        convergence_status = "Stopped Early" if final_generation_reached < max_generations else "Max Gen Reached"

        simulated_y_scaled = None
        if model_type == "ARMAX":
            nc = int(active_config.get("nc", 2))
            fold_param_count = get_armax_parameter_count(na, nb, nc, n_u=n_u, n_y=n_y)
            predicted_y_scaled, max_delay_lag = simulate_armax_system(
                u_test_scaled, y_test_scaled, optimal_theta, na, nb, nc, nk
            )
            simulated_y_scaled, _ = simulate_armax_free_run(
                u_test_scaled, y_test_scaled, optimal_theta, na, nb, nc, nk
            )
        elif model_type == "SVR":
            fold_param_count = get_svr_parameter_count(na, nb, n_u=n_u, n_y=n_y)
            predicted_y_scaled, max_delay_lag = simulate_svr_system(
                u_test_scaled, y_test_scaled, optimal_theta, na, nb, nk,
                u_train=u_train_scaled, y_train=y_train_scaled
            )
            simulated_y_scaled, _ = simulate_svr_free_run(
                u_test_scaled, y_test_scaled, optimal_theta, na, nb, nk,
                u_train=u_train_scaled, y_train=y_train_scaled
            )
        else:
            fold_param_count = get_mlp_parameter_count(na, nb, hidden_units, n_u=n_u, n_y=n_y)
            predicted_y_scaled, max_delay_lag = simulate_mlp_system(
                u_test_scaled, y_test_scaled, optimal_theta, na, nb, nk, hidden_units
            )
            simulated_y_scaled, _ = simulate_mlp_free_run(
                u_test_scaled, y_test_scaled, optimal_theta, na, nb, nk, hidden_units
            )

        # Invert scaling back to original physical units before calculating metrics
        y_actual_unscaled = _inverse_transform_nd(y_scaler, y_test_scaled[max_delay_lag:])
        y_pred_unscaled = _inverse_transform_nd(y_scaler, predicted_y_scaled[max_delay_lag:])
        y_sim_unscaled = _inverse_transform_nd(y_scaler, simulated_y_scaled[max_delay_lag:]) if simulated_y_scaled is not None else None

        fold_prediction_tuples.append((y_actual_unscaled, y_pred_unscaled, y_sim_unscaled))

        validation_scores = generate_validation_report(
            y_actual_unscaled, y_pred_unscaled, model_parameter_count=fold_param_count,
            simulated_outputs=y_sim_unscaled
        )
        for key, value in validation_scores.items():
            if key in fold_telemetry:
                fold_telemetry[key].append(value)

        sim_str = f" | SimFit: {validation_scores['Sim_Fit_Pct']:>5.1f}%" if "Sim_Fit_Pct" in validation_scores else ""
        status_line = (
            f"Fold {current_fold:02d} | Gen: {final_generation_reached:>3d}/{max_generations} ({convergence_status:<15}) | "
            f"Time: {fold_elapsed_time:>5.2f}s | "
            f"MSE: {validation_scores['MSE']:>8.4f} | "
            f"RMSE: {validation_scores['RMSE']:>8.4f} | "
            f"MAPE: {validation_scores['MAPE']:>5.2f}% | "
            f"R2: {validation_scores['R-Squared']:>7.4f} | "
            f"AIC: {validation_scores['AIC']:>8.4f} | "
            f"BIC: {validation_scores['BIC']:>8.4f} | "
            f"WN%: {validation_scores['White_Noise_Pct']:>5.1f}%"
            f"{sim_str}"
        )
        print(status_line)
        console_report_lines.append(status_line)

        if progress_callback:
            try:
                progress_callback({
                    "type": "fold_complete",
                    "job_key": output_dir,
                    "model_type": model_type,
                    "system_mode": system_mode,
                    "dataset_name": dataset_name,
                    "output_dir": output_dir,
                    "fold": current_fold,
                    "k_folds": k_folds,
                    "val_mse": float(validation_scores.get("MSE", 0.0)),
                    "val_r2": float(validation_scores.get("R-Squared", 0.0))
                })
            except Exception:
                pass

    if len(fold_theta_matrices) < k_folds:
        if progress_callback:
            try:
                progress_callback({
                    "type": "job_stopped",
                    "job_key": output_dir,
                    "model_type": model_type,
                    "system_mode": system_mode,
                    "dataset_name": dataset_name,
                    "output_dir": output_dir,
                    "fold": current_fold,
                    "k_folds": k_folds
                })
            except Exception:
                pass
            progress_callback(f"Training aborted during Fold {current_fold}/{k_folds}. Checkpoints saved.")
        abort_path = os.path.join(output_dir, "abort_log.txt")
        tmp_abort = abort_path + ".tmp"
        with open(tmp_abort, "w", encoding="utf-8") as f:
            f.write(f"Training manually aborted by user during Fold {current_fold}/{k_folds}.\n")
            f.write(f"Completed Folds: {len(fold_theta_matrices)}/{k_folds}.\n")
            f.write("Checkpoints are saved in this folder. Use 'Resume Previous Run' to continue.\n")
        atomic_replace_retry(tmp_abort, abort_path)
        print(f"[!] Training aborted at Fold {current_fold}. State saved to: {output_dir}")
        return

    abort_path = os.path.join(output_dir, "abort_log.txt")
    if os.path.exists(abort_path):
        os.remove(abort_path)

    wall_pipeline_time = time.perf_counter() - pipeline_start_time
    sum_fold_times = float(np.sum(fold_telemetry["Execution_Time_Sec"]))
    pipeline_elapsed_time = max(wall_pipeline_time, sum_fold_times)

    parameter_names = _build_parameter_names(model_type, na, nb, nc=nc, hidden_units=hidden_units, n_u=n_u, n_y=n_y)

    base_struct = {
        "model_type": model_type, "system_mode": system_mode,
        "input_col_idx": input_col_idx, "output_col_idx": output_col_idx,
        "n_u": n_u, "n_y": n_y, "na": na, "nb": nb, "nk": nk
    }
    if model_type == "ARMAX":
        structure_dict = {**base_struct, "nc": nc}
    elif model_type == "SVR":
        structure_dict = base_struct
    else:
        structure_dict = {**base_struct, "mlp_hidden_nodes": hidden_units}

    mean_theta_array = np.mean(fold_theta_matrices, axis=0)
    std_theta_array = np.std(fold_theta_matrices, axis=0)

    best_fold_index = int(np.argmin(fold_telemetry["MSE"]))
    winning_fold_number = best_fold_index + 1
    best_u_scaler, best_y_scaler = fold_scaler_objects[best_fold_index]
    best_u_train_scaled, best_y_train_scaled = fold_train_scaled_tuples[best_fold_index]
    best_actual_targets, best_predicted_outputs, best_simulated_outputs = fold_prediction_tuples[best_fold_index]

    summary_statistics = []
    for metric, value_array in fold_telemetry.items():
        if not value_array:
            continue
        mean_val, std_val = np.mean(value_array), np.std(value_array)
        if metric == "MAPE":
            summary_statistics.append(f"MAPE: {mean_val:.2f}% +/- {std_val:.2f}%")
        elif metric == "White_Noise_Pct":
            summary_statistics.append(f"White-Noise ACF Pass Rate: {mean_val:.1f}% +/- {std_val:.1f}%")
        elif metric == "Sim_Fit_Pct":
            summary_statistics.append(f"Autonomous Simulation Fit: {mean_val:.2f}% +/- {std_val:.2f}% (Best Fold: {fold_telemetry['Sim_Fit_Pct'][best_fold_index]:.2f}%)")
        elif metric == "Generations":
            summary_statistics.append(f"Generations Run: {mean_val:.1f} +/- {std_val:.1f} (Max: {max_generations})")
        elif metric == "Execution_Time_Sec":
            summary_statistics.append(f"Execution Time: {mean_val:.2f}s +/- {std_val:.2f}s")
        else:
            summary_statistics.append(f"{metric}: {mean_val:.4f} +/- {std_val:.4f}")

    artifact_basename = _build_model_artifact_basename(output_dir, model_type, dataset_name)

    model_binary_payload = {
        "dataset": dataset_name,
        "structure": structure_dict,
        "param_names": parameter_names,
        "best_fold_index": winning_fold_number,
        "best_fold_theta": fold_theta_matrices[best_fold_index],
        "best_fold_mse": float(fold_telemetry["MSE"][best_fold_index]),
        "mean_theta_10fold": mean_theta_array.tolist(),
        "std_theta_10fold": std_theta_array.tolist(),
        "all_fold_thetas": fold_theta_matrices,
        "scaler_u": best_u_scaler,
        "scaler_y": best_y_scaler,
        "svr_u_train": best_u_train_scaled if model_type == "SVR" else None,
        "svr_y_train": best_y_train_scaled if model_type == "SVR" else None
    }
    pkl_filepath = os.path.join(output_dir, f"{artifact_basename}.pkl")
    tmp_pkl = pkl_filepath + ".tmp"
    with open(tmp_pkl, "wb") as file:
        pickle.dump(model_binary_payload, file)
    atomic_replace_retry(tmp_pkl, pkl_filepath)

    # Save 1D scalars or 2D lists for mean/std depending on SISO vs multivariable
    u_mean_json = float(best_u_scaler.mean_[0]) if len(best_u_scaler.mean_) == 1 else best_u_scaler.mean_.tolist()
    u_std_json = float(best_u_scaler.scale_[0]) if len(best_u_scaler.scale_) == 1 else best_u_scaler.scale_.tolist()
    y_mean_json = float(best_y_scaler.mean_[0]) if len(best_y_scaler.mean_) == 1 else best_y_scaler.mean_.tolist()
    y_std_json = float(best_y_scaler.scale_[0]) if len(best_y_scaler.scale_) == 1 else best_y_scaler.scale_.tolist()

    model_json_payload = {
        "dataset": dataset_name,
        "structure": structure_dict,
        "best_fold": winning_fold_number,
        "best_fold_mse": float(fold_telemetry["MSE"][best_fold_index]),
        "best_fold_parameters": dict(zip(parameter_names, [float(v) for v in fold_theta_matrices[best_fold_index]])),
        "mean_10fold_parameters": dict(zip(parameter_names, [float(v) for v in mean_theta_array])),
        "std_10fold_parameters": dict(zip(parameter_names, [float(v) for v in std_theta_array])),
        "scaler_u": {"mean": u_mean_json, "std": u_std_json},
        "scaler_y": {"mean": y_mean_json, "std": y_std_json},
        "svr_u_train": best_u_train_scaled.tolist() if model_type == "SVR" else None,
        "svr_y_train": best_y_train_scaled.tolist() if model_type == "SVR" else None
    }
    json_filepath = os.path.join(output_dir, f"{artifact_basename}.json")
    tmp_json = json_filepath + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as file:
        json.dump(model_json_payload, file, indent=4)
    atomic_replace_retry(tmp_json, json_filepath)

    report_filepath = os.path.join(output_dir, "training_report.txt")
    with open(report_filepath, "w", encoding="utf-8") as file:
        file.write("=== Training Report ===\n")
        file.write(f"Dataset         : {dataset_name}\n")
        file.write(f"Model Type      : {model_type}\n")
        file.write(f"System Mode     : {system_mode} ({n_u} Inputs -> {n_y} Outputs)\n")
        file.write(f"Saved Model     : {artifact_basename}.pkl / .json\n")
        file.write(f"Architecture    : {arch_summary}\n")
        file.write(f"Execution Time  : {pipeline_elapsed_time:.2f} seconds\n")

        file.write("\n--- Parameters ---\n")
        for key, val in model_specific_config.items():
            file.write(f"{get_parameter_label(key):<35}: {val}\n")

        file.write("\n--- Fold Results ---\n")
        file.write("\n".join(console_report_lines) + "\n")

        file.write("\n--- Model Parameters by Fold ---\n")
        header_formatting = "Fold    | " + " | ".join([f"{name:>8}" for name in parameter_names])
        file.write(header_formatting + "\n" + "-" * len(header_formatting) + "\n")
        for i, theta_vals in enumerate(fold_theta_matrices):
            formatted_vals = " | ".join([f"{val:>8.4f}" for val in theta_vals])
            file.write(f"Fold {i + 1:02d} | {formatted_vals}\n")

        file.write("\n--- Average Parameter Values ---\n")
        for name, mean_val, std_val in zip(parameter_names, mean_theta_array, std_theta_array):
            file.write(f"{name}: {mean_val:>8.4f} +/- {std_val:.4f}\n")

        file.write(f"\n--- Best Parameters (Fold {winning_fold_number:02d}, MSE: {fold_telemetry['MSE'][best_fold_index]:.4f}) ---\n")
        for name, best_val in zip(parameter_names, fold_theta_matrices[best_fold_index]):
            file.write(f"{name}: {best_val:>8.4f}\n")

        file.write("\n--- Cross-Validation Summary ---\n")
        file.write("\n".join(summary_statistics) + "\n")

    print(f"\n[*] Training complete. Model saved as: {artifact_basename}.pkl")
    print(f"[*] Report and plots saved to: {output_dir}")
    print(f"\n--- Summary for {dataset_name} ---")
    print("\n".join(summary_statistics))

    chart_suffix = f" ({model_type} - Best of {k_folds} Folds)"
    generate_convergence_plot(fold_ep_logbooks[best_fold_index], dataset_name, winning_fold_number, output_dir=output_dir, title_suffix=chart_suffix)
    generate_prediction_tracking_plot(best_actual_targets, best_predicted_outputs, dataset_name, winning_fold_number, output_dir=output_dir, title_suffix=chart_suffix)
    generate_residual_diagnostics_plot(best_actual_targets, best_predicted_outputs, dataset_name, winning_fold_number, output_dir=output_dir, title_suffix=chart_suffix)
    generate_cross_validation_summary(fold_telemetry, fold_theta_matrices, parameter_names, dataset_name, output_dir=output_dir)

    if best_simulated_outputs is not None:
        generate_simulation_vs_prediction_plot(
            best_actual_targets, best_predicted_outputs, best_simulated_outputs,
            dataset_name, winning_fold_number, output_dir=output_dir, title_suffix=chart_suffix
        )
    if model_type == "ARMAX":
        generate_bode_diagram(
            fold_theta_matrices[best_fold_index], na, nb, nc, nk,
            dataset_name, winning_fold_number, output_dir=output_dir, title_suffix=chart_suffix,
            n_u=n_u, n_y=n_y
        )

    generate_latex_results_table(
        fold_telemetry, parameter_names, mean_theta_array, std_theta_array,
        output_dir=output_dir, dataset_name=dataset_name, model_type=model_type, system_mode=system_mode
    )
    generate_transfer_function_latex(
        model_type, fold_theta_matrices[best_fold_index], parameter_names,
        na, nb, nc, nk, output_dir=output_dir, dataset_name=dataset_name,
        n_u=n_u, n_y=n_y
    )

    if progress_callback:
        try:
            progress_callback({
                "type": "job_complete",
                "job_key": output_dir,
                "model_type": model_type,
                "system_mode": system_mode,
                "dataset_name": dataset_name,
                "output_dir": output_dir,
                "best_mse": float(fold_telemetry["MSE"][best_fold_index]),
                "best_r2": float(fold_telemetry["R-Squared"][best_fold_index]),
                "total_time": wall_pipeline_time
            })
        except Exception:
            pass

def execute_model_comparison_pipeline(model_filepaths, u_raw, y_raw, target_dataset_name, output_dir="results_comparison", progress_callback=None):
    # Evaluates multiple saved models on the same dataset, aligns burn-in windows, and exports ranked reports
    if not model_filepaths or len(model_filepaths) < 2:
        raise ValueError("Please select at least 2 model files to compare.")

    execution_start = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    raw_simulations = []
    used_labels = set()

    # If any SISO or MISO model is in the comparison list, use its trained output_col_idx as the shared
    # reference channel (since SIMO/MIMO models predict all channels and can be sliced to match).
    ref_out_col = 0
    for fp_check in model_filepaths:
        try:
            chk_payload = _parse_serialized_model(fp_check)
            chk_struct = chk_payload.get("structure", {})
            chk_mode = str(chk_struct.get("system_mode", "SISO")).upper()
            if chk_mode in ("SISO", "MISO"):
                ref_out_col = int(chk_struct.get("output_col_idx", 0))
                break
        except Exception:
            pass

    _, y_ref_1d = slice_signals_by_system_mode(u_raw, y_raw, system_mode="SISO", output_col_idx=ref_out_col)

    for idx, filepath in enumerate(model_filepaths, start=1):
        if progress_callback:
            progress_callback(f"Simulating model {idx}/{len(model_filepaths)}: {os.path.basename(filepath)}...")

        model_payload = _parse_serialized_model(filepath)
        architecture = model_payload["structure"]
        model_type = architecture.get("model_type", "ARMAX")
        system_mode = architecture.get("system_mode", "SISO")
        input_col_idx = int(architecture.get("input_col_idx", 0))
        output_col_idx = int(architecture.get("output_col_idx", 0))
        na, nb, nk = int(architecture["na"]), int(architecture["nb"]), int(architecture["nk"])
        hidden_units = architecture.get("mlp_hidden_nodes", [5])
        nc = int(architecture.get("nc", 2))
        best_theta_weights = np.asarray(model_payload["best_fold_theta"], dtype=float)

        u_signal, y_signal = slice_signals_by_system_mode(
            u_raw, y_raw, system_mode=system_mode, input_col_idx=input_col_idx, output_col_idx=output_col_idx
        )
        n_u = 1 if u_signal.ndim == 1 else u_signal.shape[1]
        n_y = 1 if y_signal.ndim == 1 else y_signal.shape[1]

        stem = os.path.splitext(os.path.basename(filepath))[0].replace("_model", "")
        parts = stem.split("_")
        short_id = parts[-1] if (len(parts) >= 2 and len(parts[-1]) == 6) else f"M{idx}"
        base_label = f"{model_type}_{short_id}"
        label = base_label
        counter = 2
        while label in used_labels:
            label = f"{base_label}_{counter}"
            counter += 1
        used_labels.add(label)

        source_dataset = model_payload.get("dataset", target_dataset_name)
        is_cross_domain = (source_dataset != target_dataset_name)

        u_scaler = model_payload.get("scaler_u")
        y_scaler = model_payload.get("scaler_y")
        if is_cross_domain or u_scaler is None or getattr(u_scaler, "n_features_in_", 1) != n_u:
            u_scaler = _fit_scaler_nd(u_signal)
        if is_cross_domain or y_scaler is None or getattr(y_scaler, "n_features_in_", 1) != n_y:
            y_scaler = _fit_scaler_nd(y_signal)

        u_scaled = _transform_nd(u_scaler, u_signal)
        y_scaled = _transform_nd(y_scaler, y_signal)

        if model_type == "ARMAX":
            total_parameters = get_armax_parameter_count(na, nb, nc, n_u=n_u, n_y=1)
            arch_str = f"ARMAX[{system_mode}](na={na},nb={nb},nc={nc},nk={nk})"
            predicted_y_scaled, max_delay_lag = simulate_armax_system(u_scaled, y_scaled, best_theta_weights, na, nb, nc, nk)
        elif model_type == "SVR":
            total_parameters = get_svr_parameter_count(na, nb, n_u=n_u, n_y=n_y)
            arch_str = f"SVR[{system_mode}](na={na},nb={nb},nk={nk})"
            predicted_y_scaled, max_delay_lag = simulate_svr_system(
                u_scaled, y_scaled, best_theta_weights, na, nb, nk,
                u_train=model_payload.get("svr_u_train"), y_train=model_payload.get("svr_y_train"),
                cross_domain=is_cross_domain
            )
        else:
            total_parameters = get_mlp_parameter_count(na, nb, hidden_units, n_u=n_u, n_y=n_y)
            arch_str = f"MLP[{system_mode}](na={na},nb={nb},nk={nk},H={hidden_units})"
            u_in = np.clip(u_scaled, -3.0, 3.0) if is_cross_domain else u_scaled
            y_in = np.clip(y_scaled, -3.0, 3.0) if is_cross_domain else y_scaled
            predicted_y_scaled, max_delay_lag = simulate_mlp_system(u_in, y_in, best_theta_weights, na, nb, nk, hidden_units)
            if is_cross_domain:
                pred_valid = predicted_y_scaled[max_delay_lag:]
                pred_std = np.std(pred_valid, axis=0, keepdims=True)
                pred_mean = np.mean(pred_valid, axis=0, keepdims=True)
                predicted_y_scaled[max_delay_lag:] = np.where(pred_std > 1e-6, (pred_valid - pred_mean) / np.maximum(pred_std, 1e-6), pred_valid - pred_mean)

        if not np.isfinite(predicted_y_scaled).all():
            raise RuntimeError(f"Model '{os.path.basename(filepath)}' produced unstable predictions.")

        y_unscaled_valid = _inverse_transform_nd(y_scaler, predicted_y_scaled[max_delay_lag:])
        # Extract reference output channel (1D) so SISO, MISO, SIMO, and MIMO models can be compared head-to-head
        if y_unscaled_valid.ndim > 1:
            safe_ch = min(max(0, ref_out_col), y_unscaled_valid.shape[1] - 1)
            y_unscaled_valid = y_unscaled_valid[:, safe_ch]

        y_pred_full_unscaled = np.zeros_like(y_ref_1d, dtype=float)
        y_pred_full_unscaled[max_delay_lag:] = y_unscaled_valid

        raw_simulations.append({
            "label": label,
            "filepath": os.path.abspath(filepath),
            "model_type": model_type,
            "arch_str": arch_str,
            "param_count": total_parameters,
            "max_delay_lag": max_delay_lag,
            "y_pred_full": y_pred_full_unscaled
        })

    common_start_lag = max(item["max_delay_lag"] for item in raw_simulations)
    y_actual_aligned = y_ref_1d[common_start_lag:]

    model_predictions_dict = {}
    model_metrics_dict = {}

    for item in raw_simulations:
        label = item["label"]
        y_pred_aligned = item["y_pred_full"][common_start_lag:]
        model_predictions_dict[label] = y_pred_aligned
        scores = generate_validation_report(y_actual_aligned, y_pred_aligned, model_parameter_count=item["param_count"])
        model_metrics_dict[label] = scores
        item["scores"] = scores
        item["y_pred_aligned"] = y_pred_aligned

    if progress_callback:
        progress_callback("Generating multi-model comparison plots and tables...")

    generate_multi_model_comparison_plots(
        y_actual_aligned, model_predictions_dict, model_metrics_dict,
        target_dataset_name, output_dir=output_dir
    )

    ranked_models = sorted(raw_simulations, key=lambda m: m["scores"]["MSE"])
    max_idx = int(np.argmax(y_actual_aligned))
    min_idx = int(np.argmin(y_actual_aligned))
    true_max_val = float(y_actual_aligned[max_idx])
    true_min_val = float(y_actual_aligned[min_idx])

    time_indices = np.arange(common_start_lag, len(y_ref_1d))
    csv_cols = [time_indices, y_actual_aligned] + [m["y_pred_aligned"] for m in ranked_models]
    csv_header = "Time_Step,Actual_y," + ",".join([f"Pred_{m['label']}" for m in ranked_models])
    csv_fmt = ["%d", "%.6f"] + ["%.6f"] * len(ranked_models)
    np.savetxt(
        os.path.join(output_dir, "comparison_predictions.csv"),
        np.column_stack(csv_cols),
        delimiter=",", header=csv_header, comments="", fmt=csv_fmt
    )

    metrics_csv_path = os.path.join(output_dir, "comparison_metrics.csv")
    with open(metrics_csv_path, "w", encoding="utf-8") as f:
        f.write("Rank,Model_Label,Architecture,MSE,RMSE,MAPE_Percent,R_Squared,AIC,BIC,White_Noise_Pct,Pred_At_Global_Max,Delta_Max,Pred_At_Global_Min,Delta_Min\n")
        for rank, m in enumerate(ranked_models, start=1):
            sc = m["scores"]
            p_max = float(m["y_pred_aligned"][max_idx])
            p_min = float(m["y_pred_aligned"][min_idx])
            f.write(
                f"{rank},{m['label']},\"{m['arch_str']}\",{sc['MSE']:.6f},{sc['RMSE']:.6f},"
                f"{sc['MAPE']:.4f},{sc['R-Squared']:.6f},{sc['AIC']:.4f},{sc['BIC']:.4f},{sc['White_Noise_Pct']:.1f},"
                f"{p_max:.6f},{p_max - true_max_val:+.6f},{p_min:.6f},{p_min - true_min_val:+.6f}\n"
            )

    execution_elapsed = time.perf_counter() - execution_start

    report_filepath = os.path.join(output_dir, "comparison_report.txt")
    with open(report_filepath, "w", encoding="utf-8") as file:
        file.write("=== Multi-Model Comparison Report ===\n")
        file.write(f"Evaluated Dataset    : {target_dataset_name}\n")
        file.write(f"Models Compared      : {len(ranked_models)}\n")
        file.write(f"Aligned Burn-in Lag  : {common_start_lag} steps (N = {len(y_actual_aligned)} aligned samples)\n")
        file.write(f"Execution Time       : {execution_elapsed:.4f} seconds\n\n")

        file.write("--- 1. Overall Validation Metrics Table (Ranked by MSE) ---\n")
        header = f"{'Rank':<5} | {'Model Label':<14} | {'Architecture':<32} | {'MSE':>10} | {'RMSE':>10} | {'MAPE':>8} | {'R-Squared':>10} | {'AIC':>11} | {'BIC':>11}"
        file.write(header + "\n" + "-" * len(header) + "\n")
        for rank, m in enumerate(ranked_models, start=1):
            sc = m["scores"]
            file.write(
                f"{rank:<5d} | {m['label']:<14} | {m['arch_str']:<32} | "
                f"{sc['MSE']:>10.4f} | {sc['RMSE']:>10.4f} | {sc['MAPE']:>7.2f}% | "
                f"{sc['R-Squared']:>10.4f} | {sc['AIC']:>11.2f} | {sc['BIC']:>11.2f}\n"
            )

        file.write(f"\n--- 2. Extrema Reach Analysis (Global Max @ t={max_idx}: {true_max_val:.4f} | Global Min @ t={min_idx}: {true_min_val:.4f}) ---\n")
        ext_header = f"{'Model Label':<14} | {'Pred @ Max':>12} | {'Max Gap (Δy)':>13} | {'Pred @ Min':>12} | {'Min Gap (Δy)':>13}"
        file.write(ext_header + "\n" + "-" * len(ext_header) + "\n")
        for m in ranked_models:
            p_max = float(m["y_pred_aligned"][max_idx])
            p_min = float(m["y_pred_aligned"][min_idx])
            file.write(
                f"{m['label']:<14} | {p_max:>12.4f} | {p_max - true_max_val:>+13.4f} | "
                f"{p_min:>12.4f} | {p_min - true_min_val:>+13.4f}\n"
            )

        file.write("\n--- 3. Source Model Files ---\n")
        for m in ranked_models:
            file.write(f"• {m['label']:<14} : {m['filepath']}\n")

    print(f"\n[*] Model comparison complete! Saved to: {output_dir}")
    return ranked_models


def execute_source_dataset_check(model_filepaths, loaded_datasets_dict, base_results_dir="results", timestamp_sig="", progress_callback=None):
    # Evaluates 1 or more models on their original training dataset
    if not model_filepaths:
        raise ValueError("No model files selected.")

    grouped_by_dataset = {}
    for fp in model_filepaths:
        payload = _parse_serialized_model(fp)
        source_ds = payload.get("dataset")
        if not source_ds or source_ds not in loaded_datasets_dict:
            raise ValueError(
                f"Model '{os.path.basename(fp)}' was trained on dataset '{source_ds}', "
                f"which is not loaded. Please load '{source_ds}' first."
            )
        grouped_by_dataset.setdefault(source_ds, []).append(fp)

    output_summaries = []
    for source_ds, fp_list in grouped_by_dataset.items():
        u_sig, y_sig = loaded_datasets_dict[source_ds]
        if len(fp_list) == 1:
            stem = os.path.splitext(os.path.basename(fp_list[0]))[0]
            out_dir = os.path.join(base_results_dir, f"Check_{stem}_on_{source_ds}_{timestamp_sig}")
            scores, _ = execute_inference_pipeline(
                fp_list[0], u_sig, y_sig, source_ds,
                output_dir=out_dir, progress_callback=progress_callback
            )
            output_summaries.append(
                f"[{source_ds}] {stem}: MSE={scores['MSE']:.4f}, R²={scores['R-Squared']:.4f} -> {out_dir}"
            )
        else:
            out_dir = os.path.join(base_results_dir, f"CheckComparison_{len(fp_list)}Models_{source_ds}_{timestamp_sig}")
            ranked = execute_model_comparison_pipeline(
                fp_list, u_sig, y_sig, source_ds,
                output_dir=out_dir, progress_callback=progress_callback
            )
            winner = ranked[0]
            output_summaries.append(
                f"[{source_ds}] Compared {len(fp_list)} models (Best: {winner['label']}, MSE={winner['scores']['MSE']:.4f}) -> {out_dir}"
            )

    return " | ".join(output_summaries)


def generate_results_summary_sheet(base_results_dir="results", loaded_datasets=None):
    # Recursively scans 'base_results_dir' for all completed and in-progress runs
    # and compiles a master CSV and multi-sheet Excel (.xlsx) workbook
    base_results_dir = os.path.abspath(base_results_dir)
    os.makedirs(base_results_dir, exist_ok=True)
    if loaded_datasets is None:
        try:
            loaded_datasets = load_cached_uci_benchmarks()
        except Exception:
            loaded_datasets = {}

    training_rows = []
    fold_records_by_run = {}
    comparison_rows = []

    fold_line_pattern = re.compile(
        r"Fold\s+(\d+)\s*\|\s*Gen:\s*(\d+)/(\d+)\s*\(([^)]+)\)\s*\|\s*Time:\s*([\d.]+)s\s*\|\s*"
        r"MSE:\s*([-\d.infnanINFNAN]+)\s*\|\s*RMSE:\s*([-\d.infnanINFNAN]+)\s*\|\s*MAPE:\s*([-\d.infnanINFNAN]+)%\s*\|\s*"
        r"R2:\s*([-\d.infnanINFNAN]+)\s*\|\s*AIC:\s*([-\d.infnanINFNAN]+)"
        r"(?:\s*\|\s*BIC:\s*([-\d.infnanINFNAN]+))?(?:\s*\|\s*WN%:\s*([-\d.infnanINFNAN]+)%)?"
        r"(?:\s*\|\s*SimFit:\s*([-\d.infnanINFNAN]+)%)?"
    )

    def _safe_float(val_str):
        if val_str is None or val_str == "":
            return ""
        try:
            val = float(val_str)
            return val if np.isfinite(val) else str(val_str)
        except Exception:
            return str(val_str)

    for root, _, files in os.walk(base_results_dir):
        abs_root = os.path.abspath(root)
        folder_name = os.path.basename(abs_root)
        parent_dir = os.path.dirname(abs_root)
        if abs_root == base_results_dir or parent_dir == base_results_dir:
            subfolder_group = "Root"
        else:
            subfolder_group = os.path.relpath(parent_dir, base_results_dir)

        # CASE 1: Completed 10-Fold Run (has training_report.txt)
        if "training_report.txt" in files:
            report_path = os.path.join(abs_root, "training_report.txt")
            row_data = {
                "Subfolder_Group": subfolder_group,
                "Run_Status": "COMPLETED (10/10 Folds)",
                "Folder": folder_name,
                "_root_key": abs_root
            }

            meta_hparams = {}
            meta_sys_mode = "SISO"
            meta_path = os.path.join(abs_root, "run_metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8", errors="replace") as mf:
                        meta_payload = json.load(mf)
                        meta_hparams = meta_payload.get("hyperparameters", {})
                        meta_sys_mode = meta_payload.get("system_mode", meta_hparams.get("system_mode", "SISO"))
                        if "dataset_name" in meta_payload:
                            row_data["Dataset"] = meta_payload["dataset_name"]
                except Exception:
                    pass

            try:
                with open(report_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = [line.strip() for line in f.readlines()]

                section = None
                param_headers = []
                fold_metrics_list = []
                fold_params_map = {}

                for line in lines:
                    if not line:
                        continue
                    if line.startswith("--- Parameters ---"):
                        section = "params"
                        continue
                    elif line.startswith("--- Fold Results ---"):
                        section = "folds"
                        continue
                    elif line.startswith("--- Model Parameters by Fold ---"):
                        section = "fold_params"
                        continue
                    elif line.startswith("--- Average Parameter Values ---"):
                        section = "avg_params"
                        continue
                    elif line.startswith("--- Best Parameters"):
                        section = "best_params"
                        match = re.search(r"Fold\s+(\d+),\s*MSE:\s*([-\d.infnanINFNAN]+)", line)
                        if match:
                            row_data["Best_Fold"] = int(match.group(1))
                            row_data["Best_Fold_MSE"] = _safe_float(match.group(2))
                        continue
                    elif line.startswith("--- Cross-Validation Summary ---"):
                        section = "cv_summary"
                        continue

                    if section is None:
                        if line.startswith("Dataset") and ":" in line:
                            row_data["Dataset"] = line.split(":", 1)[1].strip()
                        elif line.startswith("Model Type") and ":" in line:
                            row_data["Model_Type"] = line.split(":", 1)[1].strip()
                        elif line.startswith("System Mode") and ":" in line:
                            row_data["System_Mode"] = line.split(":", 1)[1].strip()
                        elif line.startswith("Saved Model") and ":" in line:
                            row_data["Saved_Model"] = line.split(":", 1)[1].strip()
                        elif line.startswith("Architecture") and ":" in line:
                            row_data["Architecture"] = line.split(":", 1)[1].strip()
                        elif line.startswith("Execution Time") and ":" in line:
                            row_data["Total_Time_Sec"] = line.split(":", 1)[1].replace("seconds", "").strip()

                    elif section == "params" and ":" in line:
                        k_label, v_str = [x.strip() for x in line.split(":", 1)]
                        row_data[f"_param_{k_label}"] = v_str

                    elif section == "folds":
                        m = fold_line_pattern.search(line)
                        if m:
                            fold_metrics_list.append({
                                "Fold": int(m.group(1)),
                                "Generations": f"{m.group(2)}/{m.group(3)}",
                                "Status": m.group(4).strip(),
                                "Time_Sec": _safe_float(m.group(5)),
                                "MSE": _safe_float(m.group(6)),
                                "RMSE": _safe_float(m.group(7)),
                                "MAPE_Pct": _safe_float(m.group(8)),
                                "R_Squared": _safe_float(m.group(9)),
                                "AIC": _safe_float(m.group(10)),
                                "BIC": _safe_float(m.group(11)),
                                "WN_Pct": _safe_float(m.group(12)),
                                "SimFit_Pct": _safe_float(m.group(13)) if m.lastindex and m.lastindex >= 13 else ""
                            })

                    elif section == "fold_params":
                        if "|" in line:
                            parts = [p.strip() for p in line.split("|")]
                            fold_num_match = re.match(r"^Fold\s+(\d+)$", parts[0])
                            if fold_num_match:
                                f_num = int(fold_num_match.group(1))
                                p_vals = parts[1:]
                                if param_headers:
                                    fold_params_map[f_num] = ", ".join(
                                        [f"{k}={v}" for k, v in zip(param_headers, p_vals)]
                                    )
                                else:
                                    fold_params_map[f_num] = ", ".join(p_vals)
                            elif parts[0] == "Fold":
                                param_headers = parts[1:]

                    elif section == "cv_summary" and ":" in line:
                        k, v = [x.strip() for x in line.split(":", 1)]
                        row_data[f"CV_{k}"] = v

                ds_name = row_data.get("Dataset", "Unknown")
                m_type = row_data.get("Model_Type", meta_hparams.get("model_type", "Unknown"))
                row_data["Model_Type"] = m_type
                row_data.setdefault("System_Mode", meta_sys_mode)
                if "Saved_Model" in row_data:
                    row_data["Saved_Model"] = row_data["Saved_Model"].replace(" / .json", "").strip()
                row_data.update(get_dataset_summary_info(ds_name, loaded_datasets))

                # Check both current PARAMETER_LABELS keys (with symbols) and older report labels
                na_val = meta_hparams.get("na", row_data.get("_param_AR Lags (na)", ""))
                nb_val = meta_hparams.get("nb", row_data.get("_param_Input Lags (nb)", ""))
                nc_val = meta_hparams.get("nc", row_data.get("_param_Error Lags (nc)", "-")) if m_type == "ARMAX" else "-"
                nk_val = meta_hparams.get("nk", row_data.get("_param_Time Delay (nk)", ""))
                mlp_nodes = str(meta_hparams.get("mlp_hidden_nodes", row_data.get("_param_MLP Hidden Neurons", "-"))) if m_type == "MLP" else "-"
                pop_val = meta_hparams.get(
                    "pop_size",
                    row_data.get("_param_Population Size (μ)", row_data.get("_param_Population Size", ""))
                )
                gen_val = meta_hparams.get("num_gen", row_data.get("_param_Max Generations", ""))
                cx_val = meta_hparams.get(
                    "crossover_prob",
                    row_data.get("_param_Crossover Probability (pc)", row_data.get("_param_Crossover Probability", ""))
                )
                sig_val = meta_hparams.get(
                    "mutation_sigma",
                    row_data.get("_param_Initial Mutation Step (σ)", row_data.get("_param_Initial Mutation Step Size", ""))
                )

                row_data.update({
                    "na": na_val, "nb": nb_val, "nc": nc_val, "nk": nk_val,
                    "mlp_hidden_nodes": mlp_nodes, "pop_size": pop_val, "num_gen": gen_val,
                    "crossover_prob": cx_val, "mutation_sigma": sig_val,
                    "Parameter_Group": f"mode={row_data['System_Mode']}, na={na_val}, nb={nb_val}, nk={nk_val} | Pop={pop_val}, Gen={gen_val}, Cx={cx_val}, Sigma={sig_val}"
                })

                for k in list(row_data.keys()):
                    if k.startswith("_param_"):
                        del row_data[k]

                expected_k = int(meta_hparams.get("k_folds", len(fold_metrics_list) if fold_metrics_list else 10))
                row_data["Run_Status"] = f"COMPLETED ({len(fold_metrics_list) or expected_k}/{expected_k} Folds)"

                training_rows.append(row_data)

                run_fold_rows = []
                for fm in fold_metrics_list:
                    f_idx = fm["Fold"]
                    run_fold_rows.append({
                        "Subfolder_Group": subfolder_group,
                        "Dataset": ds_name,
                        "Model_Type": m_type,
                        "System_Mode": row_data["System_Mode"],
                        "Saved_Model": row_data.get("Saved_Model", "").replace(" / .json", ""),
                        "Parameter_Group": row_data["Parameter_Group"],
                        "Fold": f_idx,
                        "Is_Best_Fold": "BEST" if f_idx == row_data.get("Best_Fold") else "",
                        "Generations": fm["Generations"],
                        "Status": fm["Status"],
                        "Time_Sec": fm["Time_Sec"],
                        "MSE": fm["MSE"],
                        "RMSE": fm["RMSE"],
                        "MAPE_Pct": fm["MAPE_Pct"],
                        "R_Squared": fm["R_Squared"],
                        "AIC": fm["AIC"],
                        "BIC": fm["BIC"],
                        "WN_Pct": fm["WN_Pct"],
                        "SimFit_Pct": fm.get("SimFit_Pct", ""),
                        "Fold_Parameters": fold_params_map.get(f_idx, ""),
                        "Folder": folder_name
                    })
                fold_records_by_run[abs_root] = run_fold_rows

            except Exception as e:
                print(f"[!] Skipped report '{report_path}': {e}")

        # CASE 2: Unfinished / In-Progress Run (has checkpoint_fold_*.pkl but no training_report.txt)
        elif any(f.startswith("checkpoint_fold_") and f.endswith(".pkl") for f in files):
            ckpt_files = sorted(
                [f for f in files if f.startswith("checkpoint_fold_") and f.endswith(".pkl")],
                key=lambda x: int(re.search(r"fold_(\d+)", x).group(1)) if re.search(r"fold_(\d+)", x) else 0
            )
            meta_hparams = {}
            ds_name = "Unknown"
            meta_sys_mode = "SISO"
            meta_path = os.path.join(abs_root, "run_metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8", errors="replace") as mf:
                        meta_payload = json.load(mf)
                        meta_hparams = meta_payload.get("hyperparameters", {})
                        ds_name = meta_payload.get("dataset_name", "Unknown")
                        meta_sys_mode = meta_payload.get("system_mode", meta_hparams.get("system_mode", "SISO"))
                except Exception:
                    pass

            partial_fold_rows = []
            partial_mses = []
            partial_times = []
            sig_dict = {}

            for cf_name in ckpt_files:
                cf_path = os.path.join(abs_root, cf_name)
                f_match = re.search(r"fold_(\d+)", cf_name)
                f_num = (int(f_match.group(1)) + 1) if f_match else 1
                try:
                    with open(cf_path, "rb") as pf:
                        st = pickle.load(pf)
                    if not sig_dict and st.get("signature"):
                        sig_dict = st["signature"]
                    gen_reached = int(st.get("generation", 0))
                    is_done = bool(st.get("is_completed", False))
                    el_time = float(st.get("elapsed_time", 0.0))
                    if el_time <= 0.0:
                        el_time = _estimate_legacy_checkpoint_time(cf_path)
                    hof = st.get("hall_of_fame")
                    best_train_mse = float(hof[0].fitness.values[0]) if (hof and len(hof) > 0) else np.nan
                    if np.isfinite(best_train_mse):
                        partial_mses.append(best_train_mse)
                    partial_times.append(el_time)

                    partial_fold_rows.append({
                        "Subfolder_Group": subfolder_group,
                        "Dataset": ds_name,
                        "Model_Type": meta_hparams.get("model_type", sig_dict.get("model_type", "Unknown")),
                        "System_Mode": meta_sys_mode,
                        "Saved_Model": "(In Progress / Unfinished)",
                        "Parameter_Group": "",
                        "Fold": f_num,
                        "Is_Best_Fold": "",
                        "Generations": f"{gen_reached}/{meta_hparams.get('num_gen', '?')}",
                        "Status": "Fold Done" if is_done else f"Paused @ Gen {gen_reached}",
                        "Time_Sec": round(el_time, 2),
                        "MSE": round(best_train_mse, 4) if np.isfinite(best_train_mse) else "",
                        "RMSE": round(float(np.sqrt(best_train_mse)), 4) if np.isfinite(best_train_mse) else "",
                        "MAPE_Pct": "",
                        "R_Squared": "",
                        "AIC": "",
                        "BIC": "",
                        "WN_Pct": "",
                        "SimFit_Pct": "",
                        "Fold_Parameters": "",
                        "Folder": folder_name
                    })
                except Exception:
                    pass

            if partial_fold_rows:
                m_type = meta_hparams.get("model_type", sig_dict.get("model_type", "Unknown"))
                na_val = meta_hparams.get("na", sig_dict.get("na", ""))
                nb_val = meta_hparams.get("nb", sig_dict.get("nb", ""))
                nc_val = meta_hparams.get("nc", sig_dict.get("nc", "-")) if m_type == "ARMAX" else "-"
                nk_val = meta_hparams.get("nk", sig_dict.get("nk", ""))
                mlp_nodes = str(meta_hparams.get("mlp_hidden_nodes", sig_dict.get("mlp_hidden_nodes", "-"))) if m_type == "MLP" else "-"
                pop_val = meta_hparams.get("pop_size", sig_dict.get("pop_size", ""))
                gen_val = meta_hparams.get("num_gen", "")
                cx_val = meta_hparams.get("crossover_prob", "")
                sig_val = meta_hparams.get("mutation_sigma", "")
                param_grp = f"mode={meta_sys_mode}, na={na_val}, nb={nb_val}, nk={nk_val} | Pop={pop_val}, Gen={gen_val}, Cx={cx_val}, Sigma={sig_val}"

                for pfr in partial_fold_rows:
                    pfr["Parameter_Group"] = param_grp

                expected_k = int(meta_hparams.get("k_folds", sig_dict.get("k_folds", 10)))
                row_data = {
                    "Subfolder_Group": subfolder_group,
                    "Run_Status": f"IN PROGRESS ({len(ckpt_files)}/{expected_k} Folds)",
                    "Dataset": ds_name,
                    "Model_Type": m_type,
                    "System_Mode": meta_sys_mode,
                    "na": na_val, "nb": nb_val, "nc": nc_val, "nk": nk_val,
                    "mlp_hidden_nodes": mlp_nodes, "pop_size": pop_val, "num_gen": gen_val,
                    "crossover_prob": cx_val, "mutation_sigma": sig_val,
                    "Parameter_Group": param_grp,
                    "Best_Fold": "",
                    "Best_Fold_MSE": round(float(np.min(partial_mses)), 4) if partial_mses else "",
                    "CV_MSE": f"{np.mean(partial_mses):.4f} +/- {np.std(partial_mses):.4f} (Train Ckpt)" if partial_mses else "",
                    "Total_Time_Sec": round(float(np.sum(partial_times)), 2) if partial_times else "",
                    "Saved_Model": "(Unfinished)",
                    "Folder": folder_name,
                    "_root_key": abs_root
                }
                row_data.update(get_dataset_summary_info(ds_name, loaded_datasets))
                training_rows.append(row_data)
                fold_records_by_run[abs_root] = partial_fold_rows

        # CASE 3: Multi-Model Comparison CSVs
        if "comparison_metrics.csv" in files:
            comp_csv_path = os.path.join(abs_root, "comparison_metrics.csv")
            try:
                with open(comp_csv_path, "r", encoding="utf-8", errors="replace") as f:
                    comp_lines = [line.strip() for line in f.readlines() if line.strip()]
                if len(comp_lines) > 1:
                    headers = comp_lines[0].split(",")
                    for row_line in comp_lines[1:]:
                        vals = re.findall(r'(?:[^,"]|"(?:\\.|[^"])*")+', row_line)
                        vals = [v.strip('"') for v in vals]
                        comp_dict = {
                            "Subfolder_Group": subfolder_group,
                            "Comparison_Folder": folder_name
                        }
                        for h, val in zip(headers, vals):
                            comp_dict[h] = val
                        comparison_rows.append(comp_dict)
            except Exception as e:
                print(f"[!] Skipped comparison CSV '{comp_csv_path}': {e}")

    if not training_rows and not comparison_rows:
        raise RuntimeError(f"No completed or in-progress runs found in '{base_results_dir}'.")

    df_train = pd.DataFrame(training_rows) if training_rows else pd.DataFrame()
    ordered_fold_rows = []

    if not df_train.empty:
        sort_keys = [c for c in ["Subfolder_Group", "Dataset", "Model_Type", "Folder"] if c in df_train.columns]
        df_train = df_train.sort_values(by=sort_keys, kind="mergesort").reset_index(drop=True)

        for rk in df_train.get("_root_key", []):
            f_rows = fold_records_by_run.get(rk, [])
            if f_rows:
                ordered_fold_rows.extend(f_rows)
                ordered_fold_rows.append({k: "" for k in f_rows[0].keys()})

        if "_root_key" in df_train.columns:
            df_train = df_train.drop(columns=["_root_key"])

        ordered_cols = [
            "Subfolder_Group", "Run_Status", "Dataset", "Dataset_Samples", "IO_Channels", "Input_Range", "Output_Mean_Std", "Output_Min_Max",
            "Parameter_Group", "Model_Type", "System_Mode", "na", "nb", "nc", "nk", "mlp_hidden_nodes",
            "pop_size", "num_gen", "crossover_prob", "mutation_sigma",
            "Best_Fold", "Best_Fold_MSE", "CV_MSE", "CV_RMSE", "CV_MAPE", "CV_R-Squared",
            "CV_AIC", "CV_AICc", "CV_BIC", "CV_FPE", "CV_White-Noise ACF Pass Rate", "CV_Autonomous Simulation Fit",
            "CV_Generations Run", "CV_Execution Time", "Total_Time_Sec", "Saved_Model", "Folder"
        ]
        existing_cols = [c for c in ordered_cols if c in df_train.columns]
        extra_cols = [c for c in df_train.columns if c not in existing_cols]
        df_train = df_train[existing_cols + extra_cols]

    df_folds = pd.DataFrame(ordered_fold_rows) if ordered_fold_rows else pd.DataFrame()
    df_comp = pd.DataFrame(comparison_rows) if comparison_rows else pd.DataFrame()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_out_path = os.path.join(base_results_dir, f"results_summary_{timestamp}.csv")

    with open(csv_out_path, "w", encoding="utf-8") as f:
        if not df_train.empty:
            f.write("=== 1. TRAINING RUNS SUMMARY (GROUPED BY SUBFOLDER, DATASET & PARAMETERS) ===\n")
            df_train.to_csv(f, index=False, lineterminator="\n")
            f.write("\n")
        if not df_comp.empty:
            f.write("=== 2. MULTI-MODEL HEAD-TO-HEAD COMPARISONS ===\n")
            df_comp.to_csv(f, index=False, lineterminator="\n")
            f.write("\n")
        if not df_folds.empty:
            f.write("=== 3. 10-FOLD CROSS-VALIDATION BREAKDOWN PER MODEL ===\n")
            df_folds.to_csv(f, index=False, lineterminator="\n")

    xlsx_out_path = os.path.join(base_results_dir, f"results_summary_{timestamp}.xlsx")
    try:
        with pd.ExcelWriter(xlsx_out_path) as writer:
            if not df_train.empty:
                df_train.to_excel(writer, sheet_name="Summary_Grouped_By_Params", index=False)
            if not df_folds.empty:
                df_folds.to_excel(writer, sheet_name="10Fold_Per_Model_Breakdown", index=False)
            if not df_comp.empty:
                df_comp.to_excel(writer, sheet_name="Model_Comparisons", index=False)

            for sheet_name, worksheet in writer.sheets.items():
                for col in worksheet.columns:
                    max_len = max((len(str(cell.value or "")) for cell in col), default=10)
                    col_letter = getattr(col[0], "column_letter", None)
                    if not col_letter and hasattr(col[0], "column"):
                        try:
                            from openpyxl.utils import get_column_letter
                            col_letter = get_column_letter(col[0].column)
                        except Exception:
                            col_letter = None
                    if col_letter and hasattr(worksheet, "column_dimensions") and col_letter in worksheet.column_dimensions:
                        worksheet.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 55)
        return xlsx_out_path
    except Exception:
        return csv_out_path