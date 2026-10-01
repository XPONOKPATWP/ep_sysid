"""
training_pipeline.py

Runs the training process, including data prep,
optimization, and cross-validation for the ARMAX model.
"""

import json
import pickle
import os
import time
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from siso_data_parser import load_cached_uci_benchmarks
from ep_algorithm_core import execute_evolutionary_programming
from validation_metrics import generate_validation_report
from config_handler import load_hyperparameters, get_parameter_label
from diagnostic_plotter import (
    generate_convergence_plot,
    generate_prediction_tracking_plot,
    generate_residual_diagnostics_plot,
    generate_cross_validation_summary
)
from armax_simulator import simulate_armax_system


def enforce_reproducibility_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)


def _parse_serialized_model(model_filepath):
    """
    Loads a previously saved model configuration and parameters.
    """
    file_extension = os.path.splitext(model_filepath)[1].lower()

    if file_extension == ".json":
        with open(model_filepath, "r") as file:
            parsed_data = json.load(file)

        required_keys = {"structure", "best_fold_parameters"}
        if not isinstance(parsed_data, dict) or not required_keys.issubset(parsed_data.keys()):
            raise ValueError("Invalid JSON format.")

        architecture = parsed_data["structure"]
        na, nb, nc = int(architecture["na"]), int(architecture["nb"]), int(architecture["nc"])
        parameter_names = [f"a{i + 1}" for i in range(na)] + [f"b{i + 1}" for i in range(nb)] + [f"c{i + 1}" for i in range(nc)]

        best_weights_dict = parsed_data["best_fold_parameters"]
        mean_weights_dict = parsed_data.get("mean_10fold_parameters", best_weights_dict)

        best_theta_vector = [float(best_weights_dict[k]) for k in parameter_names]
        mean_theta_vector = [float(mean_weights_dict[k]) for k in parameter_names]

        # Reconstruct scikit-learn standard scalers manually from saved moments
        u_scaler, y_scaler = None, None
        if "scaler_u" in parsed_data and "scaler_y" in parsed_data:
            u_scaler = StandardScaler()
            u_scaler.mean_ = np.array([float(parsed_data["scaler_u"]["mean"])])
            u_scaler.scale_ = np.array([float(parsed_data["scaler_u"]["std"])])
            u_scaler.var_ = u_scaler.scale_ ** 2
            u_scaler.n_features_in_ = 1

            y_scaler = StandardScaler()
            y_scaler.mean_ = np.array([float(parsed_data["scaler_y"]["mean"])])
            y_scaler.scale_ = np.array([float(parsed_data["scaler_y"]["std"])])
            y_scaler.var_ = y_scaler.scale_ ** 2
            y_scaler.n_features_in_ = 1

        return {
            "dataset": parsed_data.get("dataset", "Unknown_Source"),
            "structure": architecture,
            "param_names": parameter_names,
            "best_fold_index": int(parsed_data.get("best_fold", 1)),
            "best_fold_theta": best_theta_vector,
            "mean_theta_10fold": mean_theta_vector,
            "scaler_u": u_scaler,
            "scaler_y": y_scaler
        }

    else:
        with open(model_filepath, "rb") as file:
            model_payload = pickle.load(file)

        required_keys = {"structure", "best_fold_theta", "scaler_u", "scaler_y"}
        if not isinstance(model_payload, dict) or not required_keys.issubset(model_payload.keys()):
            raise ValueError("Invalid PKL file format.")
        return model_payload


def execute_inference_pipeline(model_filepath, u_signal, y_signal, target_dataset_name, output_dir="results_inference", progress_callback=None):
    """
    Runs a saved ARMAX model on a target dataset and evaluates performance.
    """
    execution_start = time.perf_counter()

    if progress_callback:
        progress_callback(f"Loading model from {os.path.basename(model_filepath)}...")

    model_payload = _parse_serialized_model(model_filepath)
    os.makedirs(output_dir, exist_ok=True)

    architecture = model_payload["structure"]
    na, nb, nc, nk = int(architecture["na"]), int(architecture["nb"]), int(architecture["nc"]), int(architecture["nk"])
    total_parameters = na + nb + nc
    parameter_names = model_payload.get(
        "param_names",
        [f"a{i + 1}" for i in range(na)] + [f"b{i + 1}" for i in range(nb)] + [f"c{i + 1}" for i in range(nc)]
    )

    source_dataset = model_payload.get("dataset", "Unknown_Source")
    winning_fold_index = int(model_payload.get("best_fold_index", 1))
    best_theta_weights = np.asarray(model_payload["best_fold_theta"], dtype=float)
    mean_theta_weights = np.asarray(model_payload.get("mean_theta_10fold", best_theta_weights), dtype=float)

    # Retrieve the exact scaling moments used during training to apply to the target data
    u_scaler = model_payload.get("scaler_u")
    y_scaler = model_payload.get("scaler_y")

    if u_scaler is None or y_scaler is None:
        u_scaler = StandardScaler().fit(u_signal.reshape(-1, 1))
        y_scaler = StandardScaler().fit(y_signal.reshape(-1, 1))

    if progress_callback:
        progress_callback(f"Simulating on {target_dataset_name}...")

    u_scaled = u_scaler.transform(u_signal.reshape(-1, 1)).flatten()
    y_scaled = y_scaler.transform(y_signal.reshape(-1, 1)).flatten()

    predicted_y_scaled, max_delay_lag = simulate_armax_system(u_scaled, y_scaled, best_theta_weights, na, nb, nc, nk)
    if not np.isfinite(predicted_y_scaled).all():
        raise RuntimeError("Model produced unstable predictions.")

    y_actual_unscaled = y_scaler.inverse_transform(y_scaled[max_delay_lag:].reshape(-1, 1)).flatten()
    y_pred_unscaled = y_scaler.inverse_transform(predicted_y_scaled[max_delay_lag:].reshape(-1, 1)).flatten()
    best_validation_scores = generate_validation_report(y_actual_unscaled, y_pred_unscaled, model_parameter_count=total_parameters)

    predicted_mean_scaled, _ = simulate_armax_system(u_scaled, y_scaled, mean_theta_weights, na, nb, nc, nk)
    mean_validation_scores = None
    if np.isfinite(predicted_mean_scaled).all():
        y_mean_unscaled = y_scaler.inverse_transform(predicted_mean_scaled[max_delay_lag:].reshape(-1, 1)).flatten()
        mean_validation_scores = generate_validation_report(y_actual_unscaled, y_mean_unscaled, model_parameter_count=total_parameters)

    chart_suffix = f" (Loaded Model)"
    generate_prediction_tracking_plot(y_actual_unscaled, y_pred_unscaled, target_dataset_name, winning_fold_index, output_dir=output_dir, title_suffix=chart_suffix)
    generate_residual_diagnostics_plot(y_actual_unscaled, y_pred_unscaled, target_dataset_name, winning_fold_index, output_dir=output_dir, title_suffix=chart_suffix)

    residual_sequence = y_actual_unscaled - y_pred_unscaled
    time_indices = np.arange(max_delay_lag, len(y_signal))
    csv_data = np.column_stack((time_indices, y_actual_unscaled, y_pred_unscaled, residual_sequence))
    csv_filepath = os.path.join(output_dir, "predictions.csv")
    np.savetxt(
        csv_filepath,
        csv_data,
        delimiter=",",
        header="Time_Step,Actual_y,Predicted_y_hat,Error",
        comments="",
        fmt=["%d", "%.6f", "%.6f", "%.6f"]
    )

    execution_elapsed = time.perf_counter() - execution_start

    report_filepath = os.path.join(output_dir, "inference_report.txt")
    with open(report_filepath, "w") as file:
        file.write("=== Model Inference Report ===\n")
        file.write(f"Model File           : {os.path.abspath(model_filepath)}\n")
        file.write(f"Trained On           : {source_dataset} (Fold {winning_fold_index:02d})\n")
        file.write(f"Evaluated On         : {target_dataset_name}\n")
        file.write(f"Architecture         : na={na}, nb={nb}, nc={nc}, nk={nk}\n")
        file.write(f"Time Taken           : {execution_elapsed:.4f} seconds\n")

        file.write("\n--- Best-Fold Parameters ---\n")
        for name, val in zip(parameter_names, best_theta_weights):
            file.write(f"{name}: {val:>8.4f}\n")

        file.write("\n--- Performance Metrics ---\n")
        file.write(f"MSE       : {best_validation_scores['MSE']:.4f}\n")
        file.write(f"RMSE      : {best_validation_scores['RMSE']:.4f}\n")
        file.write(f"MAPE      : {best_validation_scores['MAPE']:.2f}%\n")
        file.write(f"R-Squared : {best_validation_scores['R-Squared']:.4f}\n")
        file.write(f"AIC       : {best_validation_scores['AIC']:.4f}\n")

    print(f"\n[*] Inference complete. Results saved to: {output_dir}")
    return best_validation_scores, model_payload


def execute_kfold_training_pipeline(u_signal, y_signal, dataset_name, hyperparameters, output_dir="plots", progress_callback=None):
    """
    Executes the training and evaluation loop using 10-Fold cross-validation.
    """
    enforce_reproducibility_seed(42)
    pipeline_start_time = time.perf_counter()

    os.makedirs(output_dir, exist_ok=True)

    active_config = hyperparameters.copy()
    na, nb, nc, nk = active_config["na"], active_config["nb"], active_config["nc"], active_config["nk"]
    max_generations = active_config["num_gen"]

    print(f"\n{'=' * 60}")
    print(f"Training on Dataset: {dataset_name}")
    print(f"Structure: na={na}, nb={nb}, nc={nc}, nk={nk}")
    print(f"{'=' * 60}")

    # Standard temporal shuffle=False is maintained to preserve time-series integrity
    cross_val_splits = KFold(n_splits=10, shuffle=False)

    fold_telemetry = {
        "MSE": [], "RMSE": [], "MAPE": [], "R-Squared": [], "AIC": [],
        "Generations": [], "Execution_Time_Sec": []
    }
    fold_theta_matrices = []
    fold_scaler_objects = []
    fold_ep_logbooks = []
    fold_prediction_tuples = []
    console_report_lines = []

    for fold_idx, (train_indices, test_indices) in enumerate(cross_val_splits.split(u_signal)):
        fold_start = time.perf_counter()
        current_fold = fold_idx + 1

        if progress_callback:
            progress_callback(f"Running {dataset_name}: Fold {current_fold}/10...")

        u_scaler = StandardScaler()
        y_scaler = StandardScaler()

        # Fit scale only on the training slice to prevent data leakage into the test set
        u_train_scaled = u_scaler.fit_transform(u_signal[train_indices].reshape(-1, 1)).flatten()
        y_train_scaled = y_scaler.fit_transform(y_signal[train_indices].reshape(-1, 1)).flatten()
        u_test_scaled = u_scaler.transform(u_signal[test_indices].reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(y_signal[test_indices].reshape(-1, 1)).flatten()

        active_config["checkpoint_file"] = f"checkpoint_{dataset_name}_fold_{fold_idx}.pkl"

        optimal_theta, ep_logbook = execute_evolutionary_programming(u_train_scaled, y_train_scaled, active_config)

        fold_theta_matrices.append(optimal_theta)
        fold_scaler_objects.append((u_scaler, y_scaler))
        fold_ep_logbooks.append(ep_logbook)

        final_generation_reached = ep_logbook.select("gen")[-1]
        fold_elapsed_time = time.perf_counter() - fold_start

        fold_telemetry["Generations"].append(final_generation_reached)
        fold_telemetry["Execution_Time_Sec"].append(fold_elapsed_time)
        convergence_status = "Stopped Early" if final_generation_reached < max_generations else "Max Gen Reached"

        # Validate the winning individual against the unseen test fold
        predicted_y_scaled, max_delay_lag = simulate_armax_system(u_test_scaled, y_test_scaled, optimal_theta, na, nb, nc, nk)

        # Invert scaling back to original engineering units for accurate error reporting
        y_actual_unscaled = y_scaler.inverse_transform(y_test_scaled[max_delay_lag:].reshape(-1, 1)).flatten()
        y_pred_unscaled = y_scaler.inverse_transform(predicted_y_scaled[max_delay_lag:].reshape(-1, 1)).flatten()
        fold_prediction_tuples.append((y_actual_unscaled, y_pred_unscaled))

        validation_scores = generate_validation_report(y_actual_unscaled, y_pred_unscaled, model_parameter_count=na + nb + nc)
        for key, value in validation_scores.items():
            fold_telemetry[key].append(value)

        status_line = (
            f"Fold {current_fold:02d} | Gen: {final_generation_reached:>3d}/{max_generations} ({convergence_status:<15}) | "
            f"Time: {fold_elapsed_time:>5.2f}s | "
            f"MSE: {validation_scores['MSE']:>8.4f} | "
            f"RMSE: {validation_scores['RMSE']:>8.4f} | "
            f"MAPE: {validation_scores['MAPE']:>5.2f}% | "
            f"R2: {validation_scores['R-Squared']:>7.4f} | "
            f"AIC: {validation_scores['AIC']:>8.4f}")
        print(status_line)
        console_report_lines.append(status_line)

    pipeline_elapsed_time = time.perf_counter() - pipeline_start_time

    parameter_names = [f"a{i + 1}" for i in range(na)] + [f"b{i + 1}" for i in range(nb)] + [f"c{i + 1}" for i in range(nc)]
    mean_theta_array = np.mean(fold_theta_matrices, axis=0)
    std_theta_array = np.std(fold_theta_matrices, axis=0)

    best_fold_index = int(np.argmin(fold_telemetry["MSE"]))
    winning_fold_number = best_fold_index + 1
    best_u_scaler, best_y_scaler = fold_scaler_objects[best_fold_index]
    best_actual_targets, best_predicted_outputs = fold_prediction_tuples[best_fold_index]

    chart_suffix = " (Best of 10 Folds)"
    generate_convergence_plot(fold_ep_logbooks[best_fold_index], dataset_name, winning_fold_number, output_dir=output_dir, title_suffix=chart_suffix)
    generate_prediction_tracking_plot(best_actual_targets, best_predicted_outputs, dataset_name, winning_fold_number, output_dir=output_dir, title_suffix=chart_suffix)
    generate_residual_diagnostics_plot(best_actual_targets, best_predicted_outputs, dataset_name, winning_fold_number, output_dir=output_dir, title_suffix=chart_suffix)
    generate_cross_validation_summary(fold_telemetry, fold_theta_matrices, parameter_names, dataset_name, output_dir=output_dir)

    summary_statistics = []
    for metric, value_array in fold_telemetry.items():
        mean_val, std_val = np.mean(value_array), np.std(value_array)
        if metric == "MAPE":
            summary_statistics.append(f"MAPE: {mean_val:.2f}% +/- {std_val:.2f}%")
        elif metric == "Generations":
            summary_statistics.append(f"Generations Run: {mean_val:.1f} +/- {std_val:.1f} (Max: {max_generations})")
        elif metric == "Execution_Time_Sec":
            summary_statistics.append(f"Execution Time: {mean_val:.2f}s +/- {std_val:.2f}s")
        else:
            summary_statistics.append(f"{metric}: {mean_val:.4f} +/- {std_val:.4f}")

    model_binary_payload = {
        "dataset": dataset_name,
        "structure": {"na": na, "nb": nb, "nc": nc, "nk": nk},
        "param_names": parameter_names,
        "best_fold_index": winning_fold_number,
        "best_fold_theta": fold_theta_matrices[best_fold_index],
        "best_fold_mse": float(fold_telemetry["MSE"][best_fold_index]),
        "mean_theta_10fold": mean_theta_array.tolist(),
        "std_theta_10fold": std_theta_array.tolist(),
        "all_fold_thetas": fold_theta_matrices,
        "scaler_u": best_u_scaler,
        "scaler_y": best_y_scaler
    }
    pkl_filepath = os.path.join(output_dir, "identified_armax_model.pkl")
    with open(pkl_filepath, "wb") as file:
        pickle.dump(model_binary_payload, file)

    model_json_payload = {
        "dataset": dataset_name,
        "structure": {"na": na, "nb": nb, "nc": nc, "nk": nk},
        "best_fold": winning_fold_number,
        "best_fold_mse": float(fold_telemetry["MSE"][best_fold_index]),
        "best_fold_parameters": dict(zip(parameter_names, [float(v) for v in fold_theta_matrices[best_fold_index]])),
        "mean_10fold_parameters": dict(zip(parameter_names, [float(v) for v in mean_theta_array])),
        "std_10fold_parameters": dict(zip(parameter_names, [float(v) for v in std_theta_array])),
        "scaler_u": {
            "mean": float(best_u_scaler.mean_[0]),
            "std": float(best_u_scaler.scale_[0])
        },
        "scaler_y": {
            "mean": float(best_y_scaler.mean_[0]),
            "std": float(best_y_scaler.scale_[0])
        }
    }
    json_filepath = os.path.join(output_dir, "identified_armax_model.json")
    with open(json_filepath, "w") as file:
        json.dump(model_json_payload, file, indent=4)

    report_filepath = os.path.join(output_dir, "training_report.txt")
    with open(report_filepath, "w") as file:
        file.write("=== Training Report ===\n")
        file.write(f"Dataset         : {dataset_name}\n")
        file.write(f"Architecture    : na={na}, nb={nb}, nc={nc}, nk={nk}\n")
        file.write(f"Execution Time  : {pipeline_elapsed_time:.2f} seconds\n")

        file.write("\n--- Parameters ---\n")
        for key, val in hyperparameters.items():
            if key in ("checkpoint_file", "mutation_mu"):
                continue
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

    print(f"\n[*] Training complete. Report and plots saved to: {output_dir}")
    print(f"\n--- Summary for {dataset_name} ---")
    print("\n".join(summary_statistics))


if __name__ == "__main__":
    benchmark_datasets = load_cached_uci_benchmarks()
    active_hyperparameters = load_hyperparameters("config.json")

    for dataset, (input_signal, output_signal) in benchmark_datasets.items():
        execute_kfold_training_pipeline(input_signal, output_signal, dataset_name=dataset, hyperparameters=active_hyperparameters)