"""
diagnostic_plotter.py

Generates plots for model validation, tracking convergence,
cross-validation stability, and residual analysis.
"""

import matplotlib
# Enforce headless rendering for background plotting threads
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def _ensure_output_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def generate_convergence_plot(ep_logbook, dataset_name, run_index, output_dir="plots", run_label="Fold", y_axis_label="Mean Squared Error", best_curve_label="Best Fitness (MSE)", title_suffix=""):
    """
    Plots the best and average population fitness across generations.
    """
    _ensure_output_directory(output_dir)

    generations = ep_logbook.select("gen")
    minimum_fitness = ep_logbook.select("min")
    average_fitness = ep_logbook.select("avg")

    plt.figure(figsize=(10, 5))
    plt.plot(generations, minimum_fitness, label=best_curve_label, color="red", linewidth=2)
    plt.plot(generations, average_fitness, label="Average Fitness", color="blue", linestyle="--")
    plt.title(f"Optimization Convergence\n{dataset_name} - {run_label} {run_index}{title_suffix}")
    plt.xlabel("Generation")
    plt.ylabel(y_axis_label)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    filepath = os.path.join(output_dir, f"convergence_{dataset_name}_{run_label.lower()}_{run_index}.png")
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def generate_prediction_tracking_plot(actual_targets, predicted_outputs, dataset_name, fold_num, output_dir="plots", title_suffix=""):
    r"""
    Plots the time-domain prediction tracking alongside a correlation scatter.
    """
    _ensure_output_directory(output_dir)

    display_limit = min(len(actual_targets), 200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [2.2, 1]})

    ax1.plot(actual_targets[:display_limit], label="Actual Output y(t)", color="black", linewidth=1.5)
    ax1.plot(predicted_outputs[:display_limit], label=r"Predicted Output $\hat{y}(t|\theta)$", color="orange", linestyle="--", linewidth=1.5)
    ax1.set_title(f"Time-Series Tracking (First {display_limit} Steps)")
    ax1.set_xlabel("Time Step (t)")
    ax1.set_ylabel("System Output")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.scatter(actual_targets, predicted_outputs, alpha=0.5, color="teal", s=15, label="Test Samples")
    min_bound = min(np.min(actual_targets), np.min(predicted_outputs))
    max_bound = max(np.max(actual_targets), np.max(predicted_outputs))
    ax2.plot([min_bound, max_bound], [min_bound, max_bound], color="red", linestyle="--", linewidth=1.5, label="Ideal Fit (y = y_hat)")
    ax2.set_title("Actual vs. Predicted Plot")
    ax2.set_xlabel("Actual Output y(t)")
    ax2.set_ylabel(r"Predicted Output $\hat{y}(t|\theta)$")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Model Performance: {dataset_name} - Fold {fold_num}{title_suffix}")

    filepath = os.path.join(output_dir, f"predictions_{dataset_name}_fold_{fold_num}.png")
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def generate_residual_diagnostics_plot(actual_targets, predicted_outputs, dataset_name, fold_num, output_dir="plots", title_suffix=""):
    """
    Generates autocorrelation and distribution plots for error residuals.
    """
    _ensure_output_directory(output_dir)

    residual_array = actual_targets - predicted_outputs
    sample_count = len(residual_array)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.scatter(range(sample_count), residual_array, alpha=0.5, color="purple", s=10)
    ax1.axhline(0, color='black', linestyle='--')
    ax1.set_title("Prediction Errors over Time")
    ax1.set_xlabel("Time Step (t)")
    ax1.set_ylabel(r"Error $\epsilon(t, \theta)$")
    ax1.grid(True, alpha=0.3)

    ax2.hist(residual_array, bins=30, color="purple", alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='black', linestyle='--')
    ax2.set_title("Distribution of Errors")
    ax2.set_xlabel("Error Value")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    max_lag_limit = min(40, sample_count - 1)
    centered_residuals = residual_array - np.mean(residual_array)
    residual_variance = np.sum(centered_residuals ** 2)

    if residual_variance > 0 and max_lag_limit > 0:
        acf_full_sequence = np.correlate(centered_residuals, centered_residuals, mode='full') / residual_variance
        acf_positive_lags = acf_full_sequence[sample_count - 1: sample_count + max_lag_limit]
        lag_indices = np.arange(len(acf_positive_lags))

        ax3.stem(lag_indices, acf_positive_lags, linefmt='purple', markerfmt='o', basefmt='black')

        confidence_bound = 1.96 / np.sqrt(sample_count)
        ax3.axhline(confidence_bound, color='red', linestyle='--', label='95% Confidence Bound')
        ax3.axhline(-confidence_bound, color='red', linestyle='--')
        ax3.set_ylim(-1.05, 1.05)
        ax3.legend(loc='upper right')

    ax3.set_title("Error Autocorrelation")
    ax3.set_xlabel("Lag")
    ax3.set_ylabel("Correlation")
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f"Residual Analysis: {dataset_name} - Fold {fold_num}{title_suffix}")

    filepath = os.path.join(output_dir, f"residuals_{dataset_name}_fold_{fold_num}.png")
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def generate_cross_validation_summary(fold_metrics_dict, fold_parameter_matrix, parameter_names, dataset_name, output_dir="plots"):
    """
    Visualizes MSE consistency and parameter variance across the k-folds.
    """
    _ensure_output_directory(output_dir)

    mse_array = fold_metrics_dict["MSE"]
    fold_indices = np.arange(1, len(mse_array) + 1)
    overall_mean_mse = np.mean(mse_array)

    mean_theta_weights = np.mean(fold_parameter_matrix, axis=0)
    std_theta_weights = np.std(fold_parameter_matrix, axis=0)
    parameter_x_positions = np.arange(len(parameter_names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.bar(fold_indices, mse_array, color="#3B8ED0", alpha=0.85, edgecolor="black", label="Fold Test MSE")
    ax1.axhline(overall_mean_mse, color="red", linestyle="--", linewidth=2, label=f"Mean MSE ({overall_mean_mse:.4f})")
    ax1.set_title("Validation MSE Across Folds")
    ax1.set_xlabel("Fold Number")
    ax1.set_ylabel("Mean Squared Error (MSE)")
    ax1.set_xticks(fold_indices)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(parameter_x_positions, mean_theta_weights, yerr=std_theta_weights, fmt='o', color='darkgreen', ecolor='black', elinewidth=2, capsize=5, markersize=7, label="Mean +/- 1 SD")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_title("Parameter Variance Across Folds")
    ax2.set_xlabel("Model Parameter")
    ax2.set_ylabel("Parameter Value")
    ax2.set_xticks(parameter_x_positions)
    ax2.set_xticklabels(parameter_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Cross-Validation Summary: {dataset_name}")

    filepath = os.path.join(output_dir, f"cv_summary_{dataset_name}.png")
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def generate_benchmark_boxplots(benchmark_scores_dict, output_dir="results_benchmarks"):
    """
    Generates a log-scaled boxplot comparing objective value distributions across benchmarks.
    """
    _ensure_output_directory(output_dir)

    function_names = list(benchmark_scores_dict.keys())

    # Lower-bound strictly zero scores to 1e-12 for valid log-scale visualization
    score_distributions = [np.maximum(benchmark_scores_dict[func], 1e-12) for func in function_names]

    plt.figure(figsize=(10, 5))
    boxplot_elements = plt.boxplot(score_distributions, tick_labels=function_names, patch_artist=True)

    for box in boxplot_elements['boxes']:
        box.set(facecolor='#3B8ED0', alpha=0.75)

    plt.yscale('log')
    plt.title("Optimization Performance on Test Functions")
    plt.xlabel("Test Function")
    plt.ylabel("Final Objective Value [Log Scale]")
    plt.grid(True, alpha=0.3)

    filepath = os.path.join(output_dir, "benchmark_comparison_boxplot.png")
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()