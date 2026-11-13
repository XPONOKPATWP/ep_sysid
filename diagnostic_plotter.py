"""
diagnostic_plotter.py

Plotting routines for convergence, time-series tracking, extrema zoom,
residual diagnostics, cross-validation stability, and model comparisons.
Supports 1D (SISO) and 2D (MISO, SIMO, MIMO) signal arrays.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import threading
import matplotlib.ticker as ticker
from scipy import signal as sp_signal

def _ensure_output_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def _extract_1d_channel(signal_array, channel_idx=0):
    # Extracts primary 1D channel from 1D or 2D (N, C) arrays so matplotlib 1D calls never crash
    if signal_array is None:
        return np.empty((0,), dtype=float)
    arr = np.asarray(signal_array, dtype=float)
    if arr.ndim <= 1 or arr.shape[1] == 0:
        return arr.flatten()
    safe_col = min(max(0, int(channel_idx)), arr.shape[1] - 1)
    return arr[:, safe_col].flatten()


# Matplotlib pyplot is not thread-safe; locking prevents race conditions during concurrent jobs
_PLOT_LOCK = threading.Lock()


def _safe_float(val, fallback=np.nan):
    if val is None:
        return fallback
    try:
        f = float(val)
        return f if np.isfinite(f) else fallback
    except (ValueError, TypeError):
        return fallback


def generate_convergence_plot(ep_logbook, dataset_name, run_index, output_dir="plots", run_label="Fold", y_axis_label="Mean Squared Error", best_curve_label="Best Fitness (MSE)", title_suffix=""):
    # Plots training convergence over generations (min and average fitness)
    with _PLOT_LOCK:
        try:
            _ensure_output_directory(output_dir)

            if not ep_logbook or len(ep_logbook) == 0:
                return

            generations = ep_logbook.select("gen")
            min_raw = np.nan_to_num(np.asarray(ep_logbook.select("min"), dtype=float), nan=1e-12, posinf=1e6, neginf=1e-12)
            minimum_fitness = np.clip(min_raw, 1e-12, 1e9)
            avg_raw = np.nan_to_num(np.asarray(ep_logbook.select("avg"), dtype=float), nan=1e-12, posinf=1e6, neginf=1e-12)
            average_fitness = np.clip(avg_raw, 1e-12, 1e9)

            fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
            ax.plot(generations, minimum_fitness, label=best_curve_label, color="#D62728", linewidth=2.2)
            ax.plot(generations, average_fitness, label="Population Average Fitness", color="#1F77B4", linestyle="--", linewidth=1.5, alpha=0.85)
            ax.set_title(f"Optimization Convergence — {dataset_name} ({run_label} {run_index}){title_suffix}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Generation", fontsize=11)
            ax.set_ylabel(y_axis_label, fontsize=11)
            ax.set_yscale("log")
            # Format log ticks as simple numbers so matplotlib doesn't crash on thread conflicts
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.4g}"))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.legend(frameon=True, fontsize=10)
            ax.grid(True, which="both", linestyle="--", alpha=0.35)

            filepath = os.path.join(output_dir, f"convergence_{dataset_name}_{run_label.lower()}_{run_index}.png")
            fig.savefig(filepath, bbox_inches="tight", dpi=300)
        finally:
            plt.close("all")


def generate_prediction_tracking_plot(actual_targets, predicted_outputs, dataset_name, fold_num, output_dir="plots", title_suffix=""):
    # Saves time-series tracking, actual vs. predicted parity scatter, and extrema zoom (max/min)
    with _PLOT_LOCK:
        try:
            _ensure_output_directory(output_dir)

            actual_targets = _extract_1d_channel(actual_targets, 0)
            predicted_outputs = _extract_1d_channel(predicted_outputs, 0)
            common_len = min(len(actual_targets), len(predicted_outputs))
            if common_len == 0:
                return
            actual_targets = np.nan_to_num(actual_targets[:common_len], nan=0.0)
            predicted_outputs = predicted_outputs[:common_len]

            if not np.all(np.isfinite(predicted_outputs)):
                # If the candidate model diverged and outputs contain Inf/NaN, clamp them to a safe finite range
                # so parity scatter and tracking plots don't crash with "Axis limits cannot be NaN or Inf"
                valid_act = actual_targets[np.isfinite(actual_targets)]
                act_std = float(np.std(valid_act)) if len(valid_act) > 0 else 1.0
                act_mean = float(np.mean(valid_act)) if len(valid_act) > 0 else 0.0
                clamp_limit = max(100.0, abs(act_mean) + 10.0 * max(1e-3, act_std))
                predicted_outputs = np.nan_to_num(predicted_outputs, nan=act_mean, posinf=clamp_limit, neginf=-clamp_limit)

            display_limit = min(len(actual_targets), 200)

            # 1. Time-series tracking
            fig1, ax1 = plt.subplots(figsize=(12, 5), dpi=300)
            time_steps = np.arange(display_limit)
            ax1.plot(time_steps, actual_targets[:display_limit], label="Actual Output $y(t)$", color="#111111", linewidth=1.8)
            ax1.plot(time_steps, predicted_outputs[:display_limit], label=r"Model Prediction $\hat{y}(t|\theta)$", color="#FF7F0E", linestyle="--", linewidth=1.8)
            ax1.fill_between(time_steps, actual_targets[:display_limit], predicted_outputs[:display_limit], color="#FF7F0E", alpha=0.12, label="Prediction Error Band")
            ax1.set_title(f"Time-Series Tracking (First {display_limit} Steps) — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=13, fontweight="bold")
            ax1.set_xlabel("Time Step (t)", fontsize=11)
            ax1.set_ylabel("System Output", fontsize=11)
            ax1.legend(loc="upper right", frameon=True, fontsize=10)
            ax1.grid(True, linestyle="--", alpha=0.35)

            fig1.savefig(os.path.join(output_dir, f"tracking_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig1)

            # 2. Actual vs. predicted scatter
            fig2, ax2 = plt.subplots(figsize=(7.5, 7), dpi=300)
            ax2.scatter(actual_targets, predicted_outputs, alpha=0.55, color="#008080", edgecolors="none", s=25, label="Validation Samples")
            min_bound = min(np.min(actual_targets), np.min(predicted_outputs))
            max_bound = max(np.max(actual_targets), np.max(predicted_outputs))
            padding = 0.03 * (max_bound - min_bound if max_bound > min_bound else 1.0)
            ax2.plot([min_bound - padding, max_bound + padding], [min_bound - padding, max_bound + padding],
                     color="#D62728", linestyle="--", linewidth=2, label=r"Ideal Fit ($y = \hat{y}$)")

            max_idx = int(np.argmax(actual_targets))
            min_idx = int(np.argmin(actual_targets))
            ax2.scatter([actual_targets[max_idx]], [predicted_outputs[max_idx]], color="#D62728", s=80, zorder=5, marker="^", label=f"Global Max (t={max_idx})")
            ax2.scatter([actual_targets[min_idx]], [predicted_outputs[min_idx]], color="#1F77B4", s=80, zorder=5, marker="v", label=f"Global Min (t={min_idx})")

            ax2.set_title(f"Actual vs. Predicted Parity — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=13, fontweight="bold")
            ax2.set_xlabel("Actual Output $y(t)$", fontsize=11)
            ax2.set_ylabel(r"Predicted Output $\hat{y}(t|\theta)$", fontsize=11)
            ax2.set_xlim(min_bound - padding, max_bound + padding)
            ax2.set_ylim(min_bound - padding, max_bound + padding)
            ax2.set_aspect("equal", adjustable="box")
            ax2.legend(loc="upper left", frameon=True, fontsize=10)
            ax2.grid(True, linestyle="--", alpha=0.35)

            fig2.savefig(os.path.join(output_dir, f"scatter_fit_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig2)

            # 3. Zoom on global maximum and minimum
            zoom_window = 20

            def get_window(idx, max_len):
                start = max(0, idx - zoom_window)
                end = min(max_len, idx + zoom_window + 1)
                return start, end, np.arange(start, end)

            fig3, (ax_max, ax_min) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

            start_max, end_max, time_max = get_window(max_idx, len(actual_targets))
            y_win_max = actual_targets[start_max:end_max]
            yhat_win_max = predicted_outputs[start_max:end_max]

            true_peak_val = actual_targets[max_idx]
            pred_at_true_peak = predicted_outputs[max_idx]
            local_pred_max_idx = start_max + int(np.argmax(yhat_win_max))
            local_pred_max_val = predicted_outputs[local_pred_max_idx]
            max_error_gap = true_peak_val - pred_at_true_peak

            ax_max.plot(time_max, y_win_max, label="Actual Output $y(t)$", color="#111111", linewidth=2.0, marker="o", markersize=4.5)
            ax_max.plot(time_max, yhat_win_max, label=r"Predicted $\hat{y}(t|\theta)$", color="#FF7F0E", linestyle="--", linewidth=2.0, marker="s", markersize=4.5)
            ax_max.fill_between(time_max, y_win_max, yhat_win_max, color="#FF7F0E", alpha=0.15)

            ax_max.axhline(true_peak_val, color="#111111", linestyle=":", linewidth=1.3, alpha=0.8, label=f"Actual Max Level ({true_peak_val:.3f})")
            ax_max.axhline(local_pred_max_val, color="#D62728", linestyle="-.", linewidth=1.3, alpha=0.8, label=f"Model Max Reach ({local_pred_max_val:.3f})")

            ax_max.plot([max_idx, max_idx], [pred_at_true_peak, true_peak_val], color="#D62728", linewidth=2.5, zorder=6)
            ax_max.scatter([max_idx], [true_peak_val], color="#111111", s=90, zorder=7)
            ax_max.scatter([max_idx], [pred_at_true_peak], color="#D62728", s=90, zorder=7)

            info_box_max = (
                f"Global Maximum (t = {max_idx})\n"
                f"• Actual Peak: {true_peak_val:.4f}\n"
                f"• Model @ t={max_idx}: {pred_at_true_peak:.4f}\n"
                f"• Window Max Reach: {local_pred_max_val:.4f}\n"
                f"• Peak Deviation (Δy): {max_error_gap:+.4f}"
            )
            ax_max.text(0.03, 0.04, info_box_max, transform=ax_max.transAxes, fontsize=9.5,
                        verticalalignment="bottom", bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8DC", edgecolor="#D62728", alpha=0.92))

            ax_max.set_title(f"Extrema Zoom: Global Maximum Region (t ∈ [{start_max}, {end_max - 1}])", fontsize=12.5, fontweight="bold")
            ax_max.set_xlabel("Time Step (t)", fontsize=11)
            ax_max.set_ylabel("System Output", fontsize=11)
            ax_max.legend(loc="upper right", frameon=True, fontsize=9.5)
            ax_max.grid(True, linestyle="--", alpha=0.35)

            start_min, end_min, time_min = get_window(min_idx, len(actual_targets))
            y_win_min = actual_targets[start_min:end_min]
            yhat_win_min = predicted_outputs[start_min:end_min]

            true_min_val = actual_targets[min_idx]
            pred_at_true_min = predicted_outputs[min_idx]
            local_pred_min_idx = start_min + int(np.argmin(yhat_win_min))
            local_pred_min_val = predicted_outputs[local_pred_min_idx]
            min_error_gap = true_min_val - pred_at_true_min

            ax_min.plot(time_min, y_win_min, label="Actual Output $y(t)$", color="#111111", linewidth=2.0, marker="o", markersize=4.5)
            ax_min.plot(time_min, yhat_win_min, label=r"Predicted $\hat{y}(t|\theta)$", color="#1F77B4", linestyle="--", linewidth=2.0, marker="s", markersize=4.5)
            ax_min.fill_between(time_min, y_win_min, yhat_win_min, color="#1F77B4", alpha=0.15)

            ax_min.axhline(true_min_val, color="#111111", linestyle=":", linewidth=1.3, alpha=0.8, label=f"Actual Min Level ({true_min_val:.3f})")
            ax_min.axhline(local_pred_min_val, color="#D62728", linestyle="-.", linewidth=1.3, alpha=0.8, label=f"Model Min Reach ({local_pred_min_val:.3f})")

            ax_min.plot([min_idx, min_idx], [pred_at_true_min, true_min_val], color="#D62728", linewidth=2.5, zorder=6)
            ax_min.scatter([min_idx], [true_min_val], color="#111111", s=90, zorder=7)
            ax_min.scatter([min_idx], [pred_at_true_min], color="#D62728", s=90, zorder=7)

            info_box_min = (
                f"Global Minimum (t = {min_idx})\n"
                f"• Actual Minimum: {true_min_val:.4f}\n"
                f"• Model @ t={min_idx}: {pred_at_true_min:.4f}\n"
                f"• Window Min Reach: {local_pred_min_val:.4f}\n"
                f"• Trough Deviation (Δy): {min_error_gap:+.4f}"
            )
            ax_min.text(0.03, 0.96, info_box_min, transform=ax_min.transAxes, fontsize=9.5,
                        verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F8FF", edgecolor="#1F77B4", alpha=0.92))

            ax_min.set_title(f"Extrema Zoom: Global Minimum Region (t ∈ [{start_min}, {end_min - 1}])", fontsize=12.5, fontweight="bold")
            ax_min.set_xlabel("Time Step (t)", fontsize=11)
            ax_min.set_ylabel("System Output", fontsize=11)
            ax_min.legend(loc="lower right", frameon=True, fontsize=9.5)
            ax_min.grid(True, linestyle="--", alpha=0.35)

            plt.suptitle(f"Extrema Reach Analysis (Maxima & Minima) — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=14, fontweight="bold")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            fig3.savefig(os.path.join(output_dir, f"extrema_zoom_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig3)

            # Standalone Maximum Region Zoom (Separated for thesis / document integration)
            fig_max, ax_s_max = plt.subplots(figsize=(9.5, 5.5), dpi=300)
            ax_s_max.plot(time_max, y_win_max, label="Actual Output $y(t)$", color="#111111", linewidth=2.0, marker="o", markersize=4.5)
            ax_s_max.plot(time_max, yhat_win_max, label=r"Predicted $\hat{y}(t|\theta)$", color="#FF7F0E", linestyle="--", linewidth=2.0, marker="s", markersize=4.5)
            ax_s_max.fill_between(time_max, y_win_max, yhat_win_max, color="#FF7F0E", alpha=0.15)
            ax_s_max.axhline(true_peak_val, color="#111111", linestyle=":", linewidth=1.3, alpha=0.8, label=f"Actual Max Level ({true_peak_val:.3f})")
            ax_s_max.axhline(local_pred_max_val, color="#D62728", linestyle="-.", linewidth=1.3, alpha=0.8, label=f"Model Max Reach ({local_pred_max_val:.3f})")
            ax_s_max.plot([max_idx, max_idx], [pred_at_true_peak, true_peak_val], color="#D62728", linewidth=2.5, zorder=6)
            ax_s_max.scatter([max_idx], [true_peak_val], color="#111111", s=90, zorder=7)
            ax_s_max.scatter([max_idx], [pred_at_true_peak], color="#D62728", s=90, zorder=7)
            ax_s_max.text(0.03, 0.04, info_box_max, transform=ax_s_max.transAxes, fontsize=9.5,
                          verticalalignment="bottom", bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8DC", edgecolor="#D62728", alpha=0.92))
            ax_s_max.set_title(f"Extrema Zoom: Global Maximum Region — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=13, fontweight="bold")
            ax_s_max.set_xlabel("Time Step (t)", fontsize=11)
            ax_s_max.set_ylabel("System Output", fontsize=11)
            ax_s_max.legend(loc="upper right", frameon=True, fontsize=9.5)
            ax_s_max.grid(True, linestyle="--", alpha=0.35)
            fig_max.savefig(os.path.join(output_dir, f"extrema_zoom_max_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig_max)

            # Standalone Minimum Region Zoom (Separated for thesis / document integration)
            fig_min, ax_s_min = plt.subplots(figsize=(9.5, 5.5), dpi=300)
            ax_s_min.plot(time_min, y_win_min, label="Actual Output $y(t)$", color="#111111", linewidth=2.0, marker="o", markersize=4.5)
            ax_s_min.plot(time_min, yhat_win_min, label=r"Predicted $\hat{y}(t|\theta)$", color="#1F77B4", linestyle="--", linewidth=2.0, marker="s", markersize=4.5)
            ax_s_min.fill_between(time_min, y_win_min, yhat_win_min, color="#1F77B4", alpha=0.15)
            ax_s_min.axhline(true_min_val, color="#111111", linestyle=":", linewidth=1.3, alpha=0.8, label=f"Actual Min Level ({true_min_val:.3f})")
            ax_s_min.axhline(local_pred_min_val, color="#D62728", linestyle="-.", linewidth=1.3, alpha=0.8, label=f"Model Min Reach ({local_pred_min_val:.3f})")
            ax_s_min.plot([min_idx, min_idx], [pred_at_true_min, true_min_val], color="#D62728", linewidth=2.5, zorder=6)
            ax_s_min.scatter([min_idx], [true_min_val], color="#111111", s=90, zorder=7)
            ax_s_min.scatter([min_idx], [pred_at_true_min], color="#D62728", s=90, zorder=7)
            ax_s_min.text(0.03, 0.96, info_box_min, transform=ax_s_min.transAxes, fontsize=9.5,
                          verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F8FF", edgecolor="#1F77B4", alpha=0.92))
            ax_s_min.set_title(f"Extrema Zoom: Global Minimum Region — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=13, fontweight="bold")
            ax_s_min.set_xlabel("Time Step (t)", fontsize=11)
            ax_s_min.set_ylabel("System Output", fontsize=11)
            ax_s_min.legend(loc="lower right", frameon=True, fontsize=9.5)
            ax_s_min.grid(True, linestyle="--", alpha=0.35)
            fig_min.savefig(os.path.join(output_dir, f"extrema_zoom_min_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig_min)
        finally:
            plt.close("all")


def generate_residual_diagnostics_plot(actual_targets, predicted_outputs, dataset_name, fold_num, output_dir="plots", title_suffix=""):
    # Generates residual time-series, error histogram, and autocorrelation (ACF) plots
    with _PLOT_LOCK:
        try:
            _ensure_output_directory(output_dir)

            y_1d = _extract_1d_channel(actual_targets, 0)
            yhat_1d = _extract_1d_channel(predicted_outputs, 0)
            common_len = min(len(y_1d), len(yhat_1d))
            if common_len == 0:
                return
            residual_array = y_1d[:common_len] - yhat_1d[:common_len]
            sample_count = len(residual_array)

            if not np.all(np.isfinite(residual_array)):
                finite_mask = np.isfinite(residual_array)
                if not np.any(finite_mask):
                    return
                residual_array = residual_array[finite_mask]
                sample_count = len(residual_array)

            # Residual time-series
            fig1, ax1 = plt.subplots(figsize=(10, 4.5), dpi=300)
            ax1.scatter(range(sample_count), residual_array, alpha=0.5, color="#6A3D9A", s=12)
            ax1.axhline(0, color="black", linestyle="--", linewidth=1.5)
            ax1.set_title(f"Prediction Residuals Over Time — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
            ax1.set_xlabel("Time Step (t)", fontsize=11)
            ax1.set_ylabel(r"Residual Error $e(t) = y(t) - \hat{y}(t)$", fontsize=11)
            ax1.grid(True, linestyle="--", alpha=0.35)
            fig1.savefig(os.path.join(output_dir, f"residuals_timeseries_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig1)

            # Residual histogram
            fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)
            ax2.hist(residual_array, bins=35, color="#6A3D9A", alpha=0.75, edgecolor="black")
            ax2.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Error")
            ax2.axvline(np.mean(residual_array), color="#D62728", linestyle="-", linewidth=1.5, label=f"Mean ({np.mean(residual_array):.4f})")
            ax2.set_title(f"Residual Error Distribution — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
            ax2.set_xlabel("Error Value", fontsize=11)
            ax2.set_ylabel("Frequency", fontsize=11)
            ax2.legend(frameon=True)
            ax2.grid(True, linestyle="--", alpha=0.35)
            fig2.savefig(os.path.join(output_dir, f"residuals_distribution_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig2)

            # Residual autocorrelation (ACF)
            fig3, ax3 = plt.subplots(figsize=(9, 4.5), dpi=300)
            max_lag_limit = min(40, sample_count - 1)
            centered_residuals = residual_array - np.mean(residual_array)
            residual_variance = np.sum(centered_residuals ** 2)

            if residual_variance > 0 and max_lag_limit > 0:
                acf_full = np.correlate(centered_residuals, centered_residuals, mode="full") / residual_variance
                acf_pos = acf_full[sample_count - 1: sample_count + max_lag_limit]
                lag_indices = np.arange(len(acf_pos))

                ax3.stem(lag_indices, acf_pos, linefmt="#6A3D9A", markerfmt="o", basefmt="black")
                confidence_bound = 1.96 / np.sqrt(sample_count)
                ax3.axhline(confidence_bound, color="#D62728", linestyle="--", label="95% Confidence Bound")
                ax3.axhline(-confidence_bound, color="#D62728", linestyle="--")
                ax3.set_ylim(-1.05, 1.05)
                ax3.legend(loc="upper right", frameon=True)

            ax3.set_title(f"Residual Autocorrelation (ACF) — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
            ax3.set_xlabel("Lag", fontsize=11)
            ax3.set_ylabel("Correlation Coefficient", fontsize=11)
            ax3.grid(True, linestyle="--", alpha=0.35)
            fig3.savefig(os.path.join(output_dir, f"residuals_acf_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig3)
        finally:
            plt.close("all")


def generate_cross_validation_summary(fold_metrics_dict, fold_parameter_matrix, parameter_names, dataset_name, output_dir="plots"):
    # Plots fold MSE consistency and parameter standard deviations across the K folds
    # Automatically switches to a shaded envelope for high-dimensional models (k > 45) to prevent 2^16 px overflow
    with _PLOT_LOCK:
        try:
            _ensure_output_directory(output_dir)

            if not fold_metrics_dict or "MSE" not in fold_metrics_dict or len(fold_metrics_dict["MSE"]) == 0 or len(fold_parameter_matrix) == 0:
                return

            mse_array = fold_metrics_dict["MSE"]
            fold_indices = np.arange(1, len(mse_array) + 1)
            overall_mean_mse = np.mean(mse_array)

            min_p = min((len(row) for row in fold_parameter_matrix), default=0)
            if min_p == 0:
                return
            fold_mat = np.asarray([row[:min_p] for row in fold_parameter_matrix], dtype=float)
            mean_theta = np.mean(fold_mat, axis=0)
            std_theta = np.std(fold_mat, axis=0)
            num_params = min(len(parameter_names), len(mean_theta))
            if num_params == 0:
                return
            mean_theta = mean_theta[:num_params]
            std_theta = std_theta[:num_params]
            parameter_names = list(parameter_names)[:num_params]
            param_x = np.arange(num_params)

            fig1, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
            ax1.bar(fold_indices, mse_array, color="#3B8ED0", alpha=0.85, edgecolor="black", label="Fold Test MSE")
            ax1.axhline(overall_mean_mse, color="#D62728", linestyle="--", linewidth=2, label=f"Mean MSE ({overall_mean_mse:.4f})")
            ax1.set_title(f"Cross-Validation MSE Consistency — {dataset_name}", fontsize=13, fontweight="bold")
            ax1.set_xlabel("Fold Number", fontsize=11)
            ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
            ax1.set_xticks(fold_indices)
            ax1.legend(frameon=True)
            ax1.grid(True, linestyle="--", alpha=0.35)
            fig1.savefig(os.path.join(output_dir, f"cv_mse_folds_{dataset_name}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig1)

            # Cap width to 18 inches max so large MLPs (k = 800..5000) never exceed Matplotlib's 2^16 pixel limit
            fig_width = min(18.0, max(9.0, num_params * 0.45)) if num_params <= 45 else 13.0
            fig2, ax2 = plt.subplots(figsize=(fig_width, 5), dpi=300)
            if num_params <= 45:
                ax2.errorbar(
                    param_x, mean_theta, yerr=std_theta, fmt="o", color="#2CA02C",
                    ecolor="black", elinewidth=1.8, capsize=4, markersize=6.5, label="Mean Parameter ± 1 SD"
                )
                ax2.set_xticks(param_x)
                ax2.set_xticklabels(
                    parameter_names,
                    rotation=45 if num_params > 8 else 0,
                    ha="right" if num_params > 8 else "center"
                )
            else:
                ax2.plot(param_x, mean_theta, color="#2CA02C", linewidth=1.4, label=f"Mean Parameter Value (k = {num_params})")
                ax2.fill_between(
                    param_x, mean_theta - std_theta, mean_theta + std_theta,
                    color="#2CA02C", alpha=0.28, label="± 1 SD Cross-Fold Envelope"
                )
                ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15, integer=True))
                ax2.set_xlim(0, max(1, num_params - 1))

            ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
            ax2.set_title(f"Parameter Stability Across {len(mse_array)} Folds — {dataset_name}", fontsize=13, fontweight="bold")
            ax2.set_xlabel("Model Parameter" if num_params <= 45 else "Parameter Index (w_i)", fontsize=11)
            ax2.set_ylabel("Parameter Value", fontsize=11)
            ax2.legend(frameon=True, loc="upper right")
            ax2.grid(True, linestyle="--", alpha=0.35)
            fig2.savefig(os.path.join(output_dir, f"cv_parameter_variance_{dataset_name}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig2)
        finally:
            plt.close("all")


def generate_benchmark_boxplots(benchmark_scores_dict, output_dir="results_benchmarks"):
    # Generates a log-scale boxplot of final objective values across benchmark functions.
    # Supports both single-algorithm dicts {"Sphere": [...]} and multi-algorithm dicts {"Sphere": {"EP": [...], "DE": [...]}}.
    with _PLOT_LOCK:
        try:
            _ensure_output_directory(output_dir)

            function_names = list(benchmark_scores_dict.keys())
            if not function_names:
                return

            first_val = benchmark_scores_dict[function_names[0]]
            is_grouped = isinstance(first_val, dict)

            fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=300)

            if is_grouped:
                algo_names = list(first_val.keys())
                n_algos = max(1, len(algo_names))
                x_centers = np.arange(len(function_names), dtype=float)
                total_group_width = 0.68
                box_width = total_group_width / n_algos
                palette = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD"]
                legend_handles = []

                for a_idx, algo_label in enumerate(algo_names):
                    offset = (a_idx - (n_algos - 1) / 2.0) * (box_width + 0.04)
                    positions = x_centers + offset
                    distributions = []
                    for func in function_names:
                        raw_scores = np.asarray(benchmark_scores_dict.get(func, {}).get(algo_label, []), dtype=float)
                        sanitized = np.nan_to_num(raw_scores, nan=1e-18, posinf=1e6, neginf=1e-18)
                        clipped = np.clip(sanitized, 1e-18, 1e9)
                        distributions.append(clipped if len(clipped) > 0 else np.array([1e-18]))

                    bp = ax.boxplot(
                        distributions,
                        positions=positions,
                        widths=box_width,
                        patch_artist=True,
                        manage_ticks=False
                    )
                    color = palette[a_idx % len(palette)]
                    for box in bp["boxes"]:
                        box.set(facecolor=color, alpha=0.75, edgecolor="black")
                    for median in bp["medians"]:
                        median.set(color="black", linewidth=1.6)
                    legend_handles.append(bp["boxes"][0])

                ax.set_xticks(x_centers)
                ax.set_xticklabels(function_names, fontsize=10.5)
                ax.set_xlim(-0.6, len(function_names) - 0.4)
                ax.legend(legend_handles, algo_names, loc="upper left", frameon=True, fontsize=10)
                ax.set_title("Head-to-Head Benchmark Performance Distribution (EP vs. DE)", fontsize=13, fontweight="bold")
            else:
                score_distributions = []
                for func in function_names:
                    raw_scores = np.asarray(benchmark_scores_dict.get(func, []), dtype=float)
                    sanitized = np.nan_to_num(raw_scores, nan=1e-18, posinf=1e6, neginf=1e-18)
                    clipped = np.clip(sanitized, 1e-18, 1e9)
                    score_distributions.append(clipped if len(clipped) > 0 else np.array([1e-18]))

                try:
                    boxplot_elements = ax.boxplot(score_distributions, tick_labels=function_names, patch_artist=True)
                except TypeError:
                    boxplot_elements = ax.boxplot(score_distributions, labels=function_names, patch_artist=True)

                for box in boxplot_elements["boxes"]:
                    box.set(facecolor="#3B8ED0", alpha=0.75)
                ax.set_title("Optimization Performance on Benchmark Test Functions", fontsize=13, fontweight="bold")

            ax.set_yscale("log")
            ax.set_xlabel("Test Function", fontsize=11)
            ax.set_ylabel("Final Objective Value [Log Scale]", fontsize=11)
            ax.grid(True, which="both", linestyle="--", alpha=0.35)

            filepath = os.path.join(output_dir, "benchmark_comparison_boxplot.png")
            fig.savefig(filepath, bbox_inches="tight", dpi=300)
        finally:
            plt.close("all")


def generate_multi_model_comparison_plots(actual_targets, model_predictions_dict, model_metrics_dict, dataset_name, output_dir="plots"):
    # Plots side-by-side comparisons for N models on the same aligned signal
    with _PLOT_LOCK:
        try:
            _ensure_output_directory(output_dir)

            actual_targets = _extract_1d_channel(actual_targets, 0)
            model_labels = list(model_predictions_dict.keys())
            num_models = len(model_labels)
            if len(actual_targets) == 0 or num_models == 0:
                return

            min_pred_len = min((len(_extract_1d_channel(model_predictions_dict[lbl], 0)) for lbl in model_labels), default=len(actual_targets))
            common_len = min(len(actual_targets), min_pred_len)
            if common_len == 0:
                return
            actual_targets = actual_targets[:common_len]
            display_limit = min(common_len, 200)

            cmap = plt.get_cmap("tab10" if num_models <= 10 else "tab20")
            model_colors = [cmap(i % cmap.N) for i in range(num_models)]
            linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

            # 1. Tracking overlay
            fig1, ax1 = plt.subplots(figsize=(13, 5.5), dpi=300)
            time_steps = np.arange(display_limit)
            ax1.plot(time_steps, actual_targets[:display_limit], label="Actual Output $y(t)$",
                     color="#111111", linewidth=2.2, zorder=10)

            for idx, label in enumerate(model_labels):
                preds = _extract_1d_channel(model_predictions_dict[label], 0)
                r2_val = _safe_float(model_metrics_dict.get(label, {}).get("R-Squared"), fallback=0.0)
                ax1.plot(
                    time_steps, preds[:display_limit],
                    label=f"{label} ($R^2={r2_val:.4f}$)",
                    color=model_colors[idx],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=1.7,
                    alpha=0.9
                )

            ax1.set_title(f"Multi-Model Time-Series Tracking Comparison — {dataset_name} (First {display_limit} Steps)", fontsize=13, fontweight="bold")
            ax1.set_xlabel("Aligned Time Step (t)", fontsize=11)
            ax1.set_ylabel("System Output", fontsize=11)
            ax1.legend(loc="upper right", frameon=True, fontsize=9.5)
            ax1.grid(True, linestyle="--", alpha=0.35)
            fig1.savefig(os.path.join(output_dir, f"comparison_tracking_{dataset_name}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig1)

            # 2. Extrema zoom overlay
            max_idx = int(np.argmax(actual_targets))
            min_idx = int(np.argmin(actual_targets))
            zoom_window = 20

            def get_window(center_idx, max_len):
                start = max(0, center_idx - zoom_window)
                end = min(max_len, center_idx + zoom_window + 1)
                return start, end, np.arange(start, end)

            fig2, (ax_max, ax_min) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

            start_max, end_max, time_max = get_window(max_idx, len(actual_targets))
            true_peak_val = actual_targets[max_idx]
            ax_max.plot(time_max, actual_targets[start_max:end_max],
                        label=f"Actual Output (Peak={true_peak_val:.3f})",
                        color="#111111", linewidth=2.4, marker="o", markersize=5, zorder=10)
            ax_max.axhline(true_peak_val, color="#111111", linestyle=":", linewidth=1.2, alpha=0.75)

            for idx, label in enumerate(model_labels):
                preds = _extract_1d_channel(model_predictions_dict[label], 0)
                pred_at_peak = preds[max_idx]
                delta_peak = pred_at_peak - true_peak_val
                ax_max.plot(
                    time_max, preds[start_max:end_max],
                    label=rf"{label} (@Peak: {pred_at_peak:.3f}, $\Delta$={delta_peak:+.3f})",
                    color=model_colors[idx],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=1.8, marker="s", markersize=4
                )
                ax_max.scatter([max_idx], [pred_at_peak], color=model_colors[idx], s=70, zorder=9)

            ax_max.scatter([max_idx], [true_peak_val], color="#111111", s=95, zorder=11)
            ax_max.set_title(f"Global Maximum Zoom (Around t={max_idx})", fontsize=12.5, fontweight="bold")
            ax_max.set_xlabel("Aligned Time Step (t)", fontsize=11)
            ax_max.set_ylabel("System Output", fontsize=11)
            ax_max.legend(loc="best", frameon=True, fontsize=8.8)
            ax_max.grid(True, linestyle="--", alpha=0.35)

            start_min, end_min, time_min = get_window(min_idx, len(actual_targets))
            true_min_val = actual_targets[min_idx]
            ax_min.plot(time_min, actual_targets[start_min:end_min],
                        label=f"Actual Output (Min={true_min_val:.3f})",
                        color="#111111", linewidth=2.4, marker="o", markersize=5, zorder=10)
            ax_min.axhline(true_min_val, color="#111111", linestyle=":", linewidth=1.2, alpha=0.75)

            for idx, label in enumerate(model_labels):
                preds = _extract_1d_channel(model_predictions_dict[label], 0)
                pred_at_min = preds[min_idx]
                delta_min = pred_at_min - true_min_val
                ax_min.plot(
                    time_min, preds[start_min:end_min],
                    label=rf"{label} (@Min: {pred_at_min:.3f}, $\Delta$={delta_min:+.3f})",
                    color=model_colors[idx],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=1.8, marker="s", markersize=4
                )
                ax_min.scatter([min_idx], [pred_at_min], color=model_colors[idx], s=70, zorder=9)

            ax_min.scatter([min_idx], [true_min_val], color="#111111", s=95, zorder=11)
            ax_min.set_title(f"Global Minimum Zoom (Around t={min_idx})", fontsize=12.5, fontweight="bold")
            ax_min.set_xlabel("Aligned Time Step (t)", fontsize=11)
            ax_min.set_ylabel("System Output", fontsize=11)
            ax_min.legend(loc="best", frameon=True, fontsize=8.8)
            ax_min.grid(True, linestyle="--", alpha=0.35)

            plt.suptitle(f"Multi-Model Extrema Reach Comparison (Maxima & Minima) — {dataset_name}", fontsize=14, fontweight="bold")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig2.savefig(os.path.join(output_dir, f"comparison_extrema_zoom_{dataset_name}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig2)

            # Standalone Multi-Model Extrema Zoom: Maximum Region
            fig_c_max, ax_c_max = plt.subplots(figsize=(9.5, 5.5), dpi=300)
            ax_c_max.plot(time_max, actual_targets[start_max:end_max],
                          label=f"Actual Output (Peak={true_peak_val:.3f})",
                          color="#111111", linewidth=2.4, marker="o", markersize=5, zorder=10)
            ax_c_max.axhline(true_peak_val, color="#111111", linestyle=":", linewidth=1.2, alpha=0.75)
            for idx, label in enumerate(model_labels):
                preds = _extract_1d_channel(model_predictions_dict[label], 0)
                pred_at_peak = preds[max_idx]
                delta_peak = pred_at_peak - true_peak_val
                ax_c_max.plot(
                    time_max, preds[start_max:end_max],
                    label=f"{label} (@Peak: {pred_at_peak:.3f}, Δ={delta_peak:+.3f})",
                    color=model_colors[idx],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=1.8, marker="s", markersize=4
                )
                ax_c_max.scatter([max_idx], [pred_at_peak], color=model_colors[idx], s=70, zorder=9)
            ax_c_max.scatter([max_idx], [true_peak_val], color="#111111", s=95, zorder=11)
            ax_c_max.set_title(f"Multi-Model Global Maximum Zoom — {dataset_name} (Around t={max_idx})", fontsize=12.5, fontweight="bold")
            ax_c_max.set_xlabel("Aligned Time Step (t)", fontsize=11)
            ax_c_max.set_ylabel("System Output", fontsize=11)
            ax_c_max.legend(loc="best", frameon=True, fontsize=9)
            ax_c_max.grid(True, linestyle="--", alpha=0.35)
            fig_c_max.savefig(os.path.join(output_dir, f"comparison_extrema_zoom_max_{dataset_name}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig_c_max)

            # Standalone Multi-Model Extrema Zoom: Minimum Region
            fig_c_min, ax_c_min = plt.subplots(figsize=(9.5, 5.5), dpi=300)
            ax_c_min.plot(time_min, actual_targets[start_min:end_min],
                          label=f"Actual Output (Min={true_min_val:.3f})",
                          color="#111111", linewidth=2.4, marker="o", markersize=5, zorder=10)
            ax_c_min.axhline(true_min_val, color="#111111", linestyle=":", linewidth=1.2, alpha=0.75)
            for idx, label in enumerate(model_labels):
                preds = _extract_1d_channel(model_predictions_dict[label], 0)
                pred_at_min = preds[min_idx]
                delta_min = pred_at_min - true_min_val
                ax_c_min.plot(
                    time_min, preds[start_min:end_min],
                    label=f"{label} (@Min: {pred_at_min:.3f}, Δ={delta_min:+.3f})",
                    color=model_colors[idx],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=1.8, marker="s", markersize=4
                )
                ax_c_min.scatter([min_idx], [pred_at_min], color=model_colors[idx], s=70, zorder=9)
            ax_c_min.scatter([min_idx], [true_min_val], color="#111111", s=95, zorder=11)
            ax_c_min.set_title(f"Multi-Model Global Minimum Zoom — {dataset_name} (Around t={min_idx})", fontsize=12.5, fontweight="bold")
            ax_c_min.set_xlabel("Aligned Time Step (t)", fontsize=11)
            ax_c_min.set_ylabel("System Output", fontsize=11)
            ax_c_min.legend(loc="best", frameon=True, fontsize=9)
            ax_c_min.grid(True, linestyle="--", alpha=0.35)
            fig_c_min.savefig(os.path.join(output_dir, f"comparison_extrema_zoom_min_{dataset_name}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig_c_min)

            # 3. Validation metrics bar chart
            candidate_metrics = [
                ("MSE", "Mean Squared Error (MSE)", "{:.4f}"),
                ("RMSE", "Root Mean Squared Error (RMSE)", "{:.4f}"),
                ("MAE", "Mean Absolute Error (MAE)", "{:.4f}"),
                ("MAPE", "Mean Absolute Percentage Error (%)", "{:.2f}%"),
                ("R-Squared", "Coefficient of Determination ($R^2$)", "{:.4f}"),
                ("VAF_Pct", "Variance Accounted For (VAF %)", "{:.2f}%"),
                ("AICc", "Corrected AIC ($AIC_c$)", "{:.1f}"),
                ("White_Noise_Pct", "White-Noise ACF Pass Rate (%)", "{:.1f}%"),
                ("CCF_Pass_Pct", "Input-Residual Cross-Correlation (%)", "{:.1f}%"),
                ("Sim_Fit_Pct", "Autonomous Simulation Fit (%)", "{:.1f}%"),
                ("Sim_RMSE", "Autonomous Simulation RMSE", "{:.4f}"),
                ("FE", "Fitness Evaluations (FE)", "{:,.0f}"),
                ("Execution_Time_Sec", "Training Wall-Clock Time (s)", "{:.2f}s"),
                ("Cost_Per_FE_ms", "Cost per Evaluation (ms/FE)", "{:.3f}ms"),
            ]
            metrics_to_plot = []
            for key, title, fmt in candidate_metrics:
                has_metric = any(
                    (key in model_metrics_dict.get(lbl, {}) and np.isfinite(_safe_float(model_metrics_dict[lbl].get(key), fallback=np.nan)))
                    for lbl in model_labels
                )
                if has_metric:
                    metrics_to_plot.append((key, title, fmt))

            if not metrics_to_plot:
                metrics_to_plot = [
                    ("MSE", "Mean Squared Error (MSE)", "{:.4f}"),
                    ("RMSE", "Root Mean Squared Error (RMSE)", "{:.4f}"),
                    ("MAPE", "Mean Absolute Percentage Error (%)", "{:.2f}%"),
                    ("R-Squared", "Coefficient of Determination ($R^2$)", "{:.4f}")
                ]

            n_m = len(metrics_to_plot)
            n_cols = 4 if n_m >= 7 else (3 if n_m in (5, 6) else 2)
            n_rows = int(np.ceil(n_m / n_cols))

            fig3, axes = plt.subplots(n_rows, n_cols, figsize=(max(4.8 * n_cols, num_models * 1.6 * n_cols), 4.2 * n_rows), dpi=300)
            axes_flat = np.atleast_1d(axes).flatten()
            x_pos = np.arange(num_models)

            for idx_m, (metric_key, metric_title, fmt_str) in enumerate(metrics_to_plot):
                ax = axes_flat[idx_m]
                values = [_safe_float(model_metrics_dict.get(label, {}).get(metric_key), fallback=np.nan) for label in model_labels]
                bars = ax.bar(x_pos, values, color=model_colors, alpha=0.85, edgecolor="black", width=0.55)
                ax.set_title(metric_title, fontsize=11, fontweight="bold")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(model_labels, rotation=20 if num_models > 3 else 0, ha="right" if num_models > 3 else "center", fontsize=9)
                ax.grid(True, axis="y", linestyle="--", alpha=0.35)

                for bar, val in zip(bars, values):
                    if np.isfinite(val):
                        height = bar.get_height()
                        va = "bottom" if height >= 0 else "top"
                        y_offset = 4 if height >= 0 else -4
                        ax.annotate(
                            fmt_str.format(val),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, y_offset),
                            textcoords="offset points",
                            ha="center", va=va, fontsize=8.5, fontweight="bold"
                        )
                    else:
                        ax.annotate(
                            "N/A",
                            xy=(bar.get_x() + bar.get_width() / 2, 0),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#D62728"
                        )

                y_min, y_max = ax.get_ylim()
                if np.isfinite(y_min) and np.isfinite(y_max):
                    span = abs(y_max - y_min) if y_max != y_min else 1.0
                    ax.set_ylim(y_min - 0.06 * span if y_min < 0 else 0, y_max + 0.16 * span)

            for idx_empty in range(n_m, len(axes_flat)):
                axes_flat[idx_empty].set_visible(False)

            plt.suptitle(f"Multi-Model Validation Metrics Comparison — {dataset_name}", fontsize=14, fontweight="bold")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig3.savefig(os.path.join(output_dir, f"comparison_metrics_bars_{dataset_name}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig3)

            # Standalone individual metric bar charts
            for metric_key, metric_title, fmt_str in metrics_to_plot:
                fig_sb, ax_sb = plt.subplots(figsize=(max(8, num_models * 1.8), 5), dpi=300)
                values = [_safe_float(model_metrics_dict.get(label, {}).get(metric_key), fallback=np.nan) for label in model_labels]
                bars = ax_sb.bar(x_pos, values, color=model_colors, alpha=0.85, edgecolor="black", width=0.55)
                ax_sb.set_title(f"Model Comparison: {metric_title} — {dataset_name}", fontsize=12.5, fontweight="bold")
                ax_sb.set_xticks(x_pos)
                ax_sb.set_xticklabels(model_labels, rotation=20 if num_models > 3 else 0, ha="right" if num_models > 3 else "center", fontsize=9.5)
                ax_sb.grid(True, axis="y", linestyle="--", alpha=0.35)
                for bar, val in zip(bars, values):
                    if np.isfinite(val):
                        height = bar.get_height()
                        va = "bottom" if height >= 0 else "top"
                        y_offset = 4 if height >= 0 else -4
                        ax_sb.annotate(
                            fmt_str.format(val),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, y_offset),
                            textcoords="offset points",
                            ha="center", va=va, fontsize=9, fontweight="bold"
                        )
                    else:
                        ax_sb.annotate(
                            "N/A",
                            xy=(bar.get_x() + bar.get_width() / 2, 0),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#D62728"
                        )
                y_min, y_max = ax_sb.get_ylim()
                if np.isfinite(y_min) and np.isfinite(y_max):
                    span = abs(y_max - y_min) if y_max != y_min else 1.0
                    ax_sb.set_ylim(y_min - 0.06 * span if y_min < 0 else 0, y_max + 0.16 * span)
                metric_slug = metric_key.lower().replace("-", "").replace(" ", "_")
                fig_sb.savefig(os.path.join(output_dir, f"comparison_bar_{metric_slug}_{dataset_name}.png"), bbox_inches="tight", dpi=300)
                plt.close(fig_sb)
        finally:
            plt.close("all")


def generate_single_model_deep_diagnostics_plots(u_aligned, actual_targets, predicted_outputs, dataset_name, fold_num, output_dir="plots", title_suffix=""):
    # Generates 4 deep diagnostic figures for single-model Check / Inference runs
    with _PLOT_LOCK:
        try:
            _ensure_output_directory(output_dir)

            u_arr = _extract_1d_channel(u_aligned, 0)
            y_arr = _extract_1d_channel(actual_targets, 0)
            yhat_arr = _extract_1d_channel(predicted_outputs, 0)
            n_samples = min(len(u_arr), len(y_arr), len(yhat_arr))
            if n_samples == 0:
                return
            u_arr, y_arr, yhat_arr = u_arr[:n_samples], y_arr[:n_samples], yhat_arr[:n_samples]
            finite_mask = np.isfinite(u_arr) & np.isfinite(y_arr) & np.isfinite(yhat_arr)
            if not np.any(finite_mask):
                return
            if not np.all(finite_mask):
                u_arr, y_arr, yhat_arr = u_arr[finite_mask], y_arr[finite_mask], yhat_arr[finite_mask]
                n_samples = len(u_arr)
            residuals = y_arr - yhat_arr
            abs_err = np.abs(residuals)

            # PLOT 1: Input-Residual Cross-Correlation R_ue(tau)
            fig1, ax1 = plt.subplots(figsize=(9.5, 4.5), dpi=300)
            max_lag = min(40, n_samples - 1)
            u_cent = u_arr - np.mean(u_arr)
            e_cent = residuals - np.mean(residuals)
            denom = np.sqrt(np.sum(u_cent ** 2) * np.sum(e_cent ** 2))

            if denom > 1e-12 and max_lag > 0:
                ccf_full = np.correlate(e_cent, u_cent, mode="full") / denom
                mid = n_samples - 1
                lags = np.arange(-max_lag, max_lag + 1)
                ccf_vals = ccf_full[mid - max_lag: mid + max_lag + 1]
                conf_bound = 1.96 / np.sqrt(n_samples)
                pass_rate = float(np.mean(np.abs(ccf_vals) <= conf_bound) * 100.0)

                ax1.stem(lags, ccf_vals, linefmt="#1F77B4", markerfmt="o", basefmt="black")
                ax1.axhline(conf_bound, color="#D62728", linestyle="--", linewidth=1.5, label=f"95% Confidence Bound (Pass: {pass_rate:.1f}%)")
                ax1.axhline(-conf_bound, color="#D62728", linestyle="--", linewidth=1.5)
                ax1.axvline(0, color="gray", linestyle=":", alpha=0.6)
                ax1.set_ylim(-1.05, 1.05)
                ax1.legend(loc="upper right", frameon=True, fontsize=9.5)

            ax1.set_title(f"Input-Residual Cross-Correlation $R_{{ue}}(\\tau)$ — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
            ax1.set_xlabel(r"Lag $\tau$ (Causal Input Leakage for $\tau \geq 0$)", fontsize=11)
            ax1.set_ylabel("Cross-Correlation Coefficient", fontsize=11)
            ax1.grid(True, linestyle="--", alpha=0.35)
            fig1.savefig(os.path.join(output_dir, f"input_residual_ccf_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig1)

            # PLOT 2: Error vs. Input Regime & Error vs. Predicted Magnitude
            fig2, (ax_u, ax_yh) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

            ax_u.scatter(u_arr, residuals, alpha=0.45, color="#2CA02C", s=18, edgecolors="none", label="Sample Residuals")
            ax_u.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Error")
            bins_u = None
            bin_centers = []
            bin_means = []
            try:
                bins_u = np.linspace(np.min(u_arr), np.max(u_arr), 12)
                if len(np.unique(bins_u)) > 3:
                    digitized = np.digitize(u_arr, bins_u)
                    bin_centers = [np.mean(u_arr[digitized == i]) for i in range(1, len(bins_u)) if np.any(digitized == i)]
                    bin_means = [np.mean(residuals[digitized == i]) for i in range(1, len(bins_u)) if np.any(digitized == i)]
                    ax_u.plot(bin_centers, bin_means, color="#D62728", linewidth=2.2, marker="o", markersize=5, label="Regime Mean Bias")
            except Exception:
                pass
            ax_u.set_title("Residuals vs. Primary Input $u_1(t)$ (Regime Bias Check)", fontsize=12, fontweight="bold")
            ax_u.set_xlabel("Input Signal $u_1(t)$", fontsize=11)
            ax_u.set_ylabel(r"Residual Error $e(t) = y(t) - \hat{y}(t)$", fontsize=11)
            ax_u.legend(loc="best", frameon=True, fontsize=9.5)
            ax_u.grid(True, linestyle="--", alpha=0.35)

            ax_yh.scatter(yhat_arr, residuals, alpha=0.45, color="#FF7F0E", s=18, edgecolors="none", label="Sample Residuals")
            ax_yh.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Error")
            ax_yh.set_title(r"Residuals vs. Predicted Output $\hat{y}(t)$ (Heteroscedasticity Check)", fontsize=12, fontweight="bold")
            ax_yh.set_xlabel(r"Predicted Output $\hat{y}(t|\theta)$", fontsize=11)
            ax_yh.set_ylabel(r"Residual Error $e(t) = y(t) - \hat{y}(t)$", fontsize=11)
            ax_yh.legend(loc="best", frameon=True, fontsize=9.5)
            ax_yh.grid(True, linestyle="--", alpha=0.35)

            plt.suptitle(f"Operating Regime & Heteroscedasticity Diagnostics — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=13.5, fontweight="bold")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig2.savefig(os.path.join(output_dir, f"error_regime_analysis_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig2)

            # Standalone Plot 2A: Residuals vs Primary Input (Regime Bias)
            fig_u_s, ax_u_s = plt.subplots(figsize=(8.5, 5.2), dpi=300)
            ax_u_s.scatter(u_arr, residuals, alpha=0.45, color="#2CA02C", s=18, edgecolors="none", label="Sample Residuals")
            ax_u_s.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Error")
            try:
                if bins_u is not None and len(np.unique(bins_u)) > 3:
                    ax_u_s.plot(bin_centers, bin_means, color="#D62728", linewidth=2.2, marker="o", markersize=5, label="Regime Mean Bias")
            except Exception:
                pass
            ax_u_s.set_title(f"Residuals vs. Primary Input $u_1(t)$ — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
            ax_u_s.set_xlabel("Input Signal $u_1(t)$", fontsize=11)
            ax_u_s.set_ylabel(r"Residual Error $e(t) = y(t) - \hat{y}(t)$", fontsize=11)
            ax_u_s.legend(loc="best", frameon=True, fontsize=9.5)
            ax_u_s.grid(True, linestyle="--", alpha=0.35)
            fig_u_s.savefig(os.path.join(output_dir, f"error_regime_input_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig_u_s)

            # Standalone Plot 2B: Residuals vs Predicted Output (Heteroscedasticity)
            fig_yh_s, ax_yh_s = plt.subplots(figsize=(8.5, 5.2), dpi=300)
            ax_yh_s.scatter(yhat_arr, residuals, alpha=0.45, color="#FF7F0E", s=18, edgecolors="none", label="Sample Residuals")
            ax_yh_s.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Error")
            ax_yh_s.set_title(rf"Residuals vs. Predicted Output $\hat{{y}}(t)$ — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
            ax_yh_s.set_xlabel(r"Predicted Output $\hat{y}(t|\theta)$", fontsize=11)
            ax_yh_s.set_ylabel(r"Residual Error $e(t) = y(t) - \hat{y}(t)$", fontsize=11)
            ax_yh_s.legend(loc="best", frameon=True, fontsize=9.5)
            ax_yh_s.grid(True, linestyle="--", alpha=0.35)
            fig_yh_s.savefig(os.path.join(output_dir, f"error_regime_predicted_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig_yh_s)

            # PLOT 3: Power Spectral Density (PSD) Overlay & Empirical Error CDF
            fig3, (ax_psd, ax_cdf) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

            y_zero = y_arr - np.mean(y_arr)
            yhat_zero = yhat_arr - np.mean(yhat_arr)
            freqs = np.fft.rfftfreq(n_samples, d=1.0)
            psd_y = (np.abs(np.fft.rfft(y_zero)) ** 2) / max(1, n_samples)
            psd_yhat = (np.abs(np.fft.rfft(yhat_zero)) ** 2) / max(1, n_samples)

            if len(psd_y) > 25:
                kernel = np.ones(5) / 5.0
                psd_y = np.convolve(psd_y, kernel, mode="same")
                psd_yhat = np.convolve(psd_yhat, kernel, mode="same")

            valid_f = freqs > 0
            if np.any(valid_f):
                ax_psd.semilogy(freqs[valid_f], np.maximum(psd_y[valid_f], 1e-12), color="#111111", linewidth=1.8, alpha=0.85, label="Actual Output $y(t)$ PSD")
                ax_psd.semilogy(freqs[valid_f], np.maximum(psd_yhat[valid_f], 1e-12), color="#D62728", linestyle="--", linewidth=1.8, alpha=0.85, label=r"Model Prediction $\hat{y}(t)$ PSD")
            ax_psd.set_title("Frequency-Domain Power Spectral Density (Bandwidth Match)", fontsize=12, fontweight="bold")
            ax_psd.set_xlabel("Normalized Frequency (cycles / sample)", fontsize=11)
            ax_psd.set_ylabel("Power Spectral Density [Log Scale]", fontsize=11)
            ax_psd.legend(loc="upper right", frameon=True, fontsize=9.5)
            ax_psd.grid(True, which="both", linestyle="--", alpha=0.35)

            sorted_ae = np.sort(abs_err)
            cdf_prob = np.arange(1, n_samples + 1) / float(n_samples) * 100.0
            p50, p90, p95 = np.percentile(abs_err, [50, 90, 95])

            ax_cdf.plot(sorted_ae, cdf_prob, color="#6A3D9A", linewidth=2.4, label="Empirical Error CDF")
            ax_cdf.axvline(p50, color="#2CA02C", linestyle="--", linewidth=1.5, label=f"Median (P50): {p50:.3f}")
            ax_cdf.axvline(p90, color="#FF7F0E", linestyle="--", linewidth=1.5, label=f"90th Pct (P90): {p90:.3f}")
            ax_cdf.axvline(p95, color="#D62728", linestyle="--", linewidth=1.5, label=f"95th Pct (P95): {p95:.3f}")
            ax_cdf.set_title("Cumulative Distribution of Absolute Error $|e(t)|$", fontsize=12, fontweight="bold")
            ax_cdf.set_xlabel("Absolute Prediction Error $|e(t)|$", fontsize=11)
            ax_cdf.set_ylabel("Cumulative Percentage of Samples (%)", fontsize=11)
            ax_cdf.set_ylim(0, 102)
            ax_cdf.legend(loc="lower right", frameon=True, fontsize=9.5)
            ax_cdf.grid(True, linestyle="--", alpha=0.35)

            plt.suptitle(f"Spectral Fidelity & Error Quantile Cumulative Profile — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=13.5, fontweight="bold")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig3.savefig(os.path.join(output_dir, f"spectral_and_error_cdf_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig3)

            # Standalone Plot 3A: Power Spectral Density (PSD)
            fig_psd_s, ax_psd_s = plt.subplots(figsize=(9, 5.2), dpi=300)
            if np.any(valid_f):
                ax_psd_s.semilogy(freqs[valid_f], np.maximum(psd_y[valid_f], 1e-12), color="#111111", linewidth=1.8, alpha=0.85, label="Actual Output $y(t)$ PSD")
                ax_psd_s.semilogy(freqs[valid_f], np.maximum(psd_yhat[valid_f], 1e-12), color="#D62728", linestyle="--", linewidth=1.8, alpha=0.85, label=r"Model Prediction $\hat{y}(t)$ PSD")
            ax_psd_s.set_title(f"Frequency-Domain Power Spectral Density — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
            ax_psd_s.set_xlabel("Normalized Frequency (cycles / sample)", fontsize=11)
            ax_psd_s.set_ylabel("Power Spectral Density [Log Scale]", fontsize=11)
            ax_psd_s.legend(loc="upper right", frameon=True, fontsize=9.5)
            ax_psd_s.grid(True, which="both", linestyle="--", alpha=0.35)
            fig_psd_s.savefig(os.path.join(output_dir, f"spectral_psd_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig_psd_s)

            # Standalone Plot 3B: Error Cumulative Distribution (CDF)
            fig_cdf_s, ax_cdf_s = plt.subplots(figsize=(9, 5.2), dpi=300)
            ax_cdf_s.plot(sorted_ae, cdf_prob, color="#6A3D9A", linewidth=2.4, label="Empirical Error CDF")
            ax_cdf_s.axvline(p50, color="#2CA02C", linestyle="--", linewidth=1.5, label=f"Median (P50): {p50:.3f}")
            ax_cdf_s.axvline(p90, color="#FF7F0E", linestyle="--", linewidth=1.5, label=f"90th Pct (P90): {p90:.3f}")
            ax_cdf_s.axvline(p95, color="#D62728", linestyle="--", linewidth=1.5, label=f"95th Pct (P95): {p95:.3f}")
            ax_cdf_s.set_title(f"Cumulative Distribution of Absolute Error $|e(t)|$ — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
            ax_cdf_s.set_xlabel("Absolute Prediction Error $|e(t)|$", fontsize=11)
            ax_cdf_s.set_ylabel("Cumulative Percentage of Samples (%)", fontsize=11)
            ax_cdf_s.set_ylim(0, 102)
            ax_cdf_s.legend(loc="lower right", frameon=True, fontsize=9.5)
            ax_cdf_s.grid(True, linestyle="--", alpha=0.35)
            fig_cdf_s.savefig(os.path.join(output_dir, f"error_cdf_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig_cdf_s)

            # PLOT 4: Full-Horizon Overview & Rolling Window RMSE
            fig4, (ax_full, ax_roll) = plt.subplots(2, 1, figsize=(13, 7.5), dpi=300, sharex=True)

            step_stride = max(1, n_samples // 2500)
            t_idx = np.arange(n_samples)[::step_stride]
            ax_full.plot(t_idx, y_arr[::step_stride], color="#111111", linewidth=1.3, alpha=0.85, label="Actual Output $y(t)$")
            ax_full.plot(t_idx, yhat_arr[::step_stride], color="#1F77B4", linestyle="--", linewidth=1.3, alpha=0.85, label=r"Model Prediction $\hat{y}(t|\theta)$")
            ax_full.set_title(f"Full-Horizon Signal Tracking Overview (N = {n_samples} Aligned Samples) — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
            ax_full.set_ylabel("System Output", fontsize=11)
            ax_full.legend(loc="upper right", frameon=True, fontsize=9.5)
            ax_full.grid(True, linestyle="--", alpha=0.35)

            win_size = min(n_samples, max(1, min(150, max(10, n_samples // 20))))
            sq_err = residuals ** 2
            kernel_w = np.ones(win_size) / float(win_size)
            rolling_rmse = np.sqrt(np.convolve(sq_err, kernel_w, mode="same"))
            overall_rmse = float(np.sqrt(np.mean(sq_err)))

            ax_roll.plot(t_idx, rolling_rmse[::step_stride], color="#D62728", linewidth=1.6, label=f"Rolling RMSE (Window = {win_size} steps)")
            ax_roll.axhline(overall_rmse, color="black", linestyle="--", linewidth=1.5, label=f"Global RMSE ({overall_rmse:.4f})")
            ax_roll.fill_between(t_idx, 0, rolling_rmse[::step_stride], color="#D62728", alpha=0.15)
            ax_roll.set_title(f"Temporal Error Stability (Rolling {win_size}-Step RMSE)", fontsize=12, fontweight="bold")
            ax_roll.set_xlabel("Aligned Time Step (t)", fontsize=11)
            ax_roll.set_ylabel("Rolling RMSE", fontsize=11)
            ax_roll.legend(loc="upper right", frameon=True, fontsize=9.5)
            ax_roll.grid(True, linestyle="--", alpha=0.35)

            plt.tight_layout()
            fig4.savefig(os.path.join(output_dir, f"full_horizon_rolling_rmse_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig4)

            # Standalone Plot 4A: Full-Horizon Signal Tracking
            fig_full_s, ax_full_s = plt.subplots(figsize=(10.5, 5), dpi=300)
            ax_full_s.plot(t_idx, y_arr[::step_stride], color="#111111", linewidth=1.3, alpha=0.85, label="Actual Output $y(t)$")
            ax_full_s.plot(t_idx, yhat_arr[::step_stride], color="#1F77B4", linestyle="--", linewidth=1.3, alpha=0.85, label=r"Model Prediction $\hat{y}(t|\theta)$")
            ax_full_s.set_title(f"Full-Horizon Signal Tracking (N = {n_samples}) — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
            ax_full_s.set_xlabel("Aligned Time Step (t)", fontsize=11)
            ax_full_s.set_ylabel("System Output", fontsize=11)
            ax_full_s.legend(loc="upper right", frameon=True, fontsize=9.5)
            ax_full_s.grid(True, linestyle="--", alpha=0.35)
            fig_full_s.savefig(os.path.join(output_dir, f"full_horizon_tracking_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig_full_s)

            # Standalone Plot 4B: Rolling Window RMSE
            fig_roll_s, ax_roll_s = plt.subplots(figsize=(10.5, 5), dpi=300)
            ax_roll_s.plot(t_idx, rolling_rmse[::step_stride], color="#D62728", linewidth=1.6, label=f"Rolling RMSE (Window = {win_size} steps)")
            ax_roll_s.axhline(overall_rmse, color="black", linestyle="--", linewidth=1.5, label=f"Global RMSE ({overall_rmse:.4f})")
            ax_roll_s.fill_between(t_idx, 0, rolling_rmse[::step_stride], color="#D62728", alpha=0.15)
            ax_roll_s.set_title(f"Temporal Error Stability (Rolling {win_size}-Step RMSE) — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
            ax_roll_s.set_xlabel("Aligned Time Step (t)", fontsize=11)
            ax_roll_s.set_ylabel("Rolling RMSE", fontsize=11)
            ax_roll_s.legend(loc="upper right", frameon=True, fontsize=9.5)
            ax_roll_s.grid(True, linestyle="--", alpha=0.35)
            fig_roll_s.savefig(os.path.join(output_dir, f"rolling_rmse_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig_roll_s)
        finally:
            plt.close("all")


def generate_simulation_vs_prediction_plot(actual_targets, predicted_outputs, simulated_outputs, dataset_name, fold_num, output_dir="plots", title_suffix=""):
    """
    Compares 1-step ahead prediction with recursive free-run simulation.
    Shows if the model drifts over long horizons without ground-truth feedback.
    """
    with _PLOT_LOCK:
        try:
            _ensure_output_directory(output_dir)

            y_true = _extract_1d_channel(actual_targets, 0)
            y_pred = _extract_1d_channel(predicted_outputs, 0)
            y_sim = _extract_1d_channel(simulated_outputs, 0)

            n_samples = min(len(y_true), len(y_pred), len(y_sim))
            if n_samples == 0:
                return
            y_true = y_true[:n_samples]
            y_pred = y_pred[:n_samples]
            y_sim = y_sim[:n_samples]
            disp_len = min(n_samples, 250)
            t_steps = np.arange(disp_len)

            fig, (ax_track, ax_err) = plt.subplots(2, 1, figsize=(13, 7.5), dpi=300, sharex=True)

            ax_track.plot(t_steps, y_true[:disp_len], color="#111111", linewidth=2.0, label="True System Output $y(t)$")
            if np.any(np.isfinite(y_pred[:disp_len])):
                ax_track.plot(t_steps, y_pred[:disp_len], color="#FF7F0E", linestyle="--", linewidth=1.8, label=r"1-Step-Ahead Prediction $\hat{y}(t|t-1)$")
            if np.any(np.isfinite(y_sim[:disp_len])):
                ax_track.plot(t_steps, y_sim[:disp_len], color="#1F77B4", linestyle="-.", linewidth=1.8, label=r"Free-Run Autonomous Simulation $\hat{y}_{sim}(t)$")
            ax_track.set_title(f"Simulation vs. Prediction Validation — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=13, fontweight="bold")
            ax_track.set_ylabel("Output Signal", fontsize=11)
            ax_track.legend(loc="upper right", frameon=True, fontsize=10)
            ax_track.grid(True, linestyle="--", alpha=0.35)

            # Residuals comparison
            err_pred = y_true[:disp_len] - y_pred[:disp_len]
            err_sim = y_true[:disp_len] - y_sim[:disp_len]
            if np.any(np.isfinite(err_pred)):
                ax_err.plot(t_steps, err_pred, color="#FF7F0E", linewidth=1.5, label="1-Step Prediction Error $e(t|t-1)$")
            if np.any(np.isfinite(err_sim)):
                ax_err.plot(t_steps, err_sim, color="#1F77B4", linestyle="-.", linewidth=1.5, label="Free-Run Simulation Error $e_{sim}(t)$")
            ax_err.axhline(0, color="black", linestyle=":", linewidth=1.0)
            ax_err.set_title("Prediction vs. Simulation Residual Trajectories", fontsize=11.5, fontweight="bold")
            ax_err.set_xlabel("Time Step (t)", fontsize=11)
            ax_err.set_ylabel("Error Residual", fontsize=11)
            if ax_err.get_legend_handles_labels()[1]:
                ax_err.legend(loc="upper right", frameon=True, fontsize=10)
            ax_err.grid(True, linestyle="--", alpha=0.35)

            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"simulation_vs_prediction_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig)

            # Standalone Plot: Simulation vs. Prediction Signal Tracking
            fig_tr_s, ax_tr_s = plt.subplots(figsize=(10.5, 5.2), dpi=300)
            ax_tr_s.plot(t_steps, y_true[:disp_len], color="#111111", linewidth=2.0, label="True System Output $y(t)$")
            if np.any(np.isfinite(y_pred[:disp_len])):
                ax_tr_s.plot(t_steps, y_pred[:disp_len], color="#FF7F0E", linestyle="--", linewidth=1.8, label=r"1-Step-Ahead Prediction $\hat{y}(t|t-1)$")
            if np.any(np.isfinite(y_sim[:disp_len])):
                ax_tr_s.plot(t_steps, y_sim[:disp_len], color="#1F77B4", linestyle="-.", linewidth=1.8, label=r"Free-Run Autonomous Simulation $\hat{y}_{sim}(t)$")
            ax_tr_s.set_title(f"Autonomous Simulation vs. Prediction Tracking — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
            ax_tr_s.set_xlabel("Time Step (t)", fontsize=11)
            ax_tr_s.set_ylabel("Output Signal", fontsize=11)
            ax_tr_s.legend(loc="upper right", frameon=True, fontsize=10)
            ax_tr_s.grid(True, linestyle="--", alpha=0.35)
            fig_tr_s.savefig(os.path.join(output_dir, f"simulation_tracking_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig_tr_s)

            # Standalone Plot: Prediction vs Simulation Residuals
            fig_res_s, ax_res_s = plt.subplots(figsize=(10.5, 5.2), dpi=300)
            if np.any(np.isfinite(err_pred)):
                ax_res_s.plot(t_steps, err_pred, color="#FF7F0E", linewidth=1.5, label="1-Step Prediction Error $e(t|t-1)$")
            if np.any(np.isfinite(err_sim)):
                ax_res_s.plot(t_steps, err_sim, color="#1F77B4", linestyle="-.", linewidth=1.5, label="Free-Run Simulation Error $e_{sim}(t)$")
            ax_res_s.axhline(0, color="black", linestyle=":", linewidth=1.0)
            ax_res_s.set_title(f"Prediction vs. Simulation Residual Trajectories — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
            ax_res_s.set_xlabel("Time Step (t)", fontsize=11)
            ax_res_s.set_ylabel("Error Residual", fontsize=11)
            if ax_res_s.get_legend_handles_labels()[1]:
                ax_res_s.legend(loc="upper right", frameon=True, fontsize=10)
            ax_res_s.grid(True, linestyle="--", alpha=0.35)
            fig_res_s.savefig(os.path.join(output_dir, f"simulation_residuals_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig_res_s)
        finally:
            plt.close("all")


def generate_bode_diagram(theta_weights, na, nb, nc, nk, dataset_name, fold_num, output_dir="plots", title_suffix="", n_u=1, n_y=1):
    """
    Plots the frequency response (Bode plot) and pole-zero distribution.
    Shows the frequency behavior and checks if the system poles lie inside the unit circle.
    """
    with _PLOT_LOCK:
        try:
            _ensure_output_directory(output_dir)

            na = max(0, int(na or 0))
            nb = max(0, int(nb or 0))
            nc = max(0, int(nc or 0))
            nk = max(0, int(nk or 0))
            n_u = max(1, int(n_u or 1))
            n_y = max(1, int(n_y or 1))

            theta = np.asarray(theta_weights, dtype=float).flatten()
            params_per_output = na + (nb * n_u) + nc
            if len(theta) < params_per_output or nb == 0:
                return

            outputs_to_plot = range(n_y) if n_y > 1 else [0]
            colors = ["#1F77B4", "#2CA02C", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]

            for out_ch in outputs_to_plot:
                offset = out_ch * params_per_output
                if len(theta) < offset + params_per_output:
                    break
                theta_ar = theta[offset: offset + na]
                theta_b_all = theta[offset + na: offset + na + (nb * n_u)]
                a_poly = np.concatenate([[1.0], theta_ar])
                channel_tag = f" (Output Channel $y_{{{out_ch + 1}}}$)" if n_y > 1 else ""
                file_out_tag = f"_out_{out_ch + 1}" if n_y > 1 else ""

                try:
                    fig, (ax_mag, ax_phase, ax_pz) = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

                    # Plot each input channel's frequency response
                    for in_idx in range(n_u):
                        b_slice = theta_b_all[in_idx * nb: (in_idx + 1) * nb]
                        b_poly = np.concatenate([np.zeros(nk), b_slice]) if nk > 0 else b_slice

                        w, h = sp_signal.freqz(b_poly, a_poly, worN=512)
                        mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
                        phase_deg = np.unwrap(np.angle(h)) * (180.0 / np.pi)

                        ch_color = colors[in_idx % len(colors)]
                        ch_label = f"Input $u_{{{in_idx + 1}}}$" if n_u > 1 else "Magnitude (dB)"
                        ax_mag.plot(w / np.pi, mag_db, color=ch_color, linewidth=1.8, label=ch_label)

                        ch_phase_label = f"Input $u_{{{in_idx + 1}}}$" if n_u > 1 else "Phase (deg)"
                        ax_phase.plot(w / np.pi, phase_deg, color=ch_color, linewidth=1.8, label=ch_phase_label)

                        # Plot zeros for each channel (skip all-zero polynomials to avoid numpy LinAlg warnings)
                        ch_zeros = np.roots(b_poly) if (len(b_poly) > 1 and not np.all(b_poly == 0)) else np.array([])
                        if len(ch_zeros) > 0:
                            z_lbl = f"Zeros ($u_{{{in_idx + 1}}}$)" if n_u > 1 and in_idx == 0 else ("Zeros" if n_u == 1 else None)
                            ax_pz.scatter(np.real(ch_zeros), np.imag(ch_zeros), marker="o", facecolors="none",
                                          edgecolors=ch_color, s=80, linewidth=1.8, label=z_lbl)

                    ax_mag.set_title(f"Bode Magnitude{channel_tag}", fontsize=12, fontweight="bold")
                    ax_mag.set_xlabel(r"Normalized Frequency ($\times \pi$ rad/sample)", fontsize=10)
                    ax_mag.set_ylabel("Magnitude (dB)", fontsize=10)
                    if n_u > 1:
                        ax_mag.legend(loc="best", frameon=True, fontsize=8.5)
                    ax_mag.grid(True, linestyle="--", alpha=0.35)

                    ax_phase.set_title(f"Bode Phase{channel_tag}", fontsize=12, fontweight="bold")
                    ax_phase.set_xlabel(r"Normalized Frequency ($\times \pi$ rad/sample)", fontsize=10)
                    ax_phase.set_ylabel("Phase (deg)", fontsize=10)
                    if n_u > 1:
                        ax_phase.legend(loc="best", frameon=True, fontsize=8.5)
                    ax_phase.grid(True, linestyle="--", alpha=0.35)

                    # Poles of shared denominator polynomial A(q)
                    try:
                        poles = np.roots(a_poly) if (np.all(np.isfinite(a_poly)) and len(a_poly) > 1) else np.array([])
                    except Exception:
                        poles = np.array([])

                    # Unit circle
                    circ = plt.Circle((0, 0), 1.0, color="#888888", fill=False, linestyle="--", linewidth=1.5, label="Unit Circle ($|z|=1$)")
                    ax_pz.add_patch(circ)

                    if len(poles) > 0:
                        ax_pz.scatter(np.real(poles), np.imag(poles), marker="x", color="#D62728", s=90, linewidth=2.2,
                                      label=f"Poles (max={np.max(np.abs(poles)):.3f})")

                    ax_pz.set_xlim(-1.6, 1.6)
                    ax_pz.set_ylim(-1.6, 1.6)
                    ax_pz.set_aspect("equal", adjustable="box")
                    ax_pz.axhline(0, color="gray", linewidth=0.8, linestyle=":")
                    ax_pz.axvline(0, color="gray", linewidth=0.8, linestyle=":")
                    ax_pz.set_title(f"Pole-Zero Constellation{channel_tag}", fontsize=12, fontweight="bold")
                    ax_pz.set_xlabel("Real Part", fontsize=10)
                    ax_pz.set_ylabel("Imaginary Part", fontsize=10)
                    ax_pz.legend(loc="upper right", frameon=True, fontsize=8.5)
                    ax_pz.grid(True, linestyle="--", alpha=0.35)

                    plt.suptitle(f"Frequency-Domain Analysis — {dataset_name}{channel_tag} (Fold {fold_num}){title_suffix}", fontsize=13, fontweight="bold")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    bode_pz_name = f"bode_pole_zero_{dataset_name}_fold_{fold_num}{file_out_tag}.png"
                    fig.savefig(os.path.join(output_dir, bode_pz_name), bbox_inches="tight", dpi=300)
                    if out_ch == 0 and n_y > 1:
                        # Also save primary channel under default legacy name for 100% backward compatibility
                        fig.savefig(os.path.join(output_dir, f"bode_pole_zero_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
                    plt.close(fig)

                    # Standalone Bode Magnitude
                    fig_mag_s, ax_m_s = plt.subplots(figsize=(9.5, 5.2), dpi=300)
                    for in_idx in range(n_u):
                        b_slice = theta_b_all[in_idx * nb: (in_idx + 1) * nb]
                        b_poly = np.concatenate([np.zeros(nk), b_slice]) if nk > 0 else b_slice
                        w, h = sp_signal.freqz(b_poly, a_poly, worN=512)
                        mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
                        ch_color = colors[in_idx % len(colors)]
                        ch_label = f"Input $u_{{{in_idx + 1}}}$" if n_u > 1 else "Magnitude (dB)"
                        ax_m_s.plot(w / np.pi, mag_db, color=ch_color, linewidth=2.0, label=ch_label)
                    ax_m_s.set_title(f"Bode Magnitude Response{channel_tag} — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
                    ax_m_s.set_xlabel(r"Normalized Frequency ($\times \pi$ rad/sample)", fontsize=11)
                    ax_m_s.set_ylabel("Magnitude (dB)", fontsize=11)
                    if n_u > 1:
                        ax_m_s.legend(loc="best", frameon=True, fontsize=9.5)
                    ax_m_s.grid(True, linestyle="--", alpha=0.35)
                    fig_mag_s.savefig(os.path.join(output_dir, f"bode_magnitude_{dataset_name}_fold_{fold_num}{file_out_tag}.png"), bbox_inches="tight", dpi=300)
                    if out_ch == 0 and n_y > 1:
                        fig_mag_s.savefig(os.path.join(output_dir, f"bode_magnitude_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
                    plt.close(fig_mag_s)

                    # Standalone Bode Phase
                    fig_pha_s, ax_p_s = plt.subplots(figsize=(9.5, 5.2), dpi=300)
                    for in_idx in range(n_u):
                        b_slice = theta_b_all[in_idx * nb: (in_idx + 1) * nb]
                        b_poly = np.concatenate([np.zeros(nk), b_slice]) if nk > 0 else b_slice
                        w, h = sp_signal.freqz(b_poly, a_poly, worN=512)
                        phase_deg = np.unwrap(np.angle(h)) * (180.0 / np.pi)
                        ch_color = colors[in_idx % len(colors)]
                        ch_label = f"Input $u_{{{in_idx + 1}}}$" if n_u > 1 else "Phase (deg)"
                        ax_p_s.plot(w / np.pi, phase_deg, color=ch_color, linewidth=2.0, label=ch_label)
                    ax_p_s.set_title(f"Bode Phase Response{channel_tag} — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
                    ax_p_s.set_xlabel(r"Normalized Frequency ($\times \pi$ rad/sample)", fontsize=11)
                    ax_p_s.set_ylabel("Phase (deg)", fontsize=11)
                    if n_u > 1:
                        ax_p_s.legend(loc="best", frameon=True, fontsize=9.5)
                    ax_p_s.grid(True, linestyle="--", alpha=0.35)
                    fig_pha_s.savefig(os.path.join(output_dir, f"bode_phase_{dataset_name}_fold_{fold_num}{file_out_tag}.png"), bbox_inches="tight", dpi=300)
                    if out_ch == 0 and n_y > 1:
                        fig_pha_s.savefig(os.path.join(output_dir, f"bode_phase_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
                    plt.close(fig_pha_s)

                    # Standalone Pole-Zero Map
                    fig_pz_s, ax_pz_s = plt.subplots(figsize=(7, 7), dpi=300)
                    circ_s = plt.Circle((0, 0), 1.0, color="#888888", fill=False, linestyle="--", linewidth=1.5, label="Unit Circle ($|z|=1$)")
                    ax_pz_s.add_patch(circ_s)
                    for in_idx in range(n_u):
                        b_slice = theta_b_all[in_idx * nb: (in_idx + 1) * nb]
                        b_poly = np.concatenate([np.zeros(nk), b_slice]) if nk > 0 else b_slice
                        ch_zeros = np.roots(b_poly) if (len(b_poly) > 1 and not np.all(b_poly == 0)) else np.array([])
                        if len(ch_zeros) > 0:
                            z_lbl = f"Zeros ($u_{{{in_idx + 1}}}$)" if n_u > 1 and in_idx == 0 else ("Zeros" if n_u == 1 else None)
                            ax_pz_s.scatter(np.real(ch_zeros), np.imag(ch_zeros), marker="o", facecolors="none",
                                            edgecolors=colors[in_idx % len(colors)], s=85, linewidth=1.8, label=z_lbl)
                    if len(poles) > 0:
                        ax_pz_s.scatter(np.real(poles), np.imag(poles), marker="x", color="#D62728", s=95, linewidth=2.2,
                                        label=f"Poles (max={np.max(np.abs(poles)):.3f})")
                    ax_pz_s.set_xlim(-1.6, 1.6)
                    ax_pz_s.set_ylim(-1.6, 1.6)
                    ax_pz_s.set_aspect("equal", adjustable="box")
                    ax_pz_s.axhline(0, color="gray", linewidth=0.8, linestyle=":")
                    ax_pz_s.axvline(0, color="gray", linewidth=0.8, linestyle=":")
                    ax_pz_s.set_title(f"Pole-Zero Constellation{channel_tag} — {dataset_name} (Fold {fold_num}){title_suffix}", fontsize=12.5, fontweight="bold")
                    ax_pz_s.set_xlabel("Real Part", fontsize=11)
                    ax_pz_s.set_ylabel("Imaginary Part", fontsize=11)
                    ax_pz_s.legend(loc="upper right", frameon=True, fontsize=9.5)
                    ax_pz_s.grid(True, linestyle="--", alpha=0.35)
                    fig_pz_s.savefig(os.path.join(output_dir, f"pole_zero_map_{dataset_name}_fold_{fold_num}{file_out_tag}.png"), bbox_inches="tight", dpi=300)
                    if out_ch == 0 and n_y > 1:
                        fig_pz_s.savefig(os.path.join(output_dir, f"pole_zero_map_{dataset_name}_fold_{fold_num}.png"), bbox_inches="tight", dpi=300)
                    plt.close(fig_pz_s)
                except Exception as e:
                    print(f"[!] Warning: Could not generate Bode plot for channel {out_ch + 1}: {e}")
        finally:
            plt.close("all")


def generate_benchmark_convergence_comparison(
    convergence_histories_dict,
    output_dir="results_benchmarks",
    filename="benchmark_convergence_comparison.png",
    suptitle="Benchmark Convergence Comparison (Mean +/- 1 SD Across Trials)"
):
    """
    Plots multi-trial convergence curves (Mean +/- 1 SD in log scale) across all benchmark functions
    comparing EP against Differential Evolution (or comparing different operator configurations).
    convergence_histories_dict: {
        "Sphere": {"EP (Gaussian)": np.ndarray(trials, gens), "Differential Evolution": np.ndarray(trials, gens)},
        ...
    }
    """
    with _PLOT_LOCK:
        try:
            _ensure_output_directory(output_dir)
            if not convergence_histories_dict:
                return

            func_names = list(convergence_histories_dict.keys())
            n_funcs = len(func_names)
            cols = 3
            rows = max(1, (n_funcs + cols - 1) // cols)

            fig, axes = plt.subplots(rows, cols, figsize=(16, 5.2 * rows), dpi=300)
            axes_flat = np.atleast_1d(axes).flatten()
            colors = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E", "#8C564B"]
            linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

            for idx, f_name in enumerate(func_names):
                ax = axes_flat[idx]
                algo_map = convergence_histories_dict[f_name]
                for a_idx, (algo_label, mat) in enumerate(algo_map.items()):
                    arr = np.asarray(mat, dtype=float)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    safe_arr = np.clip(np.nan_to_num(arr, nan=1e-18, posinf=1e6, neginf=1e-18), 1e-18, 1e9)
                    mean_curve = np.mean(safe_arr, axis=0)
                    # Compute symmetric dispersion envelope in log10 space so +/- 1 SD never turns negative
                    # or drops vertically to the bottom of the logarithmic axis
                    log_std = np.std(np.log10(safe_arr), axis=0)
                    lower_band = np.maximum(mean_curve * (10.0 ** (-log_std)), 1e-18)
                    upper_band = np.minimum(mean_curve * (10.0 ** log_std), 1e9)
                    gens = np.arange(len(mean_curve))

                    c = colors[a_idx % len(colors)]
                    ls = linestyles[a_idx % len(linestyles)]
                    ax.plot(gens, mean_curve, label=algo_label, color=c, linestyle=ls, linewidth=2.0)
                    ax.fill_between(
                        gens,
                        lower_band,
                        upper_band,
                        color=c, alpha=0.16
                    )

                ax.set_title(f"{f_name}", fontsize=12.5, fontweight="bold")
                ax.set_xlabel("Generation", fontsize=10.5)
                ax.set_ylabel("Best Objective Value [Log Scale]", fontsize=10.5)
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.4g}"))
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
                ax.grid(True, which="both", linestyle="--", alpha=0.35)
                ax.legend(loc="upper right", frameon=True, fontsize=8.5)

            for extra_ax in axes_flat[n_funcs:]:
                extra_ax.set_visible(False)

            plt.suptitle(suptitle, fontsize=14, fontweight="bold")
            plt.tight_layout(rect=[0, 0.02, 1, 0.95])
            fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=300)
            plt.close(fig)
        finally:
            plt.close("all")


def generate_sensitivity_analysis_plots(
    sensitivity_records,
    output_dir="results_sensitivity",
    metric_key="RMSE",
    metric_label="Validation RMSE",
    filename="parameter_sensitivity_overview.png",
    title_suffix=""
):
    """
    Plots One-Factor-At-A-Time (OFAT) parameter sensitivity profiles showing both
    solution quality (mean +/- SD) and execution time (seconds) for each tested parameter.
    """
    with _PLOT_LOCK:
        try:
            _ensure_output_directory(output_dir)
            if not sensitivity_records:
                return

            grouped = {}
            for rec in sensitivity_records:
                p_name = rec.get("parameter_name", "Parameter")
                grouped.setdefault(p_name, []).append(rec)

            param_names = list(grouped.keys())
            n_params = len(param_names)
            cols = min(3, n_params)
            rows = max(1, (n_params + cols - 1) // cols)

            fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 4.8 * rows), dpi=300)
            axes_flat = np.atleast_1d(axes).flatten()

            for idx, p_name in enumerate(param_names):
                ax1 = axes_flat[idx]
                items = grouped[p_name]
                x_labels = [str(it["parameter_value"]) for it in items]
                x_pos = np.arange(len(x_labels))

                y_mean = np.array([float(it.get(f"{metric_key}_mean", 0.0)) for it in items], dtype=float)
                y_std = np.array([float(it.get(f"{metric_key}_std", 0.0)) for it in items], dtype=float)
                t_mean = np.array([float(it.get("time_mean", 0.0)) for it in items], dtype=float)

                ax1.errorbar(
                    x_pos, y_mean, yerr=y_std, fmt="-o", color="#D62728",
                    ecolor="#D62728", elinewidth=1.6, capsize=4, linewidth=2.0, markersize=6,
                    label=f"{metric_label} (Mean +/- SD)"
                )
                ax1.set_title(f"Sensitivity: {p_name}", fontsize=12, fontweight="bold")
                ax1.set_xlabel(p_name, fontsize=10.5)
                ax1.set_ylabel(metric_label, color="#D62728", fontsize=10.5)
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(x_labels)
                ax1.tick_params(axis="y", labelcolor="#D62728")
                ax1.grid(True, linestyle="--", alpha=0.35)

                ax2 = ax1.twinx()
                ax2.plot(x_pos, t_mean, "--s", color="#1F77B4", linewidth=1.8, markersize=5.5, label="Exec Time (s)")
                ax2.set_ylabel("Mean Execution Time (s)", color="#1F77B4", fontsize=10.5)
                ax2.tick_params(axis="y", labelcolor="#1F77B4")

            for extra_ax in axes_flat[n_params:]:
                extra_ax.set_visible(False)

            plt.suptitle(f"Hyperparameter Sensitivity Analysis ({metric_label} vs. Execution Time){title_suffix}", fontsize=13.5, fontweight="bold")
            plt.tight_layout(rect=[0, 0.02, 1, 0.95])
            fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=300)
            plt.close(fig)
        finally:
            plt.close("all")


def generate_dimensional_scalability_plots(
    scalability_records,
    output_dir="results_scalability",
    filename="study_3_dimensional_scalability.png"
):
    """
    Plots a 2x3 grid of Final Objective Value (Mean +/- 1 SD in log10 space) vs. Problem Dimension D
    comparing Self-Adaptive EP and Differential Evolution across all 6 benchmark topologies.
    """
    with _PLOT_LOCK:
        try:
            _ensure_output_directory(output_dir)
            if not scalability_records:
                return

            grouped = {}
            for rec in scalability_records:
                fn = rec["Function"]
                grouped.setdefault(fn, []).append(rec)

            func_names = list(grouped.keys())
            n_funcs = len(func_names)
            cols = 3
            rows = max(1, (n_funcs + cols - 1) // cols)

            fig, axes = plt.subplots(rows, cols, figsize=(16, 5.2 * rows), dpi=300)
            axes_flat = np.atleast_1d(axes).flatten()

            for idx, fn in enumerate(func_names):
                ax = axes_flat[idx]
                recs = sorted(grouped[fn], key=lambda r: int(r["Dimensions"]))
                dims = np.array([int(r["Dimensions"]) for r in recs], dtype=int)

                ep_means = np.clip(np.array([float(r["EP_Mean"]) for r in recs], dtype=float), 1e-18, 1e9)
                ep_stds = np.array([float(r["EP_Std"]) for r in recs], dtype=float)
                de_means = np.clip(np.array([float(r["DE_Mean"]) for r in recs], dtype=float), 1e-18, 1e9)
                de_stds = np.array([float(r["DE_Std"]) for r in recs], dtype=float)

                ax.plot(dims, ep_means, "-o", color="#D62728", linewidth=2.0, markersize=6, label="Self-Adaptive EP")
                ax.fill_between(
                    dims,
                    np.maximum(ep_means * 0.5, ep_means - 0.5 * ep_stds),
                    ep_means + ep_stds,
                    color="#D62728", alpha=0.15
                )

                ax.plot(dims, de_means, "--s", color="#1F77B4", linewidth=2.0, markersize=6, label="Differential Evolution (DE)")
                ax.fill_between(
                    dims,
                    np.maximum(de_means * 0.5, de_means - 0.5 * de_stds),
                    de_means + de_stds,
                    color="#1F77B4", alpha=0.15
                )

                ax.set_title(f"{fn} — Dimensional Scaling", fontsize=12.5, fontweight="bold")
                ax.set_xlabel("Problem Dimension (D)", fontsize=10.5)
                ax.set_ylabel("Mean Final Objective [Log Scale]", fontsize=10.5)
                ax.set_xticks(dims)
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.4g}"))
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
                ax.grid(True, which="both", linestyle="--", alpha=0.35)
                ax.legend(loc="best", frameon=True, fontsize=8.8)

            for extra_ax in axes_flat[n_funcs:]:
                extra_ax.set_visible(False)

            plt.suptitle("Study 3: Dimensional Scalability Analysis (EP vs. DE Across D = 2..30 under Fixed N_pop)", fontsize=14, fontweight="bold")
            plt.tight_layout(rect=[0, 0.02, 1, 0.95])
            fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=300)
            plt.close(fig)
        finally:
            plt.close("all")


def generate_sysid_study_comparison_plots(
    study_histories_dict,
    fold_rmse_dict,
    output_dir="results_study",
    filename="sysid_study_comparison.png",
    suptitle="System Identification Study: Convergence (FE & Wall-Clock Time) and Fold RMSE Distribution"
):
    """
    Generates a 3-panel comparative figure for SysId experimental studies (Study 1 & Study 4):
      (a) Validation/Training MSE Convergence vs. Fitness Evaluations (FE)
      (b) Validation/Training MSE Convergence vs. Wall-Clock Time (s)
      (c) 10-Fold Validation RMSE Boxplot + Individual Fold Scatter
    """
    with _PLOT_LOCK:
        try:
            _ensure_output_directory(output_dir)
            if not study_histories_dict and not fold_rmse_dict:
                return

            fig, (ax_fe, ax_time, ax_box) = plt.subplots(1, 3, figsize=(18.5, 5.6), dpi=300)
            colors = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E", "#8C564B"]
            linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

            min_final_time = None
            for idx, (label, data_dict) in enumerate(study_histories_dict.items()):
                mse_mat = np.asarray(data_dict.get("mse_matrix", []), dtype=float)
                fe_axis = np.asarray(data_dict.get("fe_axis", []), dtype=float)
                time_axis = np.asarray(data_dict.get("time_axis", []), dtype=float)
                if mse_mat.size == 0:
                    continue
                if mse_mat.ndim == 1:
                    mse_mat = mse_mat.reshape(1, -1)

                safe_mat = np.clip(np.nan_to_num(mse_mat, nan=1e-12, posinf=1e6, neginf=1e-12), 1e-12, 1e9)
                mean_curve = np.mean(safe_mat, axis=0)
                log_std = np.std(np.log10(safe_mat), axis=0)
                lower_band = np.maximum(mean_curve * (10.0 ** (-log_std)), 1e-12)
                upper_band = np.minimum(mean_curve * (10.0 ** log_std), 1e9)

                if len(fe_axis) != len(mean_curve):
                    fe_axis = np.linspace(150, 150 * len(mean_curve), len(mean_curve))
                if len(time_axis) != len(mean_curve):
                    time_axis = np.linspace(0.0, float(data_dict.get("mean_time", len(mean_curve))), len(mean_curve))

                if time_axis[-1] > 0:
                    min_final_time = time_axis[-1] if min_final_time is None else min(min_final_time, time_axis[-1])

                c = colors[idx % len(colors)]
                ls = linestyles[idx % len(linestyles)]
                clean_lbl = str(label).replace("_", "-")

                # Panel (a): MSE vs. FE
                ax_fe.plot(fe_axis, mean_curve, label=clean_lbl, color=c, linestyle=ls, linewidth=2.0)
                ax_fe.fill_between(fe_axis, lower_band, upper_band, color=c, alpha=0.14)

                # Panel (b): MSE vs. Wall-Clock Time (s)
                ax_time.plot(time_axis, mean_curve, label=f"{clean_lbl} ({time_axis[-1]:.1f}s)", color=c, linestyle=ls, linewidth=2.0)
                ax_time.fill_between(time_axis, lower_band, upper_band, color=c, alpha=0.14)

            if min_final_time is not None and min_final_time > 0:
                ax_time.axvline(
                    min_final_time, color="#333333", linestyle=":", linewidth=1.4,
                    label=f"Fastest Budget Cutoff ({min_final_time:.1f}s)"
                )

            ax_fe.set_title("(a) Convergence vs. Fitness Evaluations", fontsize=12, fontweight="bold")
            ax_fe.set_xlabel("Cumulative Fitness Evaluations (FE)", fontsize=10.5)
            ax_fe.set_ylabel("Best Validation/Training MSE [Log Scale]", fontsize=10.5)
            ax_fe.set_yscale("log")
            ax_fe.grid(True, which="both", linestyle="--", alpha=0.35)
            ax_fe.legend(loc="upper right", frameon=True, fontsize=8.5)

            ax_time.set_title("(b) Convergence vs. Wall-Clock Time", fontsize=12, fontweight="bold")
            ax_time.set_xlabel("Mean Wall-Clock Time per Fold (s)", fontsize=10.5)
            ax_time.set_ylabel("Best Validation/Training MSE [Log Scale]", fontsize=10.5)
            ax_time.set_yscale("log")
            ax_time.grid(True, which="both", linestyle="--", alpha=0.35)
            ax_time.legend(loc="upper right", frameon=True, fontsize=8.5)

            # Panel (c): 10-Fold Validation RMSE Boxplot
            if fold_rmse_dict:
                labels = list(fold_rmse_dict.keys())
                short_labels = [str(k).replace("_", "\n") for k in labels]
                distributions = [np.asarray(fold_rmse_dict[k], dtype=float) for k in labels]
                try:
                    bp = ax_box.boxplot(distributions, tick_labels=short_labels, patch_artist=True, widths=0.5)
                except TypeError:
                    bp = ax_box.boxplot(distributions, labels=short_labels, patch_artist=True, widths=0.5)

                for i, box in enumerate(bp["boxes"]):
                    c = colors[i % len(colors)]
                    box.set(facecolor=c, alpha=0.72, edgecolor="black")
                    y_vals = distributions[i]
                    x_jitter = np.random.normal(i + 1, 0.05, size=len(y_vals))
                    ax_box.scatter(x_jitter, y_vals, color="black", alpha=0.65, s=18, zorder=5)
                for median in bp["medians"]:
                    median.set(color="black", linewidth=1.8)

                ax_box.set_title("(c) 10-Fold Validation RMSE Distribution", fontsize=12, fontweight="bold")
                ax_box.set_ylabel("Validation RMSE", fontsize=10.5)
                ax_box.tick_params(axis="x", labelsize=8.0)
                ax_box.grid(True, axis="y", linestyle="--", alpha=0.35)

            plt.suptitle(suptitle, fontsize=13.5, fontweight="bold")
            plt.tight_layout(rect=[0, 0.02, 1, 0.94])
            fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=300)
            plt.close(fig)
        finally:
            plt.close("all")


def generate_studies_overview_dashboard(
    df_s1_sysid=None,
    df_s2_sens=None,
    df_s3_scale=None,
    df_s4_fair=None,
    output_dir="results_studies_overview",
    filename="studies_overview_dashboard.png"
):
    """
    Generates a 2x2 overview dashboard comparing all 4 experimental studies:
      (a) Study 1: Operator Ablation Validation RMSE (Mean +/- SD)
      (b) Study 2: Hyperparameter Sensitivity Impact Range (Delta RMSE across levels)
      (c) Study 3: Dimensional Scalability (Overall & Multimodal A12 Effect Size across D)
      (d) Study 4: Equal-Budget Model Comparison (Validation RMSE vs. ms/FE cost)
    """
    with _PLOT_LOCK:
        try:
            _ensure_output_directory(output_dir)
            fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
            ax_s1, ax_s2, ax_s3, ax_s4 = axes.flatten()

            # Panel 1: Study 1 Operator Ablation (Supports Single Model or Multi-Model Grouped Bars with Per-Model Best Highlighted)
            if df_s1_sysid is not None and not df_s1_sysid.empty and "RMSE_Mean" in df_s1_sysid.columns:
                models_in_s1 = list(df_s1_sysid["Model"].unique()) if "Model" in df_s1_sysid.columns else ["Model"]
                if len(models_in_s1) <= 1:
                    labels = [str(c).replace("_", "\n") for c in df_s1_sysid["Configuration"]]
                    x_pos = np.arange(len(labels))
                    means = df_s1_sysid["RMSE_Mean"].values.astype(float)
                    stds = df_s1_sysid["RMSE_Std"].values.astype(float) if "RMSE_Std" in df_s1_sysid.columns else np.zeros_like(means)
                    bars = ax_s1.bar(x_pos, means, yerr=stds, capsize=4, color="#3B8ED0", alpha=0.85, edgecolor="black", width=0.55)
                    best_i = int(np.argmin(means))
                    bars[best_i].set_facecolor("#2CA02C")
                    m_tag = f" [{models_in_s1[0]}]" if models_in_s1[0] != "Model" else ""
                    ax_s1.set_xticks(x_pos)
                    ax_s1.set_xticklabels(labels, fontsize=8.0)
                    ax_s1.set_title(f"(a) Study 1: Operator Ablation{m_tag} (Green = Best)", fontsize=12, fontweight="bold")
                else:
                    configs = list(df_s1_sysid["Configuration"].unique())
                    short_cfgs = [str(c).replace("_", "\n") for c in configs]
                    x_pos = np.arange(len(configs))
                    n_m = len(models_in_s1)
                    bar_w = 0.72 / n_m
                    m_colors = ["#1F77B4", "#FF7F0E", "#9467BD", "#8C564B"]
                    for m_idx, m_name in enumerate(models_in_s1):
                        m_df = df_s1_sysid[df_s1_sysid["Model"] == m_name].set_index("Configuration")
                        means = np.array([float(m_df.loc[c, "RMSE_Mean"]) if c in m_df.index else np.nan for c in configs], dtype=float)
                        stds = np.array([float(m_df.loc[c, "RMSE_Std"]) if (c in m_df.index and "RMSE_Std" in m_df.columns) else 0.0 for c in configs], dtype=float)
                        offset = (m_idx - (n_m - 1) / 2.0) * bar_w
                        bars = ax_s1.bar(
                            x_pos + offset, means, yerr=stds, capsize=2.5,
                            width=bar_w * 0.92, color=m_colors[m_idx % len(m_colors)],
                            alpha=0.82, edgecolor="black", label=str(m_name)
                        )
                        if np.any(np.isfinite(means)):
                            best_i = int(np.nanargmin(means))
                            bars[best_i].set_facecolor("#2CA02C")
                            bars[best_i].set_linewidth(1.8)
                    ax_s1.set_xticks(x_pos)
                    ax_s1.set_xticklabels(short_cfgs, fontsize=7.5)
                    ax_s1.legend(loc="upper right", frameon=True, fontsize=8.5)
                    ax_s1.set_title("(a) Study 1: Operator Ablation by Model (Green = Model Best)", fontsize=11.5, fontweight="bold")
                ax_s1.set_ylabel("10-Fold Validation RMSE", fontsize=10.5)
                ax_s1.grid(True, axis="y", linestyle="--", alpha=0.35)
            else:
                ax_s1.text(0.5, 0.5, "Study 1 Data Not Available", ha="center", va="center", fontsize=11)
                ax_s1.set_title("(a) Study 1: Evolutionary Operator Ablation", fontsize=12, fontweight="bold")

            # Panel 2: Study 2 Hyperparameter Sensitivity Impact (Per-Model Delta RMSE & Optimal Values)
            if df_s2_sens is not None and not df_s2_sens.empty and "RMSE_mean" in df_s2_sens.columns:
                models_in_s2 = list(df_s2_sens["Model"].unique()) if "Model" in df_s2_sens.columns else ["Model"]
                if len(models_in_s2) <= 1:
                    impacts = []
                    for p_name, grp in df_s2_sens.groupby("parameter_name", sort=False):
                        vals = grp["RMSE_mean"].values.astype(float)
                        span = float(np.max(vals) - np.min(vals))
                        best_v = grp.loc[grp["RMSE_mean"].idxmin(), "parameter_value"]
                        impacts.append((f"{p_name}\n(best={best_v:g})", span))
                    impacts.sort(key=lambda x: x[1], reverse=True)
                    p_labels = [x[0] for x in impacts]
                    p_spans = [x[1] for x in impacts]
                    y_pos = np.arange(len(p_labels))
                    ax_s2.barh(y_pos, p_spans, color="#FF7F0E", alpha=0.85, edgecolor="black", height=0.55)
                    ax_s2.set_yticks(y_pos)
                    ax_s2.set_yticklabels(p_labels, fontsize=9)
                    ax_s2.invert_yaxis()
                    m_tag = f" [{models_in_s2[0]}]" if models_in_s2[0] != "Model" else ""
                    ax_s2.set_title(f"(b) Study 2: Hyperparameter Sensitivity Ranking{m_tag}", fontsize=12, fontweight="bold")
                else:
                    param_names = list(df_s2_sens["parameter_name"].unique())
                    y_pos = np.arange(len(param_names))
                    n_m = len(models_in_s2)
                    bar_h = 0.72 / n_m
                    m_colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#9467BD"]
                    best_annotations = {p: [] for p in param_names}
                    for m_idx, m_name in enumerate(models_in_s2):
                        m_df = df_s2_sens[df_s2_sens["Model"] == m_name]
                        spans = []
                        for p_name in param_names:
                            p_grp = m_df[m_df["parameter_name"] == p_name]
                            if not p_grp.empty:
                                vals = p_grp["RMSE_mean"].values.astype(float)
                                sp = float(np.max(vals) - np.min(vals))
                                bv = p_grp.loc[p_grp["RMSE_mean"].idxmin(), "parameter_value"]
                                spans.append(sp)
                                best_annotations[p_name].append(f"{m_name}:{bv:g}")
                            else:
                                spans.append(0.0)
                        offset = (m_idx - (n_m - 1) / 2.0) * bar_h
                        ax_s2.barh(
                            y_pos + offset, spans, height=bar_h * 0.92,
                            color=m_colors[m_idx % len(m_colors)], alpha=0.85,
                            edgecolor="black", label=str(m_name)
                        )
                    y_labels = [f"{p}\n({', '.join(best_annotations[p])})" for p in param_names]
                    ax_s2.set_yticks(y_pos)
                    ax_s2.set_yticklabels(y_labels, fontsize=7.8)
                    ax_s2.invert_yaxis()
                    ax_s2.legend(loc="lower right", frameon=True, fontsize=8.5)
                    ax_s2.set_title("(b) Study 2: Per-Model OFAT Sensitivity & Optimal Levels", fontsize=11.5, fontweight="bold")
                ax_s2.set_xlabel(r"Validation RMSE Sensitivity Span ($\max - \min$ Within Model)", fontsize=10.0)
                ax_s2.grid(True, axis="x", linestyle="--", alpha=0.35)
            else:
                ax_s2.text(0.5, 0.5, "Study 2 Data Not Available", ha="center", va="center", fontsize=11)
                ax_s2.set_title("(b) Study 2: Hyperparameter Sensitivity Ranking", fontsize=12, fontweight="bold")

            # Panel 3: Study 3 Dimensional Scalability (Mean A12 & Multimodal Crossover vs D)
            if df_s3_scale is not None and not df_s3_scale.empty and "A12_Effect_Size" in df_s3_scale.columns:
                dim_grp = df_s3_scale.groupby("Dimensions", as_index=False).agg(
                    mean_a12=("A12_Effect_Size", "mean"),
                    ep_win_tie_pct=("Winner", lambda s: float(np.mean(s.isin(["EP", "Tie"])) * 100.0))
                ).sort_values("Dimensions")
                multimodal_df = df_s3_scale[df_s3_scale["Function"].isin(["Rastrigin", "Schwefel", "Griewank"])]
                mm_grp = multimodal_df.groupby("Dimensions", as_index=False).agg(
                    mm_a12=("A12_Effect_Size", "mean")
                ).sort_values("Dimensions")

                dims = dim_grp["Dimensions"].values.astype(int)
                ax_s3.plot(dims, dim_grp["mean_a12"].values, "-o", color="#1F77B4", linewidth=2.2, markersize=6, label="Overall Mean $A_{12}$ (All 6 Functions)")
                if not mm_grp.empty:
                    ax_s3.plot(mm_grp["Dimensions"].values.astype(int), mm_grp["mm_a12"].values, "-s", color="#D62728", linewidth=2.2, markersize=6, label="Multimodal Mean $A_{12}$ (Rastrigin/Schwefel/Griewank)")
                ax_s3.axhline(0.5, color="black", linestyle=":", linewidth=1.3, label="Neutral Threshold ($A_{12} = 0.50$)")
                ax_s3.set_xlabel("Problem Dimension (D)", fontsize=10.5)
                ax_s3.set_ylabel(r"Vargha-Delaney Effect Size $\hat{A}_{12}$ (>0.5 Favors EP)", fontsize=10.0)
                ax_s3.set_xticks(dims)
                ax_s3.set_ylim(-0.05, 1.05)
                ax_s3.set_title("(c) Study 3: EP vs. DE Effect Size Scaling Across D = 2..30", fontsize=12, fontweight="bold")
                ax_s3.legend(loc="upper left", frameon=True, fontsize=8.5)
                ax_s3.grid(True, linestyle="--", alpha=0.35)
            else:
                ax_s3.text(0.5, 0.5, "Study 3 Data Not Available", ha="center", va="center", fontsize=11)
                ax_s3.set_title("(c) Study 3: Dimensional Scalability Stress Test", fontsize=12, fontweight="bold")

            # Panel 4: Study 4 Equal-Budget Model Comparison (RMSE & ms/FE)
            if df_s4_fair is not None and not df_s4_fair.empty and "RMSE_Mean" in df_s4_fair.columns:
                m_labels = [str(m).replace("_", "\n") for m in df_s4_fair["Model_Variant"]]
                x_pos = np.arange(len(m_labels))
                rmse_m = df_s4_fair["RMSE_Mean"].values.astype(float)
                rmse_s = df_s4_fair["RMSE_Std"].values.astype(float) if "RMSE_Std" in df_s4_fair.columns else np.zeros_like(rmse_m)
                cost_ms = df_s4_fair["Cost_Per_FE_ms"].values.astype(float) if "Cost_Per_FE_ms" in df_s4_fair.columns else np.zeros_like(rmse_m)

                ax_s4.bar(x_pos - 0.18, rmse_m, yerr=rmse_s, capsize=4, width=0.34, color="#D62728", alpha=0.85, edgecolor="black", label="Validation RMSE")
                ax_s4.set_ylabel("Validation RMSE (Lower = Better)", color="#D62728", fontsize=10.5)
                ax_s4.tick_params(axis="y", labelcolor="#D62728")
                ax_s4.set_xticks(x_pos)
                ax_s4.set_xticklabels(m_labels, fontsize=9)
                ax_s4.grid(True, axis="y", linestyle="--", alpha=0.35)

                ax_s4_r = ax_s4.twinx()
                ax_s4_r.bar(x_pos + 0.18, cost_ms, width=0.34, color="#1F77B4", alpha=0.75, edgecolor="black", label="Cost (ms / FE)")
                ax_s4_r.set_ylabel("Computational Cost (ms / FE)", color="#1F77B4", fontsize=10.5)
                ax_s4_r.tick_params(axis="y", labelcolor="#1F77B4")
                ax_s4.set_title("(d) Study 4: Equal-Budget Model Comparison (RMSE vs. ms/FE)", fontsize=12, fontweight="bold")
            else:
                ax_s4.text(0.5, 0.5, "Study 4 Data Not Available", ha="center", va="center", fontsize=11)
                ax_s4.set_title("(d) Study 4: Equal-Budget Model Comparison", fontsize=12, fontweight="bold")

            plt.suptitle("Cross-Study Overview Dashboard (Studies 1 - 4)", fontsize=14.5, fontweight="bold")
            plt.tight_layout(rect=[0, 0.02, 1, 0.95])
            fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=300)
            plt.close(fig)
        finally:
            plt.close("all")
