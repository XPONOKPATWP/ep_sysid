"""
validation_metrics.py

Evaluation metrics and deep system identification diagnostics for ARMAX, MLP, and SVR models.
Supports SISO, MISO, SIMO, and MIMO signal shapes with channel-aware centering.
"""

import numpy as np
from sklearn.metrics import r2_score


def _align_signal_shapes(y_true, y_pred):
    # Align 1D (N,) and 2D single-column (N, 1) arrays so subtraction never creates an (N, N) broadcast
    if y_true is None or y_pred is None:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.ndim != yp.ndim:
        if yt.ndim == 1 and yp.ndim == 2 and yp.shape[1] == 1:
            yp = yp.ravel()
        elif yp.ndim == 1 and yt.ndim == 2 and yt.shape[1] == 1:
            yt = yt.ravel()
    common_len = min(len(yt), len(yp))
    if len(yt) != common_len:
        yt = yt[:common_len]
    if len(yp) != common_len:
        yp = yp[:common_len]
    return yt, yp


def compute_mse(actual_targets, predicted_outputs):
    # Calculates MSE across all samples and output channels
    try:
        y_t, y_p = _align_signal_shapes(actual_targets, predicted_outputs)
        if len(y_t) == 0 or len(y_p) == 0:
            return 1e6
        diff = y_t - y_p
        mse = float(np.mean(diff * diff))
        return mse if (np.isfinite(mse) and mse <= 1e12) else 1e6
    except (ValueError, TypeError):
        return 1e6


def compute_mape(actual_targets, predicted_outputs):
    # Mean Absolute Percentage Error ignoring near-zero targets (< 1e-5)
    y_true, y_pred = _align_signal_shapes(actual_targets, predicted_outputs)

    if len(y_true) == 0 or len(y_pred) == 0 or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        return 100.0

    valid_mask = np.abs(y_true) > 1e-5
    if not np.any(valid_mask):
        return 0.0

    pct_errors = np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])
    raw_mape = float(np.mean(pct_errors) * 100.0)
    return raw_mape if np.isfinite(raw_mape) else 1000.0


def _compute_channelwise_nrmse_fit(y_true, y_pred):
    # Calculates NRMSE (FIT percentage as used in MATLAB sysid) per output channel
    yt, yp = _align_signal_shapes(y_true, y_pred)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)

    if len(yt) == 0 or len(yp) == 0 or not np.isfinite(yt).all() or not np.isfinite(yp).all():
        return 0.0

    fits = []
    num_ch = min(yt.shape[1], yp.shape[1])
    for ch in range(num_ch):
        y_c = yt[:, ch]
        yp_c = yp[:, ch]
        denom = float(np.linalg.norm(y_c - np.mean(y_c)))
        err_norm = float(np.linalg.norm(y_c - yp_c))
        fit_c = (1.0 - (err_norm / denom)) * 100.0 if denom > 1e-12 else 0.0
        fits.append(fit_c)
    return float(np.mean(fits)) if fits else 0.0


def _acf_pass_rate_1d(residuals_1d, max_lag=40):
    # Percentage of autocorrelation lags inside the 95% confidence interval (+/- 1.96/sqrt(N))
    res = np.asarray(residuals_1d, dtype=float).ravel()
    if not np.all(np.isfinite(res)):
        res = res[np.isfinite(res)]
    n_samples = len(res)
    limit_lag = min(max_lag, n_samples - 1)
    if limit_lag < 1:
        return 100.0

    centered = res - np.mean(res)
    variance = np.sum(centered ** 2)
    if variance <= 1e-12:
        return 100.0

    acf_full = np.correlate(centered, centered, mode="full") / variance
    acf_lags = acf_full[n_samples: n_samples + limit_lag]
    conf_bound = 1.96 / np.sqrt(n_samples)

    within_bounds = np.sum(np.abs(acf_lags) <= conf_bound)
    return float((within_bounds / max(1, len(acf_lags))) * 100.0)


def compute_white_noise_pass_rate(actual_targets, predicted_outputs, max_lag=40):
    # White noise check averaged across channels
    y_t, y_p = _align_signal_shapes(actual_targets, predicted_outputs)
    residuals = y_t - y_p
    if residuals.ndim == 1:
        return _acf_pass_rate_1d(residuals, max_lag=max_lag)

    channel_rates = [
        _acf_pass_rate_1d(residuals[:, col], max_lag=max_lag)
        for col in range(residuals.shape[1])
    ]
    return float(np.mean(channel_rates)) if channel_rates else 100.0


def _ccf_pass_rate_1d(u_1d, res_1d, max_lag=40):
    # Cross-correlation check between an input channel and residual to test for unmodeled dynamics
    u_vec = np.asarray(u_1d, dtype=float).ravel()
    r_vec = np.asarray(res_1d, dtype=float).ravel()
    n_samples = min(len(u_vec), len(r_vec))
    if n_samples < 2:
        return 100.0
    u_sub = u_vec[:n_samples]
    r_sub = r_vec[:n_samples]
    mask = np.isfinite(u_sub) & np.isfinite(r_sub)
    if not np.all(mask):
        u_sub = u_sub[mask]
        r_sub = r_sub[mask]
        n_samples = len(u_sub)
    limit_lag = min(max_lag, n_samples - 1)
    if limit_lag < 1:
        return 100.0

    u_centered = u_sub - np.mean(u_sub)
    e_centered = r_sub - np.mean(r_sub)
    denom = np.sqrt(np.sum(u_centered ** 2) * np.sum(e_centered ** 2))
    if denom <= 1e-12:
        return 100.0

    ccf_full = np.correlate(e_centered, u_centered, mode="full") / denom
    mid = n_samples - 1
    ccf_window = ccf_full[mid - limit_lag: mid + limit_lag + 1]
    conf_bound = 1.96 / np.sqrt(n_samples)

    within_bounds = np.sum(np.abs(ccf_window) <= conf_bound)
    return float((within_bounds / max(1, len(ccf_window))) * 100.0)


def compute_cross_correlation_pass_rate(u_aligned, actual_targets, predicted_outputs, max_lag=40):
    # Computes average CCF pass rate across all input-output channel combinations
    if u_aligned is None or actual_targets is None or predicted_outputs is None:
        return 100.0
    u_arr = np.asarray(u_aligned, dtype=float)
    y_t, y_p = _align_signal_shapes(actual_targets, predicted_outputs)
    if len(u_arr) == 0 or len(y_t) == 0:
        return 100.0
    residuals = y_t - y_p

    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(-1, 1)
    if residuals.ndim == 1:
        residuals = residuals.reshape(-1, 1)

    pair_rates = []
    for u_col in range(u_arr.shape[1]):
        for y_col in range(residuals.shape[1]):
            pair_rates.append(_ccf_pass_rate_1d(u_arr[:, u_col], residuals[:, y_col], max_lag=max_lag))

    return float(np.mean(pair_rates)) if pair_rates else 100.0


def generate_validation_report(actual_targets, predicted_outputs, model_parameter_count=6, simulated_outputs=None):
    # Compiles model validation metrics (MSE, RMSE, MAPE, R2, AIC, BIC, and simulation fit)
    y_true, y_pred = _align_signal_shapes(actual_targets, predicted_outputs)

    sample_size = len(y_true)
    k = int(model_parameter_count or 6)
    mse_score = float(compute_mse(y_true, y_pred))
    rmse_score = float(np.sqrt(mse_score)) if (np.isfinite(mse_score) and mse_score < 1e12) else 1e6

    if not np.isfinite(mse_score) or mse_score <= 0 or sample_size <= 0:
        aic_score = 1e6 if not np.isfinite(mse_score) else -1e6
        aicc_score = aic_score
        bic_score = 1e6 if not np.isfinite(mse_score) else -1e6
        fpe_score = 1e6 if not np.isfinite(mse_score) else 0.0
    else:
        n_obs = int(y_true.size)
        log_mse = n_obs * np.log(mse_score)
        aic_score = float(log_mse + 2.0 * k)
        # Guard against zero or negative dof when parameter count k is large relative to sample size
        dof_aicc = max(int(0.1 * n_obs), n_obs - k - 1, 1)
        aicc_score = float(aic_score + (2.0 * k * (k + 1)) / dof_aicc)
        bic_score = float(log_mse + k * np.log(max(1, n_obs)))
        dof_fpe = max(int(0.1 * n_obs), n_obs - k, 1)
        fpe_score = float(mse_score * (n_obs + k) / dof_fpe)

    try:
        if sample_size == 0 or float(np.var(y_true)) <= 1e-12:
            r2_val = 0.0
        else:
            r2_val = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
            if not np.isfinite(r2_val):
                r2_val = -1.0
    except Exception:
        r2_val = -1.0

    report = {
        "MSE": mse_score,
        "RMSE": rmse_score,
        "MAPE": compute_mape(y_true, y_pred),
        "R-Squared": r2_val,
        "AIC": aic_score,
        "AICc": aicc_score,
        "BIC": bic_score,
        "FPE": fpe_score,
        "White_Noise_Pct": compute_white_noise_pass_rate(y_true, y_pred)
    }

    if simulated_outputs is not None:
        y_true_sim, y_sim = _align_signal_shapes(actual_targets, simulated_outputs)
        sim_mse = float(compute_mse(y_true_sim, y_sim))
        sim_rmse = float(np.sqrt(sim_mse)) if (np.isfinite(sim_mse) and sim_mse < 1e12) else 1e6
        sim_fit_pct = _compute_channelwise_nrmse_fit(y_true_sim, y_sim)

        report["Sim_MSE"] = sim_mse
        report["Sim_RMSE"] = sim_rmse
        report["Sim_Fit_Pct"] = max(-1000.0, min(100.0, sim_fit_pct))

    return report


def compute_deep_single_model_diagnostics(u_aligned, actual_targets, predicted_outputs, model_type="MLP", theta_weights=None, na=0, nb=0, nc=0):
    # Additional diagnostics: FIT%, VAF, error percentiles, regime analysis, and stability
    na = max(0, int(na or 0))
    nb = max(0, int(nb or 0))
    nc = max(0, int(nc or 0))
    y_raw, y_hat_raw = _align_signal_shapes(actual_targets, predicted_outputs)

    empty_diag = {
        "NRMSE_Fit_Pct": 0.0, "VAF_Pct": 0.0, "Theil_U1": 1.0, "Theil_U2": 1.0,
        "Naive_RMSE": 0.0, "Skill_Over_Naive_Pct": 0.0, "Pearson_R": 0.0, "SNR_dB": 0.0,
        "MAE": 0.0, "MedAE": 0.0, "P90_AE": 0.0, "P95_AE": 0.0, "P99_AE": 0.0, "Max_AE": 0.0,
        "Mean_Bias": 0.0, "Std_Err": 0.0, "Skewness": 0.0, "Excess_Kurtosis": 0.0,
        "CCF_Pass_Pct": 100.0, "ACF_Pass_Pct": 100.0, "Regime_Table": [],
        "Max_AR_Pole_Mag": None, "Is_AR_Stable": True, "Target_Std": 0.0
    }
    if len(y_raw) == 0 or len(y_hat_raw) == 0:
        return empty_diag

    u = np.asarray(u_aligned, dtype=float)
    y_2d = y_raw.reshape(-1, 1) if y_raw.ndim == 1 else y_raw
    yhat_2d = y_hat_raw.reshape(-1, 1) if y_hat_raw.ndim == 1 else y_hat_raw

    # Clamp to common length if sample lengths differ
    min_len = min(len(y_2d), len(yhat_2d))
    if min_len == 0 or y_2d.shape[1] == 0 or yhat_2d.shape[1] == 0:
        return empty_diag

    y_2d = y_2d[:min_len]
    yhat_2d = yhat_2d[:min_len]

    n_u = max(1, 1 if u.ndim == 1 else u.shape[1])
    n_y = min(y_2d.shape[1], yhat_2d.shape[1])

    residuals_2d = y_2d[:, :n_y] - yhat_2d[:, :n_y]
    abs_err = np.abs(residuals_2d).flatten()

    # 1. Standard identification metrics (computed per output channel so SIMO/MIMO channels don't cancel out)
    nrmse_fit_pct = _compute_channelwise_nrmse_fit(y_2d, yhat_2d)

    vaf_channels, pearson_channels, snr_channels, u1_channels = [], [], [], []
    bias_channels, std_channels, skew_channels, kurt_channels = [], [], [], []

    with np.errstate(over="ignore", invalid="ignore"):
        for ch in range(n_y):
            yc = y_2d[:, ch]
            yhc = yhat_2d[:, ch]
            ec = yc - yhc
            var_yc = float(np.var(yc))
            var_ec = float(np.var(ec))
            vaf_c = max(-1000.0, (1.0 - (var_ec / var_yc)) * 100.0) if var_yc > 1e-12 else 0.0
            vaf_channels.append(vaf_c)

            if np.std(yc) > 1e-12 and np.std(yhc) > 1e-12:
                try:
                    r_c = float(np.corrcoef(yc, yhc)[0, 1])
                    pearson_channels.append(r_c if np.isfinite(r_c) else 0.0)
                except Exception:
                    pearson_channels.append(0.0)
            else:
                pearson_channels.append(0.0)

            sig_p = float(np.mean(yc ** 2))
            noi_p = float(np.mean(ec ** 2))
            snr_channels.append(float(10.0 * np.log10(max(sig_p, 1e-12) / max(noi_p, 1e-12))))

            denom_c = float(np.sqrt(sig_p) + np.sqrt(np.mean(yhc ** 2)))
            u1_channels.append(float(np.sqrt(noi_p) / denom_c) if denom_c > 1e-12 else 0.0)

            b_c = float(np.mean(ec))
            s_c = float(np.std(ec))
            bias_channels.append(b_c)
            std_channels.append(s_c)
            if s_c > 1e-12:
                z_c = (ec - b_c) / s_c
                sk_c = float(np.mean(z_c ** 3))
                ku_c = float(np.mean(z_c ** 4) - 3.0)
                skew_channels.append(sk_c if np.isfinite(sk_c) else 0.0)
                kurt_channels.append(ku_c if np.isfinite(ku_c) else 0.0)
            else:
                skew_channels.append(0.0)
                kurt_channels.append(0.0)

    vaf_pct = float(np.mean(vaf_channels)) if vaf_channels else 0.0
    pearson_r = float(np.mean(pearson_channels)) if pearson_channels else 0.0
    snr_db = float(np.mean(snr_channels)) if snr_channels else 0.0
    theil_u1 = float(np.mean(u1_channels)) if u1_channels else 0.0
    mean_bias = float(np.mean(bias_channels)) if bias_channels else 0.0
    std_err = float(np.mean(std_channels)) if std_channels else 0.0
    skewness = float(np.mean(skew_channels)) if skew_channels else 0.0
    kurtosis = float(np.mean(kurt_channels)) if kurt_channels else 0.0

    # Theil's U2: compares model RMSE against a naive persistence baseline y_hat(t) = y(t-1)
    if len(y_2d) > 1:
        naive_err = y_2d[1:, :n_y] - y_2d[:-1, :n_y]
        naive_rmse = float(np.sqrt(np.mean(naive_err ** 2)))
        model_rmse_tail = float(np.sqrt(np.mean((y_2d[1:, :n_y] - yhat_2d[1:, :n_y]) ** 2)))
        theil_u2 = (model_rmse_tail / naive_rmse) if naive_rmse > 1e-12 else np.inf
        skill_over_naive_pct = max(-1000.0, min(100.0, (1.0 - theil_u2) * 100.0)) if np.isfinite(theil_u2) else -100.0
    else:
        naive_rmse = 0.0
        theil_u2 = 1.0
        skill_over_naive_pct = 0.0

    # 2. Error Quantiles
    valid_ae = abs_err[np.isfinite(abs_err)]
    if len(valid_ae) > 0:
        mae = float(np.mean(valid_ae))
        med_ae = float(np.median(valid_ae))
        p90_ae = float(np.percentile(valid_ae, 90))
        p95_ae = float(np.percentile(valid_ae, 95))
        p99_ae = float(np.percentile(valid_ae, 99))
        max_ae = float(np.max(valid_ae))
    else:
        mae, med_ae, p90_ae, p95_ae, p99_ae, max_ae = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # 3. Residual cross-correlation and autocorrelation pass rates
    ccf_pass_pct = compute_cross_correlation_pass_rate(u, y_raw, y_hat_raw)
    acf_pass_pct = compute_white_noise_pass_rate(y_raw, y_hat_raw)

    # 4. Error performance across operating ranges (quartiles of y)
    y_prim = y_2d[:, 0]
    err_prim = y_prim - yhat_2d[:, 0]
    valid_y_prim = y_prim[np.isfinite(y_prim)]
    q25, q50, q75 = np.percentile(valid_y_prim, [25, 50, 75]) if len(valid_y_prim) > 0 else (0.0, 0.0, 0.0)
    regime_defs = [
        ("Q1: Low Load (0-25%)", y_prim <= q25),
        ("Q2: Med-Low (25-50%)", (y_prim > q25) & (y_prim <= q50)),
        ("Q3: Med-High (50-75%)", (y_prim > q50) & (y_prim <= q75)),
        ("Q4: Peak Load (75-100%)", y_prim > q75)
    ]
    regime_table = []
    for r_name, mask in regime_defs:
        if np.any(mask):
            r_y = y_prim[mask]
            r_err = err_prim[mask]
            regime_table.append({
                "name": r_name,
                "count": int(np.sum(mask)),
                "y_range": f"[{np.min(r_y):.2f}, {np.max(r_y):.2f}]",
                "rmse": float(np.sqrt(np.mean(r_err ** 2))),
                "mae": float(np.mean(np.abs(r_err))),
                "bias": float(np.mean(r_err))
            })

    # 5. Check if AR polynomial roots are strictly inside the unit circle (stability)
    max_pole_mag = None
    is_ar_stable = True
    if model_type == "ARMAX" and theta_weights is not None and na > 0:
        theta_flat = np.asarray(theta_weights, dtype=float).flatten()
        stride = int(na) + int(nb) * n_u + int(nc)
        pole_mags = []
        for out_ch in range(n_y):
            if stride <= 0 or len(theta_flat) < (out_ch + 1) * stride:
                break
            start_idx = out_ch * stride
            ar_slice = theta_flat[start_idx: start_idx + na]
            if len(ar_slice) == na:
                ar_coeffs = np.concatenate([[1.0], ar_slice])
                try:
                    poles = np.roots(ar_coeffs)
                    valid_poles = poles[np.isfinite(poles)]
                    if len(valid_poles) > 0:
                        pole_mags.append(float(np.max(np.abs(valid_poles))))
                except Exception:
                    pass
        if pole_mags:
            max_pole_mag = float(np.max(pole_mags))
            is_ar_stable = bool(max_pole_mag < 1.0)

    return {
        "NRMSE_Fit_Pct": nrmse_fit_pct,
        "VAF_Pct": vaf_pct,
        "Theil_U1": float(theil_u1) if np.isfinite(theil_u1) else 1.0,
        "Theil_U2": float(theil_u2) if np.isfinite(theil_u2) else 1.0,
        "Naive_RMSE": float(naive_rmse) if np.isfinite(naive_rmse) else 0.0,
        "Skill_Over_Naive_Pct": float(skill_over_naive_pct) if np.isfinite(skill_over_naive_pct) else 0.0,
        "Pearson_R": float(pearson_r) if np.isfinite(pearson_r) else 0.0,
        "SNR_dB": float(snr_db) if np.isfinite(snr_db) else 0.0,
        "MAE": mae,
        "MedAE": med_ae,
        "P90_AE": p90_ae,
        "P95_AE": p95_ae,
        "P99_AE": p99_ae,
        "Max_AE": max_ae,
        "Mean_Bias": float(mean_bias) if np.isfinite(mean_bias) else 0.0,
        "Std_Err": float(std_err) if np.isfinite(std_err) else 0.0,
        "Skewness": skewness,
        "Excess_Kurtosis": kurtosis,
        "CCF_Pass_Pct": ccf_pass_pct,
        "ACF_Pass_Pct": acf_pass_pct,
        "Regime_Table": regime_table,
        "Max_AR_Pole_Mag": max_pole_mag,
        "Is_AR_Stable": is_ar_stable,
        "Target_Std": float(np.std(y_prim)) if len(y_prim) > 0 else 0.0
    }
