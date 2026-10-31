"""
armax_simulator.py

Simulates discrete-time ARMAX models: A(q)y(t) = B(q)u(t-nk) + C(q)e(t).
Supports SISO, MISO, SIMO, and MIMO systems.
Provides 1-step ahead prediction and free-run recursive simulation.
"""

import numpy as np
from scipy import signal as sp_signal
import warnings


def get_armax_parameter_count(na, nb, nc, n_u=1, n_y=1):
    # Each output channel has na AR weights + (nb * n_u) input weights + nc MA weights
    return int(n_y) * (int(na) + int(nb) * int(n_u) + int(nc))


def build_armax_feature_cache(u_signal, y_signal, na, nb, nc, nk):
    """
    Builds the lag matrices for inputs and outputs once per fold.
    This saves a lot of time since we don't have to re-slice arrays inside the EA evaluation loop.
    """
    u_raw = np.asarray(u_signal, dtype=float)
    y_raw = np.asarray(y_signal, dtype=float)
    y_was_1d = (y_raw.ndim == 1)

    u_mat = u_raw.reshape(-1, 1) if u_raw.ndim == 1 else u_raw
    y_mat = y_raw.reshape(-1, 1) if y_raw.ndim == 1 else y_raw

    total_samples = y_mat.shape[0]
    n_u = u_mat.shape[1]
    n_y = y_mat.shape[1]

    # Need enough past samples to fill all AR, B input lag, and MA error buffers
    max_delay_lag = max(1, na, nb + nk - 1, nc)
    valid_len = total_samples - max_delay_lag

    u_lags = np.zeros((valid_len, nb * n_u), dtype=float)
    b_idx = 0
    for in_ch in range(n_u):
        u_ch = u_mat[:, in_ch]
        for j in range(nb):
            lag = nk + j
            u_lags[:, b_idx] = u_ch[max_delay_lag - lag: total_samples - lag]
            b_idx += 1

    y_lags_list = []
    y_spreads = []
    for out_ch in range(n_y):
        y_ch = y_mat[:, out_ch]
        y_lags = np.zeros((valid_len, na), dtype=float)
        for i in range(na):
            # Store negative past outputs so dot product directly adds -a_i * y(t-i)
            y_lags[:, i] = -y_ch[max_delay_lag - 1 - i: total_samples - 1 - i]
        y_lags_list.append(y_lags)
        # Safety spread threshold to detect if residual filter explodes on unstable candidate polynomials
        y_spreads.append(max(float(np.std(y_ch)) * 5.0, 10.0))

    return u_lags, y_lags_list, y_spreads, y_mat, max_delay_lag, total_samples, n_u, n_y, y_was_1d


def simulate_armax_system(u_signal, y_signal, theta_weights, na, nb, nc, nk, precomputed_cache=None):
    if precomputed_cache is not None:
        u_lags, y_lags_list, y_spreads, y_mat, max_delay_lag, total_samples, n_u, n_y, y_was_1d = precomputed_cache
    else:
        u_lags, y_lags_list, y_spreads, y_mat, max_delay_lag, total_samples, n_u, n_y, y_was_1d = build_armax_feature_cache(
            u_signal, y_signal, na, nb, nc, nk
        )

    predicted_output = np.zeros((total_samples, n_y), dtype=float)
    theta = np.asarray(theta_weights, dtype=float).flatten()
    params_per_output = na + (nb * n_u) + nc

    if len(theta) < params_per_output * n_y:
        padded = np.zeros(params_per_output * n_y, dtype=float)
        padded[:len(theta)] = theta
        theta = padded

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):

        for out_ch in range(n_y):
            offset = out_ch * params_per_output
            theta_ar = theta[offset: offset + na]
            theta_b = theta[offset + na: offset + na + (nb * n_u)]
            theta_ma = theta[offset + na + (nb * n_u): offset + params_per_output]

            arx_valid = np.zeros(total_samples - max_delay_lag, dtype=float)
            if na > 0:
                arx_valid += np.dot(y_lags_list[out_ch], theta_ar)
            if nb > 0:
                arx_valid += np.dot(u_lags, theta_b)

            if nc == 0:
                if not np.isfinite(arx_valid).all() or np.max(np.abs(arx_valid)) > 1e10:
                    predicted_output[:] = np.inf
                    return (predicted_output.flatten() if y_was_1d else predicted_output), max_delay_lag
                predicted_output[max_delay_lag:, out_ch] = arx_valid
                continue

            y_valid = y_mat[max_delay_lag:, out_ch]
            y_spread = y_spreads[out_ch]

            # Filter residuals through C(q) using scipy's lfilter for speed
            c_poly = np.concatenate([[1.0], theta_ma])
            e_unclipped = sp_signal.lfilter([1.0], c_poly, y_valid - arx_valid)

            if np.isfinite(e_unclipped).all() and np.max(np.abs(e_unclipped)) <= y_spread:
                y_hat_valid = y_valid - e_unclipped
                if not np.isfinite(y_hat_valid).all() or np.max(np.abs(y_hat_valid)) > 1e10:
                    predicted_output[:] = np.inf
                    return (predicted_output.flatten() if y_was_1d else predicted_output), max_delay_lag
                predicted_output[max_delay_lag:, out_ch] = y_hat_valid
            else:
                # If filter blows up or values become too large, fall back to step-by-step loop with clipping
                res_ch = np.zeros(total_samples, dtype=float)
                y_ch = y_mat[:, out_ch]
                for idx_t, t in enumerate(range(max_delay_lag, total_samples)):
                    ma_part = 0.0
                    for k in range(nc):
                        ma_part += theta_ma[k] * res_ch[t - 1 - k]
                    y_hat = arx_valid[idx_t] + ma_part
                    if not np.isfinite(y_hat) or abs(y_hat) > 1e10:
                        predicted_output[:] = np.inf
                        return (predicted_output.flatten() if y_was_1d else predicted_output), max_delay_lag
                    predicted_output[t, out_ch] = y_hat
                    err = y_ch[t] - y_hat
                    res_ch[t] = np.clip(err, -y_spread, y_spread)

    if y_was_1d:
        return predicted_output.flatten(), max_delay_lag
    return predicted_output, max_delay_lag


def simulate_armax_free_run(u_signal, y_signal, theta_weights, na, nb, nc, nk):
    """
    Simulates the model in free-run mode without feeding in true past outputs.
    Past model outputs are fed back recursively into A(q).
    """
    u_raw = np.asarray(u_signal, dtype=float)
    y_raw = np.asarray(y_signal, dtype=float)
    y_was_1d = (y_raw.ndim == 1)

    u_mat = u_raw.reshape(-1, 1) if u_raw.ndim == 1 else u_raw
    y_mat = y_raw.reshape(-1, 1) if y_raw.ndim == 1 else y_raw

    total_samples = y_mat.shape[0]
    n_u = u_mat.shape[1]
    n_y = y_mat.shape[1]

    max_delay_lag = max(1, na, nb + nk - 1, nc)
    simulated_output = np.zeros((total_samples, n_y), dtype=float)
    simulated_output[:max_delay_lag, :] = y_mat[:max_delay_lag, :]

    theta = np.asarray(theta_weights, dtype=float).flatten()
    params_per_output = na + (nb * n_u) + nc

    if len(theta) < params_per_output * n_y:
        padded = np.zeros(params_per_output * n_y, dtype=float)
        padded[:len(theta)] = theta
        theta = padded

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for out_ch in range(n_y):
            offset = out_ch * params_per_output
            theta_ar = theta[offset: offset + na]
            theta_b = theta[offset + na: offset + na + (nb * n_u)]

            u_part = np.zeros(total_samples - max_delay_lag, dtype=float)
            b_idx = 0
            for in_ch in range(n_u):
                u_ch = u_mat[:, in_ch]
                for j in range(nb):
                    lag = nk + j
                    u_part += theta_b[b_idx] * u_ch[max_delay_lag - lag: total_samples - lag]
                    b_idx += 1

            if na > 0:
                a_poly = np.concatenate([[1.0], theta_ar])
                stop_idx = max_delay_lag - 1 - na
                # Initialize IIR filter delays using the real output history before running autonomously
                y_init = y_mat[max_delay_lag - 1: stop_idx if stop_idx >= 0 else None: -1, out_ch]
                zi = sp_signal.lfiltic([1.0], a_poly, y_init)
                y_filt, _ = sp_signal.lfilter([1.0], a_poly, u_part, zi=zi)
            else:
                y_filt = u_part

            if not np.isfinite(y_filt).all() or np.max(np.abs(y_filt)) > 1e10:
                simulated_output[:] = np.inf
                return (simulated_output.flatten() if y_was_1d else simulated_output), max_delay_lag

            simulated_output[max_delay_lag:, out_ch] = y_filt

    if y_was_1d:
        return simulated_output.flatten(), max_delay_lag
    return simulated_output, max_delay_lag
