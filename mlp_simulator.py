"""
mlp_simulator.py

Multi-Layer Perceptron (MLP) NARX simulator for SISO, MISO, SIMO, and MIMO systems.
Calculates 1-step ahead predictions and recursive free-run simulation.
"""

import numpy as np


def get_mlp_parameter_count(na, nb, hidden_units, n_u=1, n_y=1):
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]

    n_in = (int(na) * int(n_y)) + (int(nb) * int(n_u))
    n_out = int(n_y)

    total_params = (n_in * hidden_units[0]) + hidden_units[0]
    for i in range(len(hidden_units) - 1):
        total_params += (hidden_units[i] * hidden_units[i + 1]) + hidden_units[i + 1]

    total_params += (hidden_units[-1] * n_out) + n_out
    return int(total_params)


def build_mlp_feature_matrix(u_signal, y_signal, na, nb, nk):
    """
    Builds the lagged regressor matrix for MLP inputs.
    Precomputing this once per fold saves memory allocations during optimization.
    """
    u_raw = np.asarray(u_signal, dtype=float)
    y_raw = np.asarray(y_signal, dtype=float)
    y_was_1d = (y_raw.ndim == 1)

    u_mat = u_raw.reshape(-1, 1) if u_raw.ndim == 1 else u_raw
    y_mat = y_raw.reshape(-1, 1) if y_raw.ndim == 1 else y_raw

    total_samples = y_mat.shape[0]
    n_u = u_mat.shape[1]
    n_y = y_mat.shape[1]

    max_delay_lag = max(1, na, nb + nk - 1)
    valid_len = total_samples - max_delay_lag
    n_in = (na * n_y) + (nb * n_u)
    features = np.zeros((valid_len, n_in), dtype=float)

    col_idx = 0
    for out_ch in range(n_y):
        y_ch = y_mat[:, out_ch]
        for i in range(na):
            features[:, col_idx] = y_ch[max_delay_lag - 1 - i: total_samples - 1 - i]
            col_idx += 1

    for in_ch in range(n_u):
        u_ch = u_mat[:, in_ch]
        for j in range(nb):
            lag = nk + j
            features[:, col_idx] = u_ch[max_delay_lag - lag: total_samples - lag]
            col_idx += 1

    return features, max_delay_lag, total_samples, n_u, n_y, y_was_1d


def simulate_mlp_system(u_signal, y_signal, theta_weights, na, nb, nk, hidden_units=None, precomputed_features=None):
    if hidden_units is None:
        hidden_units = [5]
    elif isinstance(hidden_units, int):
        hidden_units = [hidden_units]

    if precomputed_features is not None:
        features, max_delay_lag, total_samples, n_u, n_y, y_was_1d = precomputed_features
    else:
        features, max_delay_lag, total_samples, n_u, n_y, y_was_1d = build_mlp_feature_matrix(
            u_signal, y_signal, na, nb, nk
        )

    n_in = (na * n_y) + (nb * n_u)
    theta = np.asarray(theta_weights, dtype=float).flatten()
    expected_params = get_mlp_parameter_count(na, nb, hidden_units, n_u=n_u, n_y=n_y)

    if len(theta) < expected_params:
        padded = np.zeros(expected_params, dtype=float)
        padded[:len(theta)] = theta
        theta = padded

    weights, biases = [], []
    idx = 0
    layer_in = n_in

    for units in hidden_units:
        w = theta[idx: idx + layer_in * units].reshape(layer_in, units)
        idx += layer_in * units
        b = theta[idx: idx + units]
        idx += units
        weights.append(w)
        biases.append(b)
        layer_in = units

    w_out = theta[idx: idx + layer_in * n_y].reshape(layer_in, n_y)
    idx += layer_in * n_y
    b_out = theta[idx: idx + n_y]

    predicted_output = np.zeros((total_samples, n_y), dtype=float)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        activations = features
        for w, b in zip(weights, biases):
            activations = np.tanh(np.dot(activations, w) + b)

        out = np.dot(activations, w_out) + b_out
        # In case bad weights saturate or produce Inf/NaN, assign Inf penalty so the EA discards this genome
        if not np.isfinite(out).all() or np.max(np.abs(out)) > 1e10:
            predicted_output[:] = np.inf
        else:
            predicted_output[max_delay_lag:, :] = out

    if y_was_1d:
        return predicted_output.flatten(), max_delay_lag
    return predicted_output, max_delay_lag


def simulate_mlp_free_run(u_signal, y_signal, theta_weights, na, nb, nk, hidden_units=None):
    """
    Simulates the MLP model in free-run mode.
    Past predictions are fed back recursively into the lagged inputs instead of true measurements.
    """
    if hidden_units is None:
        hidden_units = [5]
    elif isinstance(hidden_units, int):
        hidden_units = [hidden_units]

    u_raw = np.asarray(u_signal, dtype=float)
    y_raw = np.asarray(y_signal, dtype=float)
    y_was_1d = (y_raw.ndim == 1)

    u_mat = u_raw.reshape(-1, 1) if u_raw.ndim == 1 else u_raw
    y_mat = y_raw.reshape(-1, 1) if y_raw.ndim == 1 else y_raw

    total_samples = y_mat.shape[0]
    n_u = u_mat.shape[1]
    n_y = y_mat.shape[1]
    n_in = (na * n_y) + (nb * n_u)

    max_delay_lag = max(1, na, nb + nk - 1)
    simulated_output = np.zeros((total_samples, n_y), dtype=float)
    simulated_output[:max_delay_lag, :] = y_mat[:max_delay_lag, :]

    theta = np.asarray(theta_weights, dtype=float).flatten()
    expected_params = get_mlp_parameter_count(na, nb, hidden_units, n_u=n_u, n_y=n_y)
    if len(theta) < expected_params:
        padded = np.zeros(expected_params, dtype=float)
        padded[:len(theta)] = theta
        theta = padded

    weights, biases = [], []
    idx = 0
    layer_in = n_in
    for units in hidden_units:
        w = theta[idx: idx + layer_in * units].reshape(layer_in, units)
        idx += layer_in * units
        b = theta[idx: idx + units]
        idx += units
        weights.append(w)
        biases.append(b)
        layer_in = units

    w_out = theta[idx: idx + layer_in * n_y].reshape(layer_in, n_y)
    b_out = theta[idx + layer_in * n_y: idx + layer_in * n_y + n_y]

    regressor = np.zeros((1, n_in), dtype=float)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for t in range(max_delay_lag, total_samples):
            col_idx = 0
            # Construct regressor from previously simulated y_hat instead of ground truth
            for out_ch in range(n_y):
                for i in range(na):
                    regressor[0, col_idx] = simulated_output[t - 1 - i, out_ch]
                    col_idx += 1

            for in_ch in range(n_u):
                for j in range(nb):
                    lag = nk + j
                    regressor[0, col_idx] = u_mat[t - lag, in_ch]
                    col_idx += 1

            activations = regressor
            for w, b in zip(weights, biases):
                activations = np.tanh(np.dot(activations, w) + b)
            y_hat_step = np.dot(activations, w_out) + b_out

            if not np.isfinite(y_hat_step).all() or np.max(np.abs(y_hat_step)) > 1e10:
                simulated_output[:] = np.inf
                break
            simulated_output[t, :] = y_hat_step[0]

    if y_was_1d:
        return simulated_output.flatten(), max_delay_lag
    return simulated_output, max_delay_lag
