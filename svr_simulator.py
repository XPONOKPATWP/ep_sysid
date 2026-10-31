"""
svr_simulator.py

Support Vector Regression (SVR) NARX simulator for SISO, MISO, SIMO, and MIMO systems.
Optimizes RBF kernel hyperparameters [C, epsilon, gamma].
"""

import numpy as np
from sklearn.svm import SVR
import warnings


def get_svr_parameter_count(na=2, nb=2, n_u=1, n_y=1):
    # SVR optimizes the 3 RBF kernel hyperparameters: C, epsilon, and gamma
    return 3


def _build_svr_lag_matrix(u_in, y_in, na, nb, nk):
    na = max(0, int(na or 0))
    nb = max(0, int(nb or 0))
    nk = max(0, int(nk or 0))
    if na <= 0 and nb <= 0:
        na = 1
    u_m = u_in.reshape(-1, 1) if u_in.ndim == 1 else u_in
    y_m = y_in.reshape(-1, 1) if y_in.ndim == 1 else y_in

    n_samples = y_m.shape[0]
    max_lag = max(1, na, (nb + nk - 1) if nb > 0 else 0)
    valid_len = max(0, n_samples - max_lag)
    n_features = (na * y_m.shape[1]) + (nb * u_m.shape[1])
    x_mat = np.zeros((valid_len, n_features), dtype=float)

    if valid_len > 0:
        col_idx = 0
        for out_ch in range(y_m.shape[1]):
            y_ch = y_m[:, out_ch]
            for i in range(na):
                x_mat[:, col_idx] = y_ch[max_lag - 1 - i: n_samples - 1 - i]
                col_idx += 1

        for in_ch in range(u_m.shape[1]):
            u_ch = u_m[:, in_ch]
            for j in range(nb):
                lag = nk + j
                x_mat[:, col_idx] = u_ch[max_lag - lag: n_samples - lag]
                col_idx += 1

    return x_mat, y_m[max_lag:, :], max_lag, n_samples


def build_svr_feature_cache(u_signal, y_signal, na, nb, nk):
    """
    Builds the lag matrix and subsamples training points once per fold
    so SVR evaluations run faster during optimization.
    """
    na = max(0, int(na or 0))
    nb = max(0, int(nb or 0))
    nk = max(0, int(nk or 0))
    if na <= 0 and nb <= 0:
        na = 1
    u_raw = np.asarray(u_signal, dtype=float)
    y_raw = np.asarray(y_signal, dtype=float)
    y_was_1d = (y_raw.ndim == 1)

    u_mat = u_raw.reshape(-1, 1) if u_raw.ndim == 1 else u_raw
    y_mat = y_raw.reshape(-1, 1) if y_raw.ndim == 1 else y_raw

    n_u = u_mat.shape[1]
    n_y = y_mat.shape[1]

    x_train, y_train_target, max_delay_lag, total_samples = _build_svr_lag_matrix(u_mat, y_mat, na, nb, nk)

    # Keep fitting subset to around 800-1200 points so training time doesn't blow up (O(N^3)).
    # We hold out 25% of the data to evaluate generalization error during EA optimization.
    max_fit_samples = 800 if n_u > 1 else 1200
    if len(x_train) > 40:
        mask = np.ones(len(x_train), dtype=bool)
        mask[::4] = False
        x_pool = x_train[mask]
        y_pool = y_train_target[mask]
        val_indices = np.where(~mask)[0]
        x_val = x_train[val_indices]
    else:
        x_pool = x_train
        y_pool = y_train_target
        val_indices = np.arange(len(x_train))
        x_val = x_train

    if len(x_pool) > max_fit_samples:
        uniform_idx = np.linspace(0, len(x_pool) - 1, max_fit_samples - 40, dtype=int)
        sorted_target_idx = np.argsort(y_pool[:, 0])
        extrema_idx = np.concatenate([sorted_target_idx[:20], sorted_target_idx[-20:]])
        idx = np.unique(np.concatenate([uniform_idx, extrema_idx]))
        x_train_fit = x_pool[idx]
        y_train_fit = y_pool[idx]
    else:
        x_train_fit = x_pool
        y_train_fit = y_pool

    return x_train_fit, y_train_fit, x_val, val_indices, y_train_target, max_delay_lag, total_samples, n_u, n_y, y_was_1d


def simulate_svr_system(u_signal, y_signal, theta_weights, na, nb, nk, u_train=None, y_train=None, cross_domain=False, precomputed_cache=None):
    # Fits SVR model with RBF kernel and evaluates predictions
    na = max(0, int(na or 0))
    nb = max(0, int(nb or 0))
    nk = max(0, int(nk or 0))
    if na <= 0 and nb <= 0:
        na = 1

    is_cached_training_eval = (precomputed_cache is not None and u_train is None and not cross_domain)
    val_indices = None
    y_train_target_ref = None

    if is_cached_training_eval:
        x_train_fit, y_train_fit, x_test, val_indices, y_train_target_ref, max_delay_lag, total_samples, n_u, n_y, y_was_1d = precomputed_cache
    else:
        u_raw = np.asarray(u_signal, dtype=float)
        y_raw = np.asarray(y_signal, dtype=float)
        y_was_1d = (y_raw.ndim == 1)

        u_mat = u_raw.reshape(-1, 1) if u_raw.ndim == 1 else u_raw
        y_mat = y_raw.reshape(-1, 1) if y_raw.ndim == 1 else y_raw

        n_u = u_mat.shape[1]
        n_y = y_mat.shape[1]

        is_training_eval = (u_train is None and not cross_domain)
        has_valid_train = (u_train is not None and y_train is not None)
        if has_valid_train:
            fit_u = np.asarray(u_train, dtype=float)
            fit_y = np.asarray(y_train, dtype=float)
            fit_u_cols = 1 if fit_u.ndim == 1 else fit_u.shape[1]
            fit_y_cols = 1 if fit_y.ndim == 1 else fit_y.shape[1]
            if fit_u_cols != n_u or fit_y_cols != n_y:
                fit_u, fit_y = u_mat, y_mat
        else:
            fit_u, fit_y = u_mat, y_mat

        x_train, y_train_target, _, _ = _build_svr_lag_matrix(fit_u, fit_y, na, nb, nk)
        x_test_full, _, max_delay_lag, total_samples = _build_svr_lag_matrix(u_mat, y_mat, na, nb, nk)

        # Use the exact same 75% pool and max_fit_samples subsampling as build_svr_feature_cache
        # so the support vectors fitted at test/inference time match the ones optimized during training
        max_fit_samples = 800 if n_u > 1 else 1200
        if len(x_train) > 40:
            mask = np.ones(len(x_train), dtype=bool)
            mask[::4] = False
            x_pool = x_train[mask]
            y_pool = y_train_target[mask]
            if is_training_eval:
                val_indices = np.where(~mask)[0]
                x_test = x_train[val_indices]
                y_train_target_ref = y_train_target
            else:
                x_test = x_test_full
        else:
            x_pool = x_train
            y_pool = y_train_target
            x_test = x_test_full

        if len(x_pool) > max_fit_samples:
            uniform_idx = np.linspace(0, len(x_pool) - 1, max_fit_samples - 40, dtype=int)
            sorted_target_idx = np.argsort(y_pool[:, 0])
            extrema_idx = np.concatenate([sorted_target_idx[:20], sorted_target_idx[-20:]])
            idx = np.unique(np.concatenate([uniform_idx, extrema_idx]))
            x_train_fit = x_pool[idx]
            y_train_fit = y_pool[idx]
        else:
            x_train_fit = x_pool
            y_train_fit = y_pool

    total_lag_features = max(1, (na * n_y) + (nb * n_u))
    theta = list(np.nan_to_num(np.asarray(theta_weights, dtype=float).flatten(), nan=0.1, posinf=1.0, neginf=-1.0))
    while len(theta) < 3:
        theta.append(0.1)

    c_val = min(max(1e-3, abs(float(theta[0]))), 50.0)
    eps_val = min(max(1e-4, abs(float(theta[1]))), 2.0)
    max_gamma = 5.0 if n_u == 1 else (4.0 / float(total_lag_features))
    gamma_val = min(max(1e-4, abs(float(theta[2]))), max_gamma)

    if cross_domain:
        gamma_val = min(gamma_val, 1.0 / float(total_lag_features))

    predicted_output = np.zeros((total_samples, n_y), dtype=float)

    if total_samples <= max_delay_lag or len(x_train_fit) == 0 or len(x_test) == 0:
        predicted_output[:] = np.inf
        return (predicted_output.flatten() if y_was_1d else predicted_output), max_delay_lag

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for out_ch in range(n_y):
            try:
                svr = SVR(
                    kernel="rbf",
                    C=c_val,
                    epsilon=eps_val,
                    gamma=gamma_val,
                    tol=1e-2,
                    cache_size=500,
                    max_iter=2000
                )
                svr.fit(x_train_fit, y_train_fit[:, out_ch])
                y_pred_sub = svr.predict(x_test)
                if val_indices is not None and y_train_target_ref is not None:
                    # Non-holdout rows get exact target, and holdout residuals are scaled
                    # so standard MSE calculation reflects 100% holdout validation error
                    y_full_ch = y_train_target_ref[:, out_ch].copy()
                    scale = np.sqrt(float(len(y_full_ch)) / max(1.0, float(len(val_indices))))
                    y_val_true = y_full_ch[val_indices]
                    y_full_ch[val_indices] = y_val_true - scale * (y_val_true - y_pred_sub)
                    predicted_output[max_delay_lag:, out_ch] = y_full_ch
                else:
                    predicted_output[max_delay_lag:, out_ch] = y_pred_sub
            except Exception:
                predicted_output[:] = np.inf
                break

    if y_was_1d:
        return predicted_output.flatten(), max_delay_lag
    return predicted_output, max_delay_lag


def simulate_svr_free_run(u_signal, y_signal, theta_weights, na, nb, nk, u_train=None, y_train=None, cross_domain=False):
    """
    Simulates the SVR model recursively in free-run mode.
    Past predictions are fed back into the autoregressive lags.
    """
    na = max(0, int(na or 0))
    nb = max(0, int(nb or 0))
    nk = max(0, int(nk or 0))
    if na <= 0 and nb <= 0:
        na = 1
    u_raw = np.asarray(u_signal, dtype=float)
    y_raw = np.asarray(y_signal, dtype=float)
    y_was_1d = (y_raw.ndim == 1)

    u_mat = u_raw.reshape(-1, 1) if u_raw.ndim == 1 else u_raw
    y_mat = y_raw.reshape(-1, 1) if y_raw.ndim == 1 else y_raw
    n_u, n_y = u_mat.shape[1], y_mat.shape[1]
    total_samples = y_mat.shape[0]
    max_delay_lag = max(1, na, (nb + nk - 1) if nb > 0 else 0)

    has_valid_train = (u_train is not None and y_train is not None)
    if has_valid_train:
        fit_u = np.asarray(u_train, dtype=float)
        fit_y = np.asarray(y_train, dtype=float)
        if (1 if fit_u.ndim == 1 else fit_u.shape[1]) != n_u or (1 if fit_y.ndim == 1 else fit_y.shape[1]) != n_y:
            fit_u, fit_y = u_mat, y_mat
    else:
        fit_u, fit_y = u_mat, y_mat

    x_train, y_train_target, _, _ = _build_svr_lag_matrix(fit_u, fit_y, na, nb, nk)

    # Match the exact 75% pool and max_fit_samples subsampling from build_svr_feature_cache
    max_fit_samples = 800 if n_u > 1 else 1200
    if len(x_train) > 40:
        mask = np.ones(len(x_train), dtype=bool)
        mask[::4] = False
        x_pool = x_train[mask]
        y_pool = y_train_target[mask]
    else:
        x_pool = x_train
        y_pool = y_train_target

    if len(x_pool) > max_fit_samples:
        uniform_idx = np.linspace(0, len(x_pool) - 1, max_fit_samples - 40, dtype=int)
        sorted_target_idx = np.argsort(y_pool[:, 0])
        extrema_idx = np.concatenate([sorted_target_idx[:20], sorted_target_idx[-20:]])
        idx = np.unique(np.concatenate([uniform_idx, extrema_idx]))
        x_train_fit = x_pool[idx]
        y_train_fit = y_pool[idx]
    else:
        x_train_fit = x_pool
        y_train_fit = y_pool

    total_lag_features = max(1, (na * n_y) + (nb * n_u))
    theta = list(np.nan_to_num(np.asarray(theta_weights, dtype=float).flatten(), nan=0.1, posinf=1.0, neginf=-1.0)) + [0.1, 0.1, 0.1]
    c_val = min(max(1e-3, abs(float(theta[0]))), 50.0)
    eps_val = min(max(1e-4, abs(float(theta[1]))), 2.0)
    max_gamma = 5.0 if n_u == 1 else (4.0 / float(total_lag_features))
    gamma_val = min(max(1e-4, abs(float(theta[2]))), max_gamma)
    if cross_domain:
        gamma_val = min(gamma_val, 1.0 / float(total_lag_features))

    simulated_output = np.zeros((total_samples, n_y), dtype=float)

    if total_samples <= max_delay_lag or len(x_train_fit) == 0:
        simulated_output[:] = np.inf
        return (simulated_output.flatten() if y_was_1d else simulated_output), max_delay_lag

    simulated_output[:max_delay_lag, :] = y_mat[:max_delay_lag, :]

    sv_list, dual_list, intercept_list = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for out_ch in range(n_y):
            try:
                svr = SVR(kernel="rbf", C=c_val, epsilon=eps_val, gamma=gamma_val, tol=1e-2, cache_size=500, max_iter=2000)
                svr.fit(x_train_fit, y_train_fit[:, out_ch])
                sv_list.append(svr.support_vectors_)
                # If epsilon is wider than the signal spread, sklearn fits 0 support vectors and dual_coef_ is shape (0, 0).
                # Using .ravel() instead of [0] prevents an IndexError when the support vector array is empty.
                dual_list.append(svr.dual_coef_.ravel())
                intercept_list.append(float(np.ravel(svr.intercept_)[0]) if len(np.ravel(svr.intercept_)) > 0 else 0.0)
            except Exception:
                simulated_output[:] = np.inf
                return (simulated_output.flatten() if y_was_1d else simulated_output), max_delay_lag

    reg = np.zeros(total_lag_features, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for t in range(max_delay_lag, total_samples):
            col_idx = 0
            for out_ch in range(n_y):
                for i in range(na):
                    reg[col_idx] = simulated_output[t - 1 - i, out_ch]
                    col_idx += 1
            for in_ch in range(n_u):
                for j in range(nb):
                    lag = nk + j
                    reg[col_idx] = u_mat[t - lag, in_ch]
                    col_idx += 1

            for out_ch in range(n_y):
                sv_mat = sv_list[out_ch]
                if len(sv_mat) == 0:
                    y_hat_step = intercept_list[out_ch]
                else:
                    diff = sv_mat - reg
                    # Direct dual evaluation: y_hat = sum(dual_coef * exp(-gamma * ||x - sv||^2)) + intercept
                    # Vectorized with einsum to avoid calling svr.predict() point by point
                    k_vec = np.exp(-gamma_val * np.einsum("ij,ij->i", diff, diff))
                    y_hat_step = float(np.dot(dual_list[out_ch], k_vec) + intercept_list[out_ch])
                if not np.isfinite(y_hat_step) or abs(y_hat_step) > 1e10:
                    simulated_output[:] = np.inf
                    return (simulated_output.flatten() if y_was_1d else simulated_output), max_delay_lag
                simulated_output[t, out_ch] = y_hat_step

    if y_was_1d:
        return simulated_output.flatten(), max_delay_lag
    return simulated_output, max_delay_lag
