"""
armax_simulator.py

Simulates the discrete-time ARMAX model structure.
Implements the difference equation: A(q)y(t) = B(q)u(t-nk) + C(q)e(t).
"""

import numpy as np
import warnings


def simulate_armax_system(u_signal, y_signal, theta_weights, na, nb, nc, nk):
    r"""
    Computes the one-step-ahead prediction \hat{y}(t|\theta).

    Parameters:
    -----------
    u_signal : np.ndarray
        Input signal array u(t).
    y_signal : np.ndarray
        Actual system output array y(t).
    theta_weights : list or np.ndarray
        Combined parameter vector [a_1...a_na, b_1...b_nb, c_1...c_nc].
    na : int
        Number of AR parameters.
    nb : int
        Number of input (B) parameters.
    nc : int
        Number of MA error parameters.
    nk : int
        Time delay in the input.

    Returns:
    --------
    predicted_output : np.ndarray
        The one-step-ahead predictions.
    max_delay_lag : int
        The initial time step required to buffer historical data.
    """
    total_samples = len(y_signal)

    # The maximum lag dictates our "burn-in" period. We cannot compute predictions
    # for t < max_delay_lag because we lack sufficient historical inputs/outputs.
    max_delay_lag = max(na, nb + nk - 1, nc)

    predicted_output = np.zeros(total_samples)
    residual_errors = np.zeros(total_samples)

    # Unpack the 1D search space vector into the distinct ARMAX polynomial coefficients
    theta_ar = theta_weights[:na]
    theta_exogenous = theta_weights[na: na + nb]
    theta_ma = theta_weights[na + nb: na + nb + nc]

    arx_component = np.zeros(total_samples)

    # Vectorize the deterministic ARX portion (A and B polynomials) for speed
    for i in range(na):
        arx_component[max_delay_lag:] -= theta_ar[i] * y_signal[max_delay_lag - 1 - i: total_samples - 1 - i]

    for j in range(nb):
        lag = nk + j
        arx_component[max_delay_lag:] += theta_exogenous[j] * u_signal[max_delay_lag - lag: total_samples - lag]

    # Evolutionary algorithms will inevitably propose unstable parameters during early generations.
    # We suppress numpy overflow warnings here and let the fitness function heavily penalize np.inf later.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # The MA component depends on past prediction errors, forcing sequential evaluation
        for t in range(max_delay_lag, total_samples):
            ma_component = 0.0
            for k in range(nc):
                ma_component += theta_ma[k] * residual_errors[t - 1 - k]

            current_prediction = arx_component[t] + ma_component

            if not np.isfinite(current_prediction) or abs(current_prediction) > 1e10:
                predicted_output[:] = np.inf
                break

            predicted_output[t] = current_prediction
            residual_errors[t] = y_signal[t] - current_prediction

    return predicted_output, max_delay_lag