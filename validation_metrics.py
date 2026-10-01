"""
validation_metrics.py

Model validation metrics for ARMAX simulation.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def compute_mse(actual_targets, predicted_outputs):
    """
    Computes the Mean Squared Error (MSE).
    """
    return mean_squared_error(actual_targets, predicted_outputs)


def compute_mape(actual_targets, predicted_outputs):
    """
    Computes the Mean Absolute Percentage Error (MAPE).
    Masks zero-value targets to avoid division by zero.
    """
    # Thresholding at 1e-5 avoids exploding MAPE values during transients
    # or instances where the true y(t) briefly crosses zero.
    valid_mask = np.abs(actual_targets) > 1e-5
    if not np.any(valid_mask):
        return 0.0

    absolute_percentage_errors = np.abs(
        (actual_targets[valid_mask] - predicted_outputs[valid_mask]) / actual_targets[valid_mask]
    )
    return np.mean(absolute_percentage_errors) * 100


def generate_validation_report(actual_targets, predicted_outputs, model_parameter_count=6):
    r"""
    Calculates evaluation metrics for the model.

    Parameters:
    -----------
    actual_targets : np.ndarray
        The true system outputs y(t).
    predicted_outputs : np.ndarray
        The model predictions \hat{y}(t|\theta).
    model_parameter_count : int
        Total parameters in the model (na + nb + nc), used for AIC calculation.

    Returns:
    --------
    dict
        A dictionary containing MSE, RMSE, MAPE, R-Squared, and AIC scores.
    """
    sample_size = len(actual_targets)
    mse_score = compute_mse(actual_targets, predicted_outputs)
    rmse_score = np.sqrt(mse_score)

    # Assuming normally distributed errors, AIC can be approximated using MSE
    # instead of the full log-likelihood function.
    aic_score = -np.inf if mse_score == 0 else sample_size * np.log(mse_score) + 2 * model_parameter_count

    return {
        "MSE": mse_score,
        "RMSE": rmse_score,
        "MAPE": compute_mape(actual_targets, predicted_outputs),
        "R-Squared": r2_score(actual_targets, predicted_outputs),
        "AIC": aic_score
    }