import numpy as np
from scipy.stats import kendalltau, spearmanr


def kendall_tau(y_true, y_pred):
    """
    Calculates the Kendall Tau correlation coefficient between true and predicted values.

    Args:
    y_true (list): A list of true values.
    y_pred (list): A list of predicted values.

    Returns:
    float: The Kendall Tau correlation coefficient.
    """
    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the Kendall Tau correlation coefficient
    tau, _ = kendalltau(y_true, y_pred)

    return tau


def spearman_correlation(y_true, y_pred):
    """
    Calculates the Spearman correlation coefficient between true and predicted values.

    Args:
    y_true (list): A list of true values.
    y_pred (list): A list of predicted values.

    Returns:
    float: The Spearman correlation coefficient.
    """
    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the Spearman correlation coefficient
    rho, _ = spearmanr(y_true, y_pred)

    return rho
