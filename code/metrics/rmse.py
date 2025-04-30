import numpy as np

def RMSE(yhat: float, y: float, eps: float = 1e-6) -> float:
    """
    Compute the Root Mean Square Error (RMSE) between two scalar values.

    This metric quantifies the average squared difference between a predicted value and the true value,
    and returns the result in the same units as the original values. A small epsilon is added to ensure
    numerical stability.

    Args:
        yhat (float): Predicted scalar value.
        y (float): True scalar value.
        eps (float, optional): Small value to prevent numerical instability. Defaults to 1e-6.

    Returns:
        float: RMSE value.
    """
    yhat, y = float(yhat), float(y)
    mse = (yhat - y) ** 2
    return np.sqrt(mse + eps)