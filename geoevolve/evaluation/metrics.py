import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)

def r2(y_true, y_pred):
    """R-squared coefficient"""
    return r2_score(y_true, y_pred)

def compute_all_metrics(y_true, y_pred):
    """Compute all metrics at once"""
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'r2': r2(y_true, y_pred)
    }
