import numpy as np
from src.config import * 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def metrics(y_test, y_pred):
    """
    Ini untuk menghitung metrics
    """
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }

def val_metrics(y_val, y_pred):
    """
    Ini untuk menghitung metrics validation set
    """
    return {
        "val_mae": mean_absolute_error(y_val, y_pred),
        "val_mse": mean_squared_error(y_val, y_pred),
        "val_rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
        "val_r2": r2_score(y_val, y_pred),
    }
