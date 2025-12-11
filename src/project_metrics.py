import numpy as np
import pandas as pd

def calculate_mase_scale_factor(y_train):
    y_train = np.array(y_train)

    naive_errors = np.abs(np.diff(y_train))

    scale_factor = np.mean(naive_errors)

    return scale_factor if scale_factor != 0 else 1e-6 #avoid division by zero


def calculate_mase_score(y_true, y_pred, scale_factor):
    model_errors = np.abs(y_true - y_pred)
    model_mae = np.mean(model_errors)

    mase_score = model_mae / scale_factor

    return mase_score

    