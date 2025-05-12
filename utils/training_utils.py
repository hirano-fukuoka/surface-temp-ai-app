import numpy as np

def compute_metrics(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    return mae, rmse
