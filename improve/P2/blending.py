import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

CURRENT_DIR = Path(__file__).resolve().parent
IMPROVE_DIR = CURRENT_DIR.parent
BASE_DIR = IMPROVE_DIR.parent
sys.path.append(str(BASE_DIR))

from Metric.metrics import calculate_wmae, evaluate_all


def weighted_blend(predictions: list, weights: list) -> np.ndarray:
    """
    Kết hợp dự báo của nhiều mô hình theo trọng số chuẩn hóa:
    y_blend = sum(w_i * y_i)
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)

    blend = np.zeros_like(predictions[0], dtype=float)
    for p, w in zip(predictions, weights):
        blend += w * np.asarray(p, dtype=float)
    return blend


def find_optimal_weights(val_preds: list, y_val: np.ndarray, val_holiday: np.ndarray, init_weights=None) -> np.ndarray:
    """
    Tìm bộ trọng số tối ưu (w_1, w_2, ..., w_n) giảm thiểu WMAE trên tập Validation:
    Ràng buộc: sum(w_i) = 1.0 và 0 <= w_i <= 1.0.
    """
    n_models = len(val_preds)
    if init_weights is None:
        init_weights = [1.0 / n_models] * n_models

    def loss(w):
        pred = weighted_blend(val_preds, w)
        return calculate_wmae(y_val, pred, val_holiday)

    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        loss,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-6, "maxiter": 200}
    )

    opt_w = np.clip(res.x, 0.0, 1.0)
    opt_w = opt_w / np.sum(opt_w)
    return opt_w
