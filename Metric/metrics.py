import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_wmae(y_true, y_pred, is_holiday):
    """
    Tính Weighted Mean Absolute Error (WMAE) theo chuẩn của Walmart Recruiting Challenge:
    Trọng số w = 5 nếu tuần là Holiday, w = 1 nếu là tuần bình thường.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_holiday = np.asarray(is_holiday)

    weights = np.where(is_holiday == 1, 5.0, 1.0)
    wmae = np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
    return float(wmae)

def calculate_mape(y_true, y_pred):
    """Tính MAPE, loại bỏ trường hợp y_true <= 0 để tránh chia cho 0."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true > 0
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def evaluate_all(y_true, y_pred, is_holiday):
    """Đánh giá toàn diện tất cả các thang đo và trả về dict kết quả."""
    wmae = calculate_wmae(y_true, y_pred, is_holiday)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = calculate_mape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))

    return {
        "WMAE": wmae,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R2": r2
    }

def format_metric_report(model_name: str, metrics: dict) -> str:
    """Định dạng kết quả metric thành chuỗi văn bản chuyên nghiệp."""
    report = [
        f"==================================================",
        f" BÁO CÁO ĐÁNH GIÁ MÔ HÌNH: {model_name.upper()}",
        f"==================================================",
        f" 1. WMAE (Trọng số ngày lễ x5) : {metrics['WMAE']:,.2f} USD",
        f" 2. MAE (Sai số tuyệt đối TB)  : {metrics['MAE']:,.2f} USD",
        f" 3. RMSE (Căn bậc hai MSE)    : {metrics['RMSE']:,.2f} USD",
        f" 4. MAPE (Sai số phần trăm TB) : {metrics['MAPE (%)']:.2f} %",
        f" 5. R2 Score (Hệ số xác định) : {metrics['R2']:.4f}",
        f"==================================================\n"
    ]
    return "\n".join(report)
