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

def seasonal_naive_scale(y_train, seasonality: int = 52) -> float:
    """
    Mẫu số của MASE cho MỘT chuỗi: sai số trung bình của dự báo Seasonal Naive trong mẫu Train
        scale = 1/(N - m) * sum_{t=m+1}^{N} |y_t - y_{t-m}|
    Bỏ qua các cặp có NaN (tuần khuyết dữ liệu).
    """
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= seasonality:
        raise ValueError(f"Chuỗi train cần dài hơn seasonality={seasonality} để tính MASE.")
    diffs = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    return float(np.nanmean(diffs))

def compute_mase_scale(train_df: pd.DataFrame, seasonality: int = 52,
                       group_cols=("Store", "Dept"), date_col: str = "Date",
                       target_col: str = "Weekly_Sales", freq: str = "W-FRI") -> float:
    """
    Mẫu số MASE cho dữ liệu PANEL (3,331 chuỗi Store x Dept):
    - Tái lập lưới tuần liên tục cho từng chuỗi (tuần khuyết = NaN) để phép trễ m tuần không lệch pha.
    - Gộp toàn bộ |y_t - y_{t-m}| hợp lệ của mọi chuỗi rồi lấy trung bình (pooled scale).
    Với m = 52 đây là sai số của Seasonal Naive cùng kỳ năm trước; với m = 1 là Naive 1 bước.
    """
    group_cols = list(group_cols)
    df = train_df[group_cols + [date_col, target_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])

    wide = df.pivot_table(index=group_cols, columns=date_col, values=target_col, aggfunc="sum")
    full_dates = pd.date_range(wide.columns.min(), wide.columns.max(), freq=freq)
    wide = wide.reindex(columns=full_dates)

    values = wide.to_numpy(dtype=float)
    if values.shape[1] <= seasonality:
        raise ValueError(f"Lịch sử train ({values.shape[1]} tuần) phải dài hơn seasonality={seasonality}.")
    diffs = np.abs(values[:, seasonality:] - values[:, :-seasonality])
    scale = float(np.nanmean(diffs))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Không tính được mẫu số MASE hợp lệ (scale <= 0 hoặc NaN).")
    return scale

def calculate_mase(y_true, y_pred, y_train=None, seasonality: int = 52, scale: float = None) -> float:
    """
    Tính Mean Absolute Scaled Error (MASE) theo Hyndman & Koehler (2006):
        MASE = MAE(dự báo) / [1/(N - m) * sum_{t=m+1}^{N} |y_t - y_{t-m}|]
    - y_train: chuỗi train của MỘT chuỗi thời gian (dùng để tính mẫu số).
    - scale: mẫu số tính sẵn (dùng cho dữ liệu panel, xem compute_mase_scale). Ưu tiên nếu được truyền.
    MASE < 1 nghĩa là tốt hơn Seasonal Naive (m = 52) / Naive (m = 1) trong mẫu.
    """
    if scale is None:
        if y_train is None:
            raise ValueError("Cần truyền y_train hoặc scale để tính MASE.")
        scale = seasonal_naive_scale(y_train, seasonality)
    mae = float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))
    return float(mae / scale)

def evaluate_all(y_true, y_pred, is_holiday, y_train=None, seasonality: int = 52, mase_scale: float = None):
    """
    Đánh giá toàn diện tất cả các thang đo và trả về dict kết quả.
    MASE chỉ được tính khi truyền y_train (một chuỗi) hoặc mase_scale (panel, từ compute_mase_scale).
    """
    wmae = calculate_wmae(y_true, y_pred, is_holiday)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = calculate_mape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))

    results = {
        "WMAE": wmae,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R2": r2
    }
    if mase_scale is not None or y_train is not None:
        results["MASE"] = calculate_mase(y_true, y_pred, y_train=y_train,
                                         seasonality=seasonality, scale=mase_scale)
    return results

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
    ]
    if "MASE" in metrics:
        report.append(f" 6. MASE (Sai số chuẩn hóa)   : {metrics['MASE']:.4f}")
    report.append(f"==================================================\n")
    return "\n".join(report)
