import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Thêm BASE_DIR vào sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from Config import load_config, get_path
from Metric.metrics import evaluate_all, format_metric_report

def load_preprocessed_data() -> pd.DataFrame:
    """
    Tải dữ liệu đã qua tiền xử lý từ Data/Preprocess Data.
    """
    # TODO: Đọc file processed_train.parquet từ config
    pass

def run_naive_baseline(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dự báo bằng mô hình Naive: Doanh số tuần sau bằng đúng tuần trước (y_t = y_{t-1}).
    """
    # TODO: Dự báo dựa trên lag_1
    pass

def run_seasonal_naive_baseline(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dự báo bằng mô hình Seasonal Naive: Doanh số bằng cùng kỳ tuần này năm trước (y_t = y_{t-52}).
    """
    # TODO: Dự báo dựa trên lag_52
    pass

def run_moving_average_baseline(df: pd.DataFrame, window: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dự báo bằng mô hình thống kê Moving Average: Trung bình trượt N tuần gần nhất.
    """
    # TODO: Dự báo bằng rolling mean N tuần
    pass

def evaluate_and_save_baselines():
    """
    Chạy toàn bộ các mô hình Baseline, tính toán các thang đo (WMAE, MAE, RMSE) và xuất ra file txt.
    """
    config = load_config()
    result_dir = get_path(config["paths"]["models"]["baseline"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Chạy các hàm baseline, đánh giá metric qua Metric.metrics và ghi file txt kết quả
    pass

if __name__ == "__main__":
    evaluate_and_save_baselines()
