import sys
from pathlib import Path

# Thêm BASE_DIR vào sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from Config import load_config, get_path
from Metric.metrics import evaluate_all, format_metric_report

def evaluate_xgboost():
    """
    Nạp mô hình XGBoost từ checkpoint, dự báo trên tập Test, tính toán các thang đo và lưu file txt.
    """
    config = load_config()
    checkpoint_dir = get_path(config["paths"]["models"]["xgboost"]["checkpoint_dir"])
    result_dir = get_path(config["paths"]["models"]["xgboost"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Load model từ checkpoint_dir, dự đoán trên test set và xuất báo cáo vào result_dir
    pass

if __name__ == "__main__":
    evaluate_xgboost()
