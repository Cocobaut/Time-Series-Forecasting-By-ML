import sys
from pathlib import Path
import pandas as pd

# Thêm BASE_DIR vào sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from Config import load_config, get_path

def load_data() -> pd.DataFrame:
    """
    Tải dữ liệu đã tiền xử lý từ config.
    """
    # TODO: Đọc dữ liệu từ đường dẫn cấu hình
    pass

def split_time_series(df: pd.DataFrame):
    """
    Phân chia dữ liệu theo thời gian (Train / Validation) để chống rò rỉ dữ liệu.
    """
    # TODO: Tách dữ liệu Train và Validation theo mốc ngày
    pass

def train_xgboost():
    """
    Cấu hình siêu tham số, thiết lập sample_weight cho WMAE, huấn luyện XGBoost và lưu checkpoint.
    """
    config = load_config()
    checkpoint_dir = get_path(config["paths"]["models"]["xgboost"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Huấn luyện XGBRegressor với early stopping và lưu model vào checkpoint_dir
    pass

if __name__ == "__main__":
    train_xgboost()
