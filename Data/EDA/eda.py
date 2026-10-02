import sys
from pathlib import Path
import pandas as pd

# Thêm đường dẫn gốc để import Config
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from Config import load_config, get_path

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tải dữ liệu thô từ thư mục Data/Origin (train.csv, features.csv, stores.csv).
    """
    # TODO: Đọc các file train.csv, features.csv, stores.csv từ config
    pass

def run_eda():
    """
    Hàm điều phối toàn bộ quy trình khám phá dữ liệu (EDA).
    """
    config = load_config()
    result_dir = get_path(config["paths"]["data"]["eda_result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Gọi lần lượt các hàm phân tích EDA và lưu biểu đồ vào result_dir
    pass

if __name__ == "__main__":
    run_eda()
