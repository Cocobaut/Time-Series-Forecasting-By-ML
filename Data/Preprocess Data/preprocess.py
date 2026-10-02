import sys
from pathlib import Path
import pandas as pd

# Thêm BASE_DIR vào đường dẫn hệ thống
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from Config import load_config, get_path

def load_data() -> pd.DataFrame:
    """
    Đọc và kết hợp train.csv, features.csv, stores.csv.
    """
    # TODO: Đọc dữ liệu từ thư mục Origin và merge theo Store, Date, IsHoliday
    pass

def preprocess_pipeline():
    """
    Điều phối toàn bộ quy trình tiền xử lý và lưu kết quả ra file parquet/csv trong Data/Preprocess Data/data/.
    """
    config = load_config()
    output_dir = get_path(config["paths"]["data"]["preprocess_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Thực thi toàn bộ pipeline và lưu vào Data/Preprocess Data/data/processed_train.parquet
    pass

if __name__ == "__main__":
    preprocess_pipeline()
