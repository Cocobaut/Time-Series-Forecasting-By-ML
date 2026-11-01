"""
Tiện ích dùng chung cho các kịch bản thí nghiệm (tuning, multi-seed, CV, benchmark):
nạp dữ liệu đã tiền xử lý, chia tập theo thời gian, chọn đặc trưng, lưu kết quả.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from Config import load_config, get_path, get_result_dir

# Một bản ghi phần cứng duy nhất cho toàn dự án, được ghi đè ở mỗi lần chạy thực nghiệm
HARDWARE_INFO_PATH = get_path(load_config()["paths"]["results"]["root"]) / "hardware_info.json"


def setup_console():
    """Đảm bảo UTF-8 stream trên Windows console."""
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def load_processed_train() -> pd.DataFrame:
    """
    Nạp bảng train đã tiền xử lý (Parquet ưu tiên, CSV dự phòng),
    làm sạch inf/NaN của các đặc trưng tỷ lệ để tương thích cả 3 thư viện GBDT.
    """
    config = load_config()
    parquet_path = get_path(config["paths"]["files"]["processed_train_parquet"])
    csv_path = get_path(config["paths"]["files"]["processed_train_csv"])
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu tiền xử lý tại {parquet_path}. Hãy chạy Data/Preprocess/preprocess.py trước."
        )
    return clean_numeric(df)


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa Date và thay inf/-inf/NaN ở cột số bằng 0."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.sort_values(["Date", "Store", "Dept"]).reset_index(drop=True)


def static_split(df: pd.DataFrame, data_cfg: dict = None):
    """
    Chia dữ liệu theo dòng thời gian (không xáo trộn):
      Train < validation_start, Val = [validation_start, test_start), Holdout test >= test_start.
    """
    data_cfg = data_cfg or {}
    validation_start = pd.Timestamp(data_cfg.get("validation_start", "2011-10-01"))
    test_start = pd.Timestamp(data_cfg.get("test_start", "2012-01-01"))
    if validation_start >= test_start:
        raise ValueError("validation_start phải nằm trước test_start.")

    train_df = df[df["Date"] < validation_start].copy()
    val_df = df[(df["Date"] >= validation_start) & (df["Date"] < test_start)].copy()
    test_df = df[df["Date"] >= test_start].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train, validation hoặc holdout rỗng; kiểm tra ngày chia trong config.")
    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame, features_cfg: dict, mode: str = None) -> list:
    """
    Danh sách đặc trưng đầu vào theo config:
      - bỏ drop_columns (target, Date)
      - chế độ "horizon_safe" bỏ thêm các lag/rolling có độ trễ ngắn hơn horizon dự báo.
    """
    mode = mode or features_cfg.get("mode", "one_step")
    drop = set(features_cfg.get("drop_columns", ["Weekly_Sales", "Date"]))
    if mode == "horizon_safe":
        drop |= set(features_cfg.get("horizon_unsafe", []))
    elif mode != "one_step":
        raise ValueError(f"features.mode không hợp lệ: {mode} (chọn 'one_step' hoặc 'horizon_safe')")
    return [c for c in df.columns if c not in drop]


def holiday_weights(df: pd.DataFrame, holiday_weight: float = 5.0) -> np.ndarray:
    """Trọng số mẫu chuẩn WMAE: x5 cho tuần lễ, x1 cho tuần thường."""
    return np.where(df["IsHoliday"].to_numpy() == 1, holiday_weight, 1.0)


def sample_series(df: pd.DataFrame, frac: float, seed: int = 0) -> pd.DataFrame:
    """
    Lấy ngẫu nhiên một phần các chuỗi (Store, Dept) — giữ nguyên toàn bộ trục thời gian của chuỗi.
    Dùng cho chế độ --quick (chạy thử nhanh).
    """
    if frac >= 1.0:
        return df
    pairs = df[["Store", "Dept"]].drop_duplicates()
    chosen = pairs.sample(frac=frac, random_state=seed)
    return df.merge(chosen, on=["Store", "Dept"], how="inner")


def mean_std(values) -> dict:
    """Trung bình và độ lệch chuẩn mẫu (ddof=1) qua các seed."""
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std, "n": int(len(arr))}


def format_mean_std(stat: dict, fmt: str = ",.2f") -> str:
    """Định dạng 'mean ± std'."""
    if stat is None:
        return "—"
    return f"{stat['mean']:{fmt}} ± {stat['std']:{fmt}}"


def tuning_dir(cfg: dict, quick: bool = False) -> Path:
    """results/03_tuning/<thí nghiệm>[_quick]/ — best_params.json, trials.csv."""
    return get_result_dir("tuning", cfg["experiment"]["name"] + ("_quick" if quick else ""))


def multiseed_dir(cfg: dict) -> Path:
    """results/05_multiseed/<thí nghiệm>/ — kết quả đa seed của một thí nghiệm."""
    return get_result_dir("multiseed", cfg["experiment"]["name"])


def load_tuned_params(cfg: dict):
    """Đọc bộ siêu tham số tốt nhất do Src/Experiments/tune.py sinh ra (None nếu chưa tuning)."""
    path = tuning_dir(cfg) / "best_params.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Không serialize được kiểu {type(obj)}")


def save_json(obj, path):
    """Ghi dict ra JSON (hỗ trợ kiểu numpy/pandas)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path):
    """Đọc JSON, trả về None nếu file chưa tồn tại."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def markdown_table(headers: list, rows: list, align: list = None) -> str:
    """Sinh bảng Markdown từ header và các dòng (list of list)."""
    align = align or ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)
