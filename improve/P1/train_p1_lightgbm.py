import sys
from pathlib import Path
import pandas as pd
import joblib
import lightgbm as lgb
from lightgbm import LGBMRegressor

# Đảm bảo UTF-8 stream trên Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thêm BASE_DIR vào sys.path để truy cập Config và Metric
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from Config import load_config, get_path


def load_data() -> pd.DataFrame:
    """
    Tải dữ liệu đã tiền xử lý từ processed_train.parquet.
    """
    config = load_config()
    data_path = get_path(config["paths"]["files"]["processed_train_parquet"])
    print(f"[1/3] Đang tải dữ liệu từ {data_path.name}...")
    df = pd.read_parquet(data_path)
    print(f"      Đã tải {len(df):,} dòng và {df.shape[1]} cột.")
    return df


def split_time_series(df: pd.DataFrame):
    """
    Chia dữ liệu thành train, validation Q4/2011 và holdout từ năm 2012 theo chuẩn chuỗi thời gian.
    """
    config = load_config()
    data_config = config.get("data", {})
    validation_start = pd.Timestamp(data_config.get("validation_start", "2011-10-01"))
    test_start = pd.Timestamp(data_config.get("test_start", "2012-01-01"))

    if validation_start >= test_start:
        raise ValueError("validation_start phải nằm trước test_start.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Store", "Dept"]).reset_index(drop=True)

    train_df = df[df["Date"] < validation_start].copy()
    val_df = df[(df["Date"] >= validation_start) & (df["Date"] < test_start)].copy()
    test_df = df[df["Date"] >= test_start].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train, validation hoặc holdout rỗng; kiểm tra ngày trong config và dữ liệu.")

    print("\n===== TIME SERIES SPLIT (CHỐNG RÒ RỈ DỮ LIỆU) =====")
    print(f"Validation start: {validation_start}")
    print(f"Holdout start:    {test_start}")
    print(f"Train period:      {train_df['Date'].min().strftime('%Y-%m-%d')} -> {train_df['Date'].max().strftime('%Y-%m-%d')} ({len(train_df):,} dòng)")
    print(f"Validation period: {val_df['Date'].min().strftime('%Y-%m-%d')} -> {val_df['Date'].max().strftime('%Y-%m-%d')} ({len(val_df):,} dòng)")
    print(f"Holdout period:    {test_df['Date'].min().strftime('%Y-%m-%d')} -> {test_df['Date'].max().strftime('%Y-%m-%d')} ({len(test_df):,} dòng)")

    return train_df, val_df, test_df


def train_lightgbm_p1():
    """
    P1 QUICK WINS:
    1. Chuyển hàm mục tiêu từ 'regression' (L2/MSE) sang 'regression_l1' (L1/MAE)
       để đồng bộ hoàn toàn với hàm đánh giá Weighted MAE (WMAE).
    2. Sử dụng sample_weight chuẩn Walmart (x5 cho tuần lễ, x1 cho tuần thường).
    3. Lưu checkpoint vào improve/P1/checkpoint/lightgbm_l1_model.pkl.
    """
    checkpoint_dir = CURRENT_DIR / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train_df, val_df, test_df = split_time_series(df)

    target = "Weekly_Sales"
    drop_columns = ["Weekly_Sales", "Date"]

    X_train = train_df.drop(columns=drop_columns)
    y_train = train_df[target]

    X_val = val_df.drop(columns=drop_columns)
    y_val = val_df[target]

    # WMAE sample weights: x5 cho tuần lễ, x1 cho tuần thường
    train_weight = train_df["IsHoliday"].map({0: 1.0, 1: 5.0}).to_numpy()
    val_weight = val_df["IsHoliday"].map({0: 1.0, 1: 5.0}).to_numpy()

    # Cấu hình thiết bị huấn luyện (GPU CUDA / OpenCL / CPU đa nhân)
    print("\n[Cấu hình thiết bị huấn luyện LightGBM P1]...")
    base_params = {
        "objective": "regression_l1",
        "metric": "l1",
        "n_estimators": 5000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "random_state": 42
    }

    used_device = "CPU"
    model = None

    for target_dev in ["cuda", "gpu"]:
        try:
            print(f"      Đang kiểm tra khả năng kích hoạt phần cứng GPU (device='{target_dev}')...")
            test_params = base_params.copy()
            test_params["device"] = target_dev
            test_params["verbose"] = -1
            test_model = LGBMRegressor(**test_params)
            test_model.fit(X_train.iloc[:30], y_train.iloc[:30])
            model = LGBMRegressor(**test_params)
            used_device = f"GPU ({target_dev.upper()})"
            print(f"      [THÀNH CÔNG] Đã kích hoạt tăng tốc phần cứng {used_device} cho LightGBM P1!")
            break
        except Exception as e:
            err_msg = str(e).split("\n")[0]
            print(f"      [NOTE] Không thể khởi tạo GPU với device='{target_dev}' ({err_msg}).")

    if model is None:
        print("      [CHUYỂN ĐỔI] Chạy trên CPU đa luồng hiệu năng cao (OpenMP n_jobs=-1)...")
        cpu_params = base_params.copy()
        cpu_params["n_jobs"] = -1
        model = LGBMRegressor(**cpu_params)
        used_device = "CPU"

    print(f"\n[3/3] Bắt đầu huấn luyện LightGBM P1 trên {used_device} với Early Stopping 100 rounds...")
    model.fit(
        X_train,
        y_train,
        sample_weight=train_weight,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[val_weight],
        categorical_feature=["Store", "Dept", "Type"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )

    best_iteration = getattr(model, "best_iteration_", model.n_estimators)
    print(f"\n[HOÀN TẤT] Điểm dừng tối ưu (Best Iteration): {best_iteration}")

    model_path = checkpoint_dir / "lightgbm_l1_model.pkl"
    joblib.dump(model, model_path)
    print(f"[CHECKPOINT] Đã lưu model LightGBM P1 tại: {model_path}")

    return model


if __name__ == "__main__":
    train_lightgbm_p1()
