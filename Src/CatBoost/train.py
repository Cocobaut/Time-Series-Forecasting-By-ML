import sys
from pathlib import Path
import warnings
import joblib
import pandas as pd
import numpy as np
import catboost as cb

# Đảm bảo UTF-8 stream trên Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# Thêm BASE_DIR vào sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from Config import load_config, get_path
from Metric.metrics import evaluate_all, format_metric_report


def load_data() -> pd.DataFrame:
    """
    Tải dữ liệu đã tiền xử lý từ config (ưu tiên định dạng Parquet, fallback sang CSV).
    Đồng thời làm sạch các giá trị vô cực (inf, -inf) và NaN để tương thích với GPU Pool.
    """
    config = load_config()
    parquet_path = get_path(config["paths"]["files"]["processed_train_parquet"])
    csv_path = get_path(config["paths"]["files"]["processed_train_csv"])

    if parquet_path.exists():
        print(f"[1/4] Đang nạp dữ liệu từ Parquet: {parquet_path.name}...")
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        print(f"[1/4] Đang nạp dữ liệu từ CSV: {csv_path.name}...")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tiền xử lý tại {parquet_path} hoặc {csv_path}")

    df['Date'] = pd.to_datetime(df['Date'])

    # Làm sạch các giá trị vô cực (inf, -inf) và NaN nếu phát sinh trong phép chia tỷ lệ
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(0.0)

    print(f"      Đã nạp {df.shape[0]:,} dòng và {df.shape[1]} cột.")
    return df


def split_time_series(df: pd.DataFrame):
    """
    Phân chia dữ liệu theo chuỗi thời gian để chống rò rỉ dữ liệu (Lookahead Bias):
      - Train set: Dữ liệu trước 2011-10-01 (dùng để học quy luật lịch sử)
      - Validation set: Từ 2011-10-01 đến 2011-12-31 (mùa mua sắm lớn Thanksgiving, Black Friday, Christmas)
      - Out-of-sample Test set: Từ 2012-01-01 trở đi (đánh giá kiểm thử tương lai)
    """
    print("[2/4] Phân chia dữ liệu theo mốc thời gian (Time-based Split)...")

    # Xác định danh sách cột đặc trưng (Feature Columns)
    exclude_cols = ['Date', 'Weekly_Sales']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    cat_cols = [c for c in ['Store', 'Dept', 'Type'] if c in feature_cols]

    # Đảm bảo kiểu số nguyên cho các đặc trưng phân loại của CatBoost
    for c in cat_cols:
        df[c] = df[c].astype(int)

    # Phân chia thời gian
    train_mask = df['Date'] < '2011-10-01'
    val_mask = (df['Date'] >= '2011-10-01') & (df['Date'] <= '2011-12-31')
    test_mask = df['Date'] >= '2012-01-01'

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    X_train, y_train = train_df[feature_cols], train_df['Weekly_Sales']
    X_val, y_val = val_df[feature_cols], val_df['Weekly_Sales']
    X_test, y_test = test_df[feature_cols], test_df['Weekly_Sales']

    # Trọng số WMAE: x5 cho tuần có ngày lễ (IsHoliday == 1), x1 cho tuần thường
    w_train = np.where(train_df['IsHoliday'] == 1, 5.0, 1.0)
    w_val = np.where(val_df['IsHoliday'] == 1, 5.0, 1.0)
    w_test = np.where(test_df['IsHoliday'] == 1, 5.0, 1.0)

    print(f"      + Tập Huấn luyện (Train):      {len(train_df):,} dòng (trước 2011-10-01)")
    print(f"      + Tập Kiểm định (Validation):  {len(val_df):,} dòng (2011-10-01 đến 2011-12-31)")
    print(f"      + Tập Kiểm thử (Holdout Test): {len(test_df):,} dòng (từ 2012-01-01 trở đi)")
    print(f"      + Số lượng đặc trưng:          {len(feature_cols)} features ({cat_cols} là cat_features)")

    return {
        "X_train": X_train, "y_train": y_train, "w_train": w_train, "val_df": val_df,
        "X_val": X_val, "y_val": y_val, "w_val": w_val, "test_df": test_df,
        "X_test": X_test, "y_test": y_test, "w_test": w_test,
        "feature_cols": feature_cols, "cat_cols": cat_cols
    }


def train_catboost():
    """
    Cấu hình siêu tham số, thiết lập cat_features, sample_weight cho WMAE,
    huấn luyện CatBoost với tăng tốc GPU (CUDA) và lưu checkpoint.
    """
    print("=" * 80)
    print(" BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH CATBOOST (TIME SERIES FORECASTING VỚI GPU)")
    print("=" * 80)

    config = load_config()
    checkpoint_dir = get_path(config["paths"]["models"]["catboost"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Nạp và phân chia dữ liệu
    df = load_data()
    data = split_time_series(df)

    X_train, y_train, w_train = data["X_train"], data["y_train"], data["w_train"]
    X_val, y_val, w_val = data["X_val"], data["y_val"], data["w_val"]
    cat_cols = data["cat_cols"]
    feature_cols = data["feature_cols"]

    # 2. Xây dựng đối tượng Pool tối ưu của CatBoost
    print("\n[3/4] Chuẩn bị CatBoost Pool và cấu hình thiết bị...")
    train_pool = cb.Pool(X_train, y_train, weight=w_train, cat_features=cat_cols)
    val_pool = cb.Pool(X_val, y_val, weight=w_val, cat_features=cat_cols)

    # 3. Cấu hình siêu tham số
    # Trên GPU của CatBoost, loss_function="RMSE" kết hợp sample_weight (x5 holiday) và eval_metric="MAE"
    # là cấu hình chuẩn xác nhất để hội tụ nhanh và tối ưu WMAE (do MAE loss trên GPU của CatBoost chưa hoàn thiện)
    base_params = {
        "iterations": 1500,
        "learning_rate": 0.08,
        "depth": 8,
        "loss_function": "RMSE",
        "eval_metric": "MAE",
        "random_seed": 42,
        "early_stopping_rounds": 50,
        "verbose": 100
    }

    used_device = "GPU"
    try:
        print("      Đang khởi tạo CatBoost với GPU (CUDA acceleration)...")
        gpu_params = base_params.copy()
        gpu_params["task_type"] = "GPU"
        model = cb.CatBoostRegressor(**gpu_params)

        # Kiểm tra nhanh GPU với slice nhỏ
        test_pool = cb.Pool(X_train.iloc[:50], y_train.iloc[:50], weight=w_train[:50], cat_features=cat_cols)
        _ = model.fit(test_pool, eval_set=test_pool, verbose=False)
        print("      [THÀNH CÔNG] Đã kích hoạt tăng tốc phần cứng GPU cho CatBoost!")
        model = cb.CatBoostRegressor(**gpu_params)
    except Exception as e:
        print(f"      [LƯU Ý] Không thể chạy GPU trên CatBoost ({e}).")
        print("      [CHUYỂN ĐỔI] Chuyển đổi sang chế độ đa luồng CPU (thread_count=-1)...")
        cpu_params = base_params.copy()
        cpu_params["task_type"] = "CPU"
        cpu_params["thread_count"] = -1
        model = cb.CatBoostRegressor(**cpu_params)
        used_device = "CPU"

    # 4. Huấn luyện mô hình với Early Stopping
    print(f"\n[4/4] Bắt đầu huấn luyện CatBoost trên {used_device} (Early Stopping 50 rounds)...")
    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=100
    )

    best_iter = model.get_best_iteration()
    print(f"\n[HOÀN THÀNH] Huấn luyện dừng tại iteration tối ưu: {best_iter}")

    # Đánh giá nhanh trên tập Validation
    val_preds = model.predict(val_pool)
    val_metrics = evaluate_all(y_val, val_preds, data["val_df"]['IsHoliday'])
    print("\n" + format_metric_report("CatBoost (Validation Set - Sơ bộ)", val_metrics))

    # Lưu checkpoint mô hình
    checkpoint_file = checkpoint_dir / "catboost_model.pkl"
    checkpoint_data = {
        "model": model,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "best_iteration": best_iter,
        "device": used_device,
        "params": model.get_params()
    }
    joblib.dump(checkpoint_data, checkpoint_file)
    print(f"[LƯU CHECKPOINT] Đã lưu model và metadata thành công tại: {checkpoint_file}")

    # Lưu model nhị phân chuẩn của CatBoost (.cbm)
    try:
        cbm_file = checkpoint_dir / "catboost_model.cbm"
        model.save_model(str(cbm_file))
        print(f"[LƯU CBM MODEL] Đã lưu mô hình CatBoost binary tại: {cbm_file}")
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể lưu file .cbm: {e}")

    return model, val_metrics


if __name__ == "__main__":
    train_catboost()
