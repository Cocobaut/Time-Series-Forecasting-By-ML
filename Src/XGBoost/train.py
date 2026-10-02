import sys
from pathlib import Path
import warnings
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

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
    Đồng thời làm sạch các giá trị vô cực (inf, -inf) và NaN để tương thích với GPU DMatrix.
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

    # Chuyển đổi kiểu dữ liệu sang 'category' để XGBoost sử dụng native categorical features (enable_categorical=True)
    for c in cat_cols:
        df[c] = df[c].astype('category')

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
    print(f"      + Số lượng đặc trưng:          {len(feature_cols)} features ({cat_cols} là categorical)")

    return {
        "X_train": X_train, "y_train": y_train, "w_train": w_train, "val_df": val_df,
        "X_val": X_val, "y_val": y_val, "w_val": w_val, "test_df": test_df,
        "X_test": X_test, "y_test": y_test, "w_test": w_test,
        "feature_cols": feature_cols, "cat_cols": cat_cols
    }


def train_xgboost():
    """
    Cấu hình siêu tham số, kích hoạt tăng tốc phần cứng GPU (CUDA) cho XGBoost,
    tối ưu hàm mục tiêu L1 kết hợp sample_weight cho WMAE, huấn luyện với Early Stopping và lưu checkpoint.
    """
    print("=" * 80)
    print(" BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH XGBOOST (TIME SERIES FORECASTING VỚI GPU CUDA)")
    print("=" * 80)

    config = load_config()
    checkpoint_dir = get_path(config["paths"]["models"]["xgboost"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Nạp và phân chia dữ liệu
    df = load_data()
    data = split_time_series(df)

    X_train, y_train, w_train = data["X_train"], data["y_train"], data["w_train"]
    X_val, y_val, w_val = data["X_val"], data["y_val"], data["w_val"]
    cat_cols = data["cat_cols"]
    feature_cols = data["feature_cols"]

    # 2. Cấu hình siêu tham số cho XGBoost với GPU CUDA
    # objective="reg:absoluteerror" (L1 Loss) kết hợp sample_weight giúp mô hình tối ưu trực tiếp WMAE
    base_params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "enable_categorical": True,
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "early_stopping_rounds": 50
    }

    # 3. Khởi tạo mô hình và kích hoạt GPU CUDA
    print("\n[3/4] Cấu hình thiết bị huấn luyện...")
    used_device = "cuda"
    try:
        print("      Đang khởi tạo XGBoost với GPU CUDA (NVIDIA GeForce RTX)...")
        gpu_params = base_params.copy()
        gpu_params["device"] = "cuda"
        model = xgb.XGBRegressor(**gpu_params)

        # Kiểm tra thử nghiệm GPU CUDA
        _ = model.fit(
            X_train.iloc[:50], y_train.iloc[:50],
            sample_weight=w_train[:50],
            eval_set=[(X_val.iloc[:50], y_val.iloc[:50])],
            sample_weight_eval_set=[w_val[:50]],
            verbose=False
        )
        print("      [THÀNH CÔNG] Đã kích hoạt tăng tốc phần cứng GPU CUDA cho XGBoost!")
        model = xgb.XGBRegressor(**gpu_params)
    except Exception as e:
        print(f"      [LƯU Ý] Không thể khởi tạo GPU CUDA ({e}).")
        print("      [CHUYỂN ĐỔI] Chuyển đổi sang chế độ đa luồng CPU (n_jobs=-1)...")
        cpu_params = base_params.copy()
        cpu_params["device"] = "cpu"
        cpu_params["n_jobs"] = -1
        model = xgb.XGBRegressor(**cpu_params)
        used_device = "cpu"

    # 4. Huấn luyện mô hình với Early Stopping
    print(f"\n[4/4] Bắt đầu huấn luyện XGBoost trên {used_device.upper()} (Early Stopping 50 rounds)...")
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        sample_weight_eval_set=[w_train, w_val],
        verbose=100
    )

    best_iter = getattr(model, "best_iteration", model.n_estimators)
    print(f"\n[HOÀN THÀNH] Huấn luyện dừng tại iteration tối ưu: {best_iter}")

    # Đánh giá nhanh trên tập Validation
    val_preds = model.predict(X_val)
    val_metrics = evaluate_all(y_val, val_preds, data["val_df"]['IsHoliday'])
    print("\n" + format_metric_report("XGBoost (Validation Set - Sơ bộ)", val_metrics))

    # Lưu checkpoint mô hình
    checkpoint_file = checkpoint_dir / "xgboost_model.pkl"
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

    # Lưu mô hình định dạng JSON chuẩn của XGBoost
    try:
        json_file = checkpoint_dir / "xgboost_model.json"
        model.save_model(str(json_file))
        print(f"[LƯU JSON MODEL] Đã lưu mô hình XGBoost JSON tại: {json_file}")
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể lưu JSON model: {e}")

    return model, val_metrics


if __name__ == "__main__":
    train_xgboost()
