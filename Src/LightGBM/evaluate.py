import sys
from pathlib import Path
import joblib
import numpy as np

# Thêm BASE_DIR vào sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from train import load_data, split_time_series
from Config import load_config, get_path
from Metric.metrics import evaluate_all, format_metric_report

def evaluate_lightgbm():
    """
    Đánh giá mô hình riêng trên validation Q4/2011 và holdout năm 2012.
    """
    config = load_config()
    checkpoint_dir = get_path(config["paths"]["models"]["lightgbm"]["checkpoint_dir"])
    result_dir = get_path(config["paths"]["models"]["lightgbm"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. Load dữ liệu
    # ============================================================

    print("[1/5] Loading preprocessed data...")

    df = load_data()

    print(f"Dataset shape: {df.shape}")

    # ============================================================
    # 2. Split Train / Validation / Holdout
    # ============================================================

    print("[2/5] Splitting validation set...")

    _, val_df, test_df = split_time_series(df)

    print(f"Validation shape: {val_df.shape}")
    print(f"Holdout shape: {test_df.shape}")

    # ============================================================
    # 3. Load model
    # ============================================================

    print("[3/5] Loading LightGBM model...")

    model_path = checkpoint_dir / "lightgbm_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy LightGBM checkpoint: {model_path}"
        )

    model = joblib.load(model_path)

    print(f"Model loaded from: {model_path}")

    # ============================================================
    # 4. Prepare validation and holdout features
    # ============================================================

    print("[4/5] Preparing validation data...")

    target = "Weekly_Sales"

    drop_columns = [
        "Weekly_Sales",
        "Date"
    ]

    X_val = val_df.drop(columns=drop_columns)
    y_val = val_df[target].to_numpy()
    val_holiday = val_df["IsHoliday"].to_numpy()

    X_test = test_df.drop(columns=drop_columns)
    y_test = test_df[target].to_numpy()
    test_holiday = test_df["IsHoliday"].to_numpy()

    # ============================================================
    # 5. Prediction
    # ============================================================

    print("[5/5] Predicting...")

    y_pred = np.asarray(model.predict(X_val))
    test_pred = np.asarray(model.predict(X_test))

    # ============================================================
    # Evaluate metrics
    # ============================================================

    val_metrics = evaluate_all(
        y_true=y_val,
        y_pred=y_pred,
        is_holiday=val_holiday
    )
    test_metrics = evaluate_all(
        y_true=y_test,
        y_pred=test_pred,
        is_holiday=test_holiday
    )

    best_iteration = getattr(model, "best_iteration_", None) or model.n_estimators
    categorical_features = ["Store", "Dept", "Type"]
    report = "\n".join([
        "=" * 80,
        " BÁO CÁO TOÀN DIỆN HIỆU NĂNG MÔ HÌNH LIGHTGBM (WALMART STORE SALES FORECASTING)",
        "=" * 80,
        "\n- Cấu hình thiết bị huấn luyện : CPU",
        f"- Điểm dừng tối ưu (Best Iter) : {best_iteration}",
        f"- Tổng số đặc trưng đầu vào   : {X_val.shape[1]}",
        f"- Đặc trưng phân loại (Cats)   : {categorical_features}",
        "\n" + format_metric_report(
            "LightGBM - TẬP KIỂM ĐỊNH (VALIDATION: 10/2011 - 12/2011)",
            val_metrics
        ),
        format_metric_report(
            "LightGBM - TẬP KIỂM THỬ NGOÀI MẪU (TEST: NĂM 2012)",
            test_metrics
        ),
        "=" * 80,
        " BẢNG TỔNG HỢP SO SÁNH GIỮA CÁC TẬP KIỂM ĐỊNH",
        "=" * 80,
        f"{'Tập Đánh Giá':<38} | {'WMAE (USD)':<14} | {'MAE (USD)':<12} | {'RMSE (USD)':<12} | {'R2':<8}",
        "-" * 94,
        f"{'Validation Set (Q4/2011 Holiday)':<38} | {val_metrics['WMAE']:<14,.2f} | {val_metrics['MAE']:<12,.2f} | {val_metrics['RMSE']:<12,.2f} | {val_metrics['R2']:<8.4f}",
        f"{'Holdout Test Set (2012 Out-of-sample)':<38} | {test_metrics['WMAE']:<14,.2f} | {test_metrics['MAE']:<12,.2f} | {test_metrics['RMSE']:<12,.2f} | {test_metrics['R2']:<8.4f}",
        "=" * 80,
    ])

    # ============================================================
    # Save report
    # ============================================================

    output_file = result_dir / "lightgbm_results.txt"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(f"Validation samples: {len(y_val):,}\n")
        f.write(f"Holdout samples: {len(y_test):,}\n")
        f.write(f"Model: {model_path}\n\n")
        f.write(report)

    # ============================================================
    # Print result
    # ============================================================

    print()
    print(report)

    print(
        f"Results saved to: {output_file}"
    )

if __name__ == "__main__":
    evaluate_lightgbm()
