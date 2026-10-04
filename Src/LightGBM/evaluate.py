import sys
from pathlib import Path
import joblib
import numpy as np

# Đảm bảo luồng xuất UTF-8 trên Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thêm BASE_DIR vào sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from train import load_data, split_time_series
from Config import load_config, get_path
from Metric.metrics import evaluate_all, format_metric_report


def format_txt_report(title: str, metrics: dict) -> str:
    """Định dạng báo cáo đánh giá thành tiếng Việt chữ thường (chỉ in hoa chữ cái đầu) cho file txt."""
    report = [
        f"--------------------------------------------------",
        f" Báo cáo đánh giá mô hình: {title}",
        f"--------------------------------------------------",
        f" 1. WMAE (Trọng số ngày lễ x5) : {metrics['WMAE']:,.2f} USD",
        f" 2. MAE (Sai số tuyệt đối trung bình) : {metrics['MAE']:,.2f} USD",
        f" 3. RMSE (Căn bậc hai sai số toàn phương) : {metrics['RMSE']:,.2f} USD",
        f" 4. MAPE (Sai số phần trăm trung bình) : {metrics['MAPE (%)']:.2f} %",
        f" 5. R2 Score (Hệ số xác định) : {metrics['R2']:.4f}",
        f"--------------------------------------------------"
    ]
    return "\n".join(report)


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
        " Báo cáo toàn diện hiệu năng mô hình LIGHTGBM (Walmart Store Sales Forecasting)",
        "=" * 80,
        "\n- Cấu hình thiết bị huấn luyện : CPU",
        f"- Điểm dừng tối ưu (Best Iter) : {best_iteration}",
        f"- Tổng số đặc trưng đầu vào   : {X_val.shape[1]}",
        f"- Đặc trưng phân loại (Cats)   : {categorical_features}",
        "\n" + format_metric_report(
            "LightGBM - Tập kiểm định (Validation: 10/2011 - 12/2011)",
            val_metrics
        ),
        format_metric_report(
            "LightGBM - Tập kiểm thử ngoàn mẫu (Test: NĂM 2012)",
            test_metrics
        ),
        "=" * 80,
        " Bảng tổng hợp so sánh giữa các tập kiểm định",
        "=" * 80,
        f"{'Tập Đánh Giá':<38} | {'WMAE (USD)':<14} | {'MAE (USD)':<12} | {'RMSE (USD)':<12} | {'R2':<8}",
        "-" * 94,
        f"{'Validation Set (Q4/2011 Holiday)':<38} | {val_metrics['WMAE']:<14,.2f} | {val_metrics['MAE']:<12,.2f} | {val_metrics['RMSE']:<12,.2f} | {val_metrics['R2']:<8.4f}",
        f"{'Holdout Test Set (2012 Out-of-sample)':<38} | {test_metrics['WMAE']:<14,.2f} | {test_metrics['MAE']:<12,.2f} | {test_metrics['RMSE']:<12,.2f} | {test_metrics['R2']:<8.4f}",
        "=" * 80,
    ])

    # ============================================================
    # Save report to txt (tiếng Việt in thường, chỉ in hoa chữ cái đầu)
    # ============================================================

    val_txt_report = format_txt_report("LightGBM - Tập kiểm định (Validation: 10/2011 - 12/2011)", val_metrics)
    test_txt_report = format_txt_report("LightGBM - Tập kiểm thử ngoài mẫu (Test: Năm 2012)", test_metrics)

    output_file = result_dir / "lightgbm_results.txt"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        f.write("================================================================================\n")
        f.write(" Báo cáo toàn diện hiệu năng mô hình LightGBM (Walmart Store Sales Forecasting)\n")
        f.write("================================================================================\n\n")
        f.write("- Cấu hình thiết bị huấn luyện : CPU\n")
        f.write(f"- Điểm dừng tối ưu (Best iteration) : {best_iteration}\n")
        f.write(f"- Tổng số đặc trưng đầu vào   : {X_val.shape[1]}\n")
        f.write(f"- Đặc trưng phân loại (Categorical) : {categorical_features}\n\n")

        f.write(val_txt_report + "\n\n")
        f.write(test_txt_report + "\n\n")

        # Bảng tổng hợp so sánh giữa các tập kiểm định
        f.write("================================================================================\n")
        f.write(" Bảng tổng hợp so sánh giữa các tập kiểm định\n")
        f.write("================================================================================\n")
        f.write(f"{'Tập đánh giá':<35} | {'WMAE (USD)':<15} | {'MAE (USD)':<12} | {'RMSE (USD)':<12} | {'R2':<8}\n")
        f.write("-" * 88 + "\n")
        f.write(f"{'Tập kiểm định (Q4/2011 Holiday)':<35} | {val_metrics['WMAE']:<15,.2f} | {val_metrics['MAE']:<12,.2f} | {val_metrics['RMSE']:<12,.2f} | {val_metrics['R2']:<8.4f}\n")
        f.write(f"{'Tập kiểm thử (2012 Out-of-sample)':<35} | {test_metrics['WMAE']:<15,.2f} | {test_metrics['MAE']:<12,.2f} | {test_metrics['RMSE']:<12,.2f} | {test_metrics['R2']:<8.4f}\n")
        f.write("================================================================================\n")

    # ============================================================
    # Print result
    # ============================================================

    print()
    print(report)

    print(
        f"Results saved to: {output_file}"
    )

    return val_metrics, test_metrics


if __name__ == "__main__":
    evaluate_lightgbm()
