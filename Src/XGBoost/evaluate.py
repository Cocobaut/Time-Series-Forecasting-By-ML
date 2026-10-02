import sys
from pathlib import Path
import warnings
import joblib
import pandas as pd
import numpy as np

# Ensure UTF-8 output stream on Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# Add BASE_DIR to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from Config import load_config, get_path
from Metric.metrics import evaluate_all
from Src.XGBoost.train import load_data, split_time_series


def format_terminal_report(title: str, metrics: dict) -> str:
    """Format metric evaluation into clean English report for terminal output."""
    report = [
        f"--------------------------------------------------",
        f" Evaluation report: {title}",
        f"--------------------------------------------------",
        f" 1. WMAE (Holiday weight x5) : {metrics['WMAE']:,.2f} USD",
        f" 2. MAE (Mean absolute error): {metrics['MAE']:,.2f} USD",
        f" 3. RMSE (Root mean squared) : {metrics['RMSE']:,.2f} USD",
        f" 4. MAPE (Mean abs pct err)  : {metrics['MAPE (%)']:.2f} %",
        f" 5. R2 Score (Determination) : {metrics['R2']:.4f}",
        f"--------------------------------------------------"
    ]
    return "\n".join(report)


def format_txt_report(title: str, metrics: dict) -> str:
    """Format metric evaluation into clean Vietnamese report for txt file (no all-caps)."""
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


def evaluate_xgboost():
    """
    Load XGBoost model from checkpoint, evaluate on Validation and Out-of-sample Test sets,
    calculate metrics (WMAE, MAE, RMSE, MAPE, R2) and save report to txt.
    Optionally generate submission file for Kaggle test data.
    """
    print("=" * 80)
    print(" Starting XGBoost model evaluation (Metric evaluation & reporting)")
    print("=" * 80)

    config = load_config()
    checkpoint_dir = get_path(config["paths"]["models"]["xgboost"]["checkpoint_dir"])
    result_dir = get_path(config["paths"]["models"]["xgboost"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / "xgboost_model.pkl"
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint model not found at: {checkpoint_file}."
            f" Please run train.py first to train the model!"
        )

    # 1. Load model from checkpoint
    print(f"[1/4] Loading model from checkpoint: {checkpoint_file.name}...")
    saved_data = joblib.load(checkpoint_file)
    model = saved_data["model"]
    feature_cols = saved_data["feature_cols"]
    cat_cols = saved_data.get("cat_cols", [])
    best_iter = saved_data.get("best_iteration", "N/A")
    device_used = saved_data.get("device", "N/A")

    print(f"      + Training device: {device_used}")
    print(f"      + Best iteration: {best_iter}")
    print(f"      + Number of features: {len(feature_cols)}")

    # 2. Load and split evaluation data
    print("\n[2/4] Loading and splitting evaluation data...")
    df = load_data()
    data = split_time_series(df)

    X_val, y_val = data["X_val"], data["y_val"]
    val_df = data["val_df"]

    X_test, y_test = data["X_test"], data["test_df"]['Weekly_Sales']
    test_df = data["test_df"]

    # 3. Predict and compute metrics
    print("\n[3/4] Predicting and calculating evaluation metrics...")
    val_preds = model.predict(X_val)
    val_metrics = evaluate_all(y_val, val_preds, val_df['IsHoliday'])

    test_preds = model.predict(X_test)
    test_metrics = evaluate_all(y_test, test_preds, test_df['IsHoliday'])

    # Print clean English reports to terminal
    print("\n" + format_terminal_report("XGBoost - Validation set (10/2011 - 12/2011)", val_metrics))
    print(format_terminal_report("XGBoost - Holdout test set (Year 2012)", test_metrics))

    # 4. Write results to txt file in Vietnamese (no all-caps)
    val_txt_report = format_txt_report("XGBoost - Tập kiểm định (Validation: 10/2011 - 12/2011)", val_metrics)
    test_txt_report = format_txt_report("XGBoost - Tập kiểm thử ngoài mẫu (Test: Năm 2012)", test_metrics)

    report_file = result_dir / "xgboost_metrics.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write(" Báo cáo toàn diện hiệu năng mô hình XGBoost (Walmart Store Sales Forecasting)\n")
        f.write("================================================================================\n\n")
        f.write(f"- Cấu hình thiết bị huấn luyện : {device_used}\n")
        f.write(f"- Điểm dừng tối ưu (Best iteration) : {best_iter}\n")
        f.write(f"- Tổng số đặc trưng đầu vào   : {len(feature_cols)}\n")
        f.write(f"- Đặc trưng phân loại (Categorical) : {cat_cols}\n\n")

        f.write(val_txt_report + "\n\n")
        f.write(test_txt_report + "\n\n")

        # Summary comparison table in Vietnamese
        f.write("================================================================================\n")
        f.write(" Bảng tổng hợp so sánh giữa các tập kiểm định\n")
        f.write("================================================================================\n")
        f.write(f"{'Tập đánh giá':<35} | {'WMAE (USD)':<15} | {'MAE (USD)':<12} | {'RMSE (USD)':<12} | {'R2':<8}\n")
        f.write("-" * 88 + "\n")
        f.write(f"{'Tập kiểm định (Q4/2011 Holiday)':<35} | {val_metrics['WMAE']:<15,.2f} | {val_metrics['MAE']:<12,.2f} | {val_metrics['RMSE']:<12,.2f} | {val_metrics['R2']:<8.4f}\n")
        f.write(f"{'Tập kiểm thử (2012 Out-of-sample)':<35} | {test_metrics['WMAE']:<15,.2f} | {test_metrics['MAE']:<12,.2f} | {test_metrics['RMSE']:<12,.2f} | {test_metrics['R2']:<8.4f}\n")
        f.write("================================================================================\n")

    print(f"[4/4] Saved detailed evaluation report to: {report_file}")

    # 5. Kaggle test predictions if file exists
    kaggle_test_parquet = get_path(config["paths"]["files"]["processed_test_parquet"])
    if kaggle_test_parquet.exists():
        try:
            print("\n[NOTE] Kaggle test set detected. Generating submission file...")
            df_kaggle_test = pd.read_parquet(kaggle_test_parquet)
            df_kaggle_test['Date'] = pd.to_datetime(df_kaggle_test['Date'])

            for c in cat_cols:
                if c in df_kaggle_test.columns:
                    df_kaggle_test[c] = df_kaggle_test[c].astype('category')

            df_kaggle_test = df_kaggle_test.replace([np.inf, -np.inf], np.nan)
            num_cols = df_kaggle_test.select_dtypes(include=np.number).columns
            df_kaggle_test[num_cols] = df_kaggle_test[num_cols].fillna(0.0)

            X_kaggle_test = df_kaggle_test[feature_cols]
            kaggle_preds = model.predict(X_kaggle_test)

            df_kaggle_test['Weekly_Sales'] = kaggle_preds
            df_kaggle_test['Id'] = (
                df_kaggle_test['Store'].astype(str) + "_" +
                df_kaggle_test['Dept'].astype(str) + "_" +
                df_kaggle_test['Date'].dt.strftime('%Y-%m-%d')
            )

            submission_path = result_dir / "submission_xgboost.csv"
            df_kaggle_test[['Id', 'Weekly_Sales']].to_csv(submission_path, index=False)
            print(f"       [OK] Saved Kaggle submission to: {submission_path} ({len(df_kaggle_test):,} rows)")
        except Exception as e:
            print(f"       [NOTE] Could not generate Kaggle submission: {e}")

    print("\n" + "=" * 80)
    print(" [COMPLETED] XGBoost evaluation pipeline finished successfully!")
    print("=" * 80)

    return val_metrics, test_metrics


if __name__ == "__main__":
    evaluate_xgboost()
