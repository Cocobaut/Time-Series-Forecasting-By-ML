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
from Src.CatBoost.train import load_data, split_time_series


def evaluate_catboost():
    """
    Nạp mô hình CatBoost từ checkpoint, dự báo trên tập Validation và Out-of-sample Test,
    tính toán các thang đo chuẩn (WMAE, MAE, RMSE, MAPE, R2) và xuất báo cáo vào file txt.
    Đồng thời tạo file submission nộp bài cho Kaggle Test nếu có.
    """
    print("=" * 80)
    print(" BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH CATBOOST (METRIC EVALUATION & REPORTING)")
    print("=" * 80)

    config = load_config()
    checkpoint_dir = get_path(config["paths"]["models"]["catboost"]["checkpoint_dir"])
    result_dir = get_path(config["paths"]["models"]["catboost"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / "catboost_model.pkl"
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint mô hình tại: {checkpoint_file}."
            f" Vui lòng chạy train.py trước để huấn luyện mô hình!"
        )

    # 1. Nạp mô hình từ checkpoint
    print(f"[1/4] Đang nạp mô hình từ checkpoint: {checkpoint_file.name}...")
    saved_data = joblib.load(checkpoint_file)
    model = saved_data["model"]
    feature_cols = saved_data["feature_cols"]
    cat_cols = saved_data.get("cat_cols", [])
    best_iter = saved_data.get("best_iteration", "N/A")
    device_used = saved_data.get("device", "N/A")

    print(f"      + Thiết bị huấn luyện trước đó: {device_used}")
    print(f"      + Best iteration: {best_iter}")
    print(f"      + Số lượng features: {len(feature_cols)}")

    # 2. Nạp dữ liệu và phân chia
    print("\n[2/4] Nạp và phân chia dữ liệu đánh giá...")
    df = load_data()
    data = split_time_series(df)

    X_val, y_val = data["X_val"], data["y_val"]
    val_df = data["val_df"]

    X_test, y_test = data["X_test"], data["test_df"]['Weekly_Sales']
    test_df = data["test_df"]

    # 3. Tạo Pool và tính toán metric
    print("\n[3/4] Đang dự báo và tính toán các thang đo chuẩn...")
    val_pool = cb.Pool(X_val, cat_features=cat_cols)
    test_pool = cb.Pool(X_test, cat_features=cat_cols)

    val_preds = model.predict(val_pool)
    val_metrics = evaluate_all(y_val, val_preds, val_df['IsHoliday'])

    test_preds = model.predict(test_pool)
    test_metrics = evaluate_all(y_test, test_preds, test_df['IsHoliday'])

    # Định dạng báo cáo văn bản
    val_report = format_metric_report("CATBOOST - TẬP KIỂM ĐỊNH (VALIDATION: 10/2011 - 12/2011)", val_metrics)
    test_report = format_metric_report("CATBOOST - TẬP KIỂM THỬ NGOÀI MẪU (TEST: NĂM 2012)", test_metrics)

    print("\n" + val_report)
    print(test_report)

    # 4. Ghi toàn bộ kết quả vào file txt
    report_file = result_dir / "catboost_metrics.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write(" BÁO CÁO TOÀN DIỆN HIỆU NĂNG MÔ HÌNH CATBOOST (WALMART STORE SALES FORECASTING)\n")
        f.write("================================================================================\n\n")
        f.write(f"- Cấu hình thiết bị huấn luyện : {device_used}\n")
        f.write(f"- Điểm dừng tối ưu (Best Iter) : {best_iter}\n")
        f.write(f"- Tổng số đặc trưng đầu vào   : {len(feature_cols)}\n")
        f.write(f"- Đặc trưng phân loại (Cats)   : {cat_cols}\n\n")

        f.write(val_report + "\n")
        f.write(test_report + "\n")

        # Bảng so sánh nhanh
        f.write("================================================================================\n")
        f.write(" BẢNG TỔNG HỢP SO SÁNH GIỮA CÁC TẬP KIỂM ĐỊNH\n")
        f.write("================================================================================\n")
        f.write(f"{'Tập Đánh Giá':<35} | {'WMAE (USD)':<15} | {'MAE (USD)':<12} | {'RMSE (USD)':<12} | {'R2':<8}\n")
        f.write("-" * 88 + "\n")
        f.write(f"{'Validation Set (Q4/2011 Holiday)':<35} | {val_metrics['WMAE']:<15,.2f} | {val_metrics['MAE']:<12,.2f} | {val_metrics['RMSE']:<12,.2f} | {val_metrics['R2']:<8.4f}\n")
        f.write(f"{'Holdout Test Set (2012 Out-of-sample)':<35} | {test_metrics['WMAE']:<15,.2f} | {test_metrics['MAE']:<12,.2f} | {test_metrics['RMSE']:<12,.2f} | {test_metrics['R2']:<8.4f}\n")
        f.write("================================================================================\n")

    print(f"[4/4] Đã lưu báo cáo đánh giá chi tiết vào: {report_file}")

    # 5. Dự báo trên tập Test Kaggle (nếu có) để tạo file Submission
    kaggle_test_parquet = get_path(config["paths"]["files"]["processed_test_parquet"])
    if kaggle_test_parquet.exists():
        try:
            print("\n[BỔ SUNG] Phát hiện tập Test Kaggle. Đang tạo file submission...")
            df_kaggle_test = pd.read_parquet(kaggle_test_parquet)
            df_kaggle_test['Date'] = pd.to_datetime(df_kaggle_test['Date'])

            for c in cat_cols:
                if c in df_kaggle_test.columns:
                    df_kaggle_test[c] = df_kaggle_test[c].astype(int)

            df_kaggle_test = df_kaggle_test.replace([np.inf, -np.inf], np.nan)
            num_cols = df_kaggle_test.select_dtypes(include=np.number).columns
            df_kaggle_test[num_cols] = df_kaggle_test[num_cols].fillna(0.0)

            # Đảm bảo thứ tự cột đặc trưng khớp 100% với lúc train
            X_kaggle_test = df_kaggle_test[feature_cols]
            kaggle_pool = cb.Pool(X_kaggle_test, cat_features=cat_cols)
            kaggle_preds = model.predict(kaggle_pool)

            df_kaggle_test['Weekly_Sales'] = kaggle_preds
            df_kaggle_test['Id'] = (
                df_kaggle_test['Store'].astype(str) + "_" +
                df_kaggle_test['Dept'].astype(str) + "_" +
                df_kaggle_test['Date'].dt.strftime('%Y-%m-%d')
            )

            submission_path = result_dir / "submission_catboost.csv"
            df_kaggle_test[['Id', 'Weekly_Sales']].to_csv(submission_path, index=False)
            print(f"         [OK] Đã lưu submission Kaggle tại: {submission_path} ({len(df_kaggle_test):,} dòng)")
        except Exception as e:
            print(f"         [LƯU Ý] Không tạo được submission Kaggle: {e}")

    print("\n" + "=" * 80)
    print(" [HOÀN TẤT] Quy trình đánh giá CatBoost thành công rực rỡ!")
    print("=" * 80)

    return val_metrics, test_metrics


if __name__ == "__main__":
    evaluate_catboost()
