import sys
import json
from pathlib import Path
import joblib
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor

# Đảm bảo UTF-8 stream trên Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
IMPROVE_DIR = CURRENT_DIR.parent
BASE_DIR = IMPROVE_DIR.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(IMPROVE_DIR))
sys.path.append(str(CURRENT_DIR))

from improve.P1.train_p1_lightgbm import load_data, split_time_series
from features_p3 import build_p3_features
from optuna_tuning import run_optuna_tuning


def train_p3_model(n_trials: int = 10):
    """
    Huấn luyện mô hình P3 hoàn chỉnh:
    1. Tạo bộ đặc trưng P3 (Target Group Statistics + Markdown treatment)
    2. Chạy Optuna tìm siêu tham số tối ưu WMAE
    3. Huấn luyện mô hình LightGBM P3 tối ưu và lưu checkpoint
    """
    checkpoint_dir = CURRENT_DIR / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train_df, val_df, test_df = split_time_series(df)

    train_p3, val_p3, test_p3, new_features = build_p3_features(train_df, val_df, test_df)

    target = "Weekly_Sales"
    drop_columns = ["Weekly_Sales", "Date"]

    X_train = train_p3.drop(columns=drop_columns)
    y_train = train_p3[target]

    X_val = val_p3.drop(columns=drop_columns)
    y_val = val_p3[target]

    X_test = test_p3.drop(columns=drop_columns)
    y_test = test_p3[target]

    train_weight = train_p3["IsHoliday"].map({0: 1.0, 1: 5.0}).to_numpy()
    val_weight = val_p3["IsHoliday"].map({0: 1.0, 1: 5.0}).to_numpy()
    val_holiday = val_p3["IsHoliday"].to_numpy()

    # Dò tìm siêu tham số nếu chưa có checkpoint
    params_file = checkpoint_dir / "best_params.json"
    if params_file.exists():
        print(f"[Optuna] Tìm thấy thông số tối ưu sẵn có: {params_file}")
        with open(params_file, "r", encoding="utf-8") as f:
            best_params = json.load(f)["params"]
    else:
        best_params = run_optuna_tuning(
            X_train, y_train, train_weight,
            X_val, y_val, val_weight, val_holiday,
            n_trials=n_trials
        )

    # Huấn luyện mô hình đầy đủ với Best Params
    print("\n[Huấn luyện P3 Model] Khởi tạo LightGBM với siêu tham số Optuna...")
    final_params = {
        "objective": "regression_l1",
        "metric": "l1",
        "n_estimators": 4000,
        "random_state": 42,
        "n_jobs": -1,
        "subsample_freq": 1,
        **best_params
    }

    model = LGBMRegressor(**final_params)
    model.fit(
        X_train,
        y_train,
        sample_weight=train_weight,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[val_weight],
        categorical_feature=["Store", "Dept", "Type"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=200)
        ]
    )

    best_iter = getattr(model, "best_iteration_", model.n_estimators)
    print(f"\n[HOÀN TẤT] LightGBM P3 tối ưu dừng tại iteration: {best_iter}")

    model_path = checkpoint_dir / "lightgbm_p3_optuna.pkl"
    joblib.dump(model, model_path)
    print(f"[CHECKPOINT] Đã lưu mô hình P3 tại: {model_path}")

    return model, train_p3, val_p3, test_p3


if __name__ == "__main__":
    train_p3_model()
