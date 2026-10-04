import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from lightgbm import LGBMRegressor

CURRENT_DIR = Path(__file__).resolve().parent
IMPROVE_DIR = CURRENT_DIR.parent
BASE_DIR = IMPROVE_DIR.parent
sys.path.append(str(BASE_DIR))

from Metric.metrics import calculate_wmae

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_tuning(X_train, y_train, train_w, X_val, y_val, val_w, val_holiday, n_trials: int = 10) -> dict:
    """
    Tự động dò tìm siêu tham số tối ưu bằng Optuna (TPESampler) trực tiếp theo độ đo WMAE:
    - learning_rate
    - num_leaves
    - min_child_samples
    - subsample
    - colsample_bytree
    - reg_alpha
    - reg_lambda
    """
    checkpoint_dir = CURRENT_DIR / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Optuna Tuning] Bắt đầu quét {n_trials} trials tối ưu siêu tham số LightGBM theo WMAE...")

    def objective(trial):
        params = {
            "objective": "regression_l1",
            "metric": "l1",
            "n_estimators": 1200,
            "random_state": 42,
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "subsample_freq": 1
        }

        model = LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            sample_weight=train_w,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[val_w],
            categorical_feature=["Store", "Dept", "Type"],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
        )

        val_preds = np.clip(model.predict(X_val), 0.0, None)
        score = calculate_wmae(y_val, val_preds, val_holiday)
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_value = study.best_value
    print(f"[Optuna Tuning] Hoàn tất! Best Trial WMAE: {best_value:,.2f} USD")
    print(f"      Best Params: {best_params}")

    # Lưu best params vào file JSON
    params_file = checkpoint_dir / "best_params.json"
    with open(params_file, "w", encoding="utf-8") as f:
        json.dump({"best_val_wmae": best_value, "params": best_params}, f, indent=4)
    print(f"      Đã lưu thông số tối ưu vào: {params_file}")

    return best_params
