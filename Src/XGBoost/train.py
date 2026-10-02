import sys
from pathlib import Path
import warnings
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

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


def load_data() -> pd.DataFrame:
    """
    Load preprocessed data from config (Parquet preferred, CSV fallback).
    Clean infinite values (inf, -inf) and NaNs for GPU DMatrix compatibility.
    """
    config = load_config()
    parquet_path = get_path(config["paths"]["files"]["processed_train_parquet"])
    csv_path = get_path(config["paths"]["files"]["processed_train_csv"])

    if parquet_path.exists():
        print(f"[1/4] Loading preprocessed data from Parquet: {parquet_path.name}...")
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        print(f"[1/4] Loading preprocessed data from CSV: {csv_path.name}...")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Preprocessed data file not found at {parquet_path} or {csv_path}")

    df['Date'] = pd.to_datetime(df['Date'])

    # Clean infinite values and NaNs from ratio features
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(0.0)

    print(f"      Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")
    return df


def split_time_series(df: pd.DataFrame):
    """
    Split time series data to prevent lookahead bias:
      - Train set: Before 2011-10-01 (learn historical patterns)
      - Validation set: From 2011-10-01 to 2011-12-31 (holiday shopping season)
      - Out-of-sample Test set: From 2012-01-01 onwards (future evaluation)
    """
    print("[2/4] Splitting time series data (time-based split)...")

    # Define feature columns
    exclude_cols = ['Date', 'Weekly_Sales']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    cat_cols = [c for c in ['Store', 'Dept', 'Type'] if c in feature_cols]

    # Convert categorical columns to 'category' dtype for native XGBoost handling
    for c in cat_cols:
        df[c] = df[c].astype('category')

    # Time-based splitting
    train_mask = df['Date'] < '2011-10-01'
    val_mask = (df['Date'] >= '2011-10-01') & (df['Date'] <= '2011-12-31')
    test_mask = df['Date'] >= '2012-01-01'

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    X_train, y_train = train_df[feature_cols], train_df['Weekly_Sales']
    X_val, y_val = val_df[feature_cols], val_df['Weekly_Sales']
    X_test, y_test = test_df[feature_cols], test_df['Weekly_Sales']

    # Sample weights: x5 for holiday weeks, x1 for regular weeks
    w_train = np.where(train_df['IsHoliday'] == 1, 5.0, 1.0)
    w_val = np.where(val_df['IsHoliday'] == 1, 5.0, 1.0)
    w_test = np.where(test_df['IsHoliday'] == 1, 5.0, 1.0)

    print(f"      + Train set:      {len(train_df):,} rows (before 2011-10-01)")
    print(f"      + Validation set: {len(val_df):,} rows (2011-10-01 to 2011-12-31)")
    print(f"      + Holdout test:   {len(test_df):,} rows (from 2012-01-01 onwards)")
    print(f"      + Feature count:  {len(feature_cols)} features ({cat_cols} categorical)")

    return {
        "X_train": X_train, "y_train": y_train, "w_train": w_train, "val_df": val_df,
        "X_val": X_val, "y_val": y_val, "w_val": w_val, "test_df": test_df,
        "X_test": X_test, "y_test": y_test, "w_test": w_test,
        "feature_cols": feature_cols, "cat_cols": cat_cols
    }


def train_xgboost():
    """
    Configure hyperparameters, enable hardware GPU CUDA acceleration for XGBoost,
    optimize L1 loss with sample weights for WMAE, train with early stopping and save checkpoint.
    """
    print("=" * 80)
    print(" Starting XGBoost model training (Time series forecasting with GPU CUDA)")
    print("=" * 80)

    config = load_config()
    checkpoint_dir = get_path(config["paths"]["models"]["xgboost"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and split data
    df = load_data()
    data = split_time_series(df)

    X_train, y_train, w_train = data["X_train"], data["y_train"], data["w_train"]
    X_val, y_val, w_val = data["X_val"], data["y_val"], data["w_val"]
    cat_cols = data["cat_cols"]
    feature_cols = data["feature_cols"]

    # 2. Hyperparameter configuration for XGBoost with GPU CUDA
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

    # 3. Model initialization with GPU CUDA
    print("\n[3/4] Configuring training device...")
    used_device = "cuda"
    try:
        print("      Initializing XGBoost with GPU CUDA (NVIDIA GeForce RTX)...")
        gpu_params = base_params.copy()
        gpu_params["device"] = "cuda"
        model = xgb.XGBRegressor(**gpu_params)

        # Quick verification of GPU CUDA readiness
        _ = model.fit(
            X_train.iloc[:50], y_train.iloc[:50],
            sample_weight=w_train[:50],
            eval_set=[(X_val.iloc[:50], y_val.iloc[:50])],
            sample_weight_eval_set=[w_val[:50]],
            verbose=False
        )
        print("      [SUCCESS] Enabled hardware GPU CUDA acceleration for XGBoost!")
        model = xgb.XGBRegressor(**gpu_params)
    except Exception as e:
        print(f"      [NOTE] Could not initialize GPU CUDA ({e}).")
        print("      [SWITCH] Falling back to multi-threaded CPU mode (n_jobs=-1)...")
        cpu_params = base_params.copy()
        cpu_params["device"] = "cpu"
        cpu_params["n_jobs"] = -1
        model = xgb.XGBRegressor(**cpu_params)
        used_device = "cpu"

    # 4. Train model with early stopping
    print(f"\n[4/4] Starting XGBoost training on {used_device.upper()} (Early stopping 50 rounds)...")
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        sample_weight_eval_set=[w_train, w_val],
        verbose=100
    )

    best_iter = getattr(model, "best_iteration", model.n_estimators)
    print(f"\n[COMPLETED] Training stopped at optimal iteration: {best_iter}")

    # Preliminary evaluation on Validation set
    val_preds = model.predict(X_val)
    val_metrics = evaluate_all(y_val, val_preds, data["val_df"]['IsHoliday'])
    print("\n" + format_terminal_report("XGBoost (Validation set - Preliminary)", val_metrics))

    # Save model checkpoint
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
    print(f"[CHECKPOINT] Model and metadata saved to: {checkpoint_file}")

    # Save native XGBoost JSON model
    try:
        json_file = checkpoint_dir / "xgboost_model.json"
        model.save_model(str(json_file))
        print(f"[JSON MODEL] Saved XGBoost JSON model to: {json_file}")
    except Exception as e:
        print(f"[WARNING] Could not save JSON model: {e}")

    return model, val_metrics


if __name__ == "__main__":
    train_xgboost()
