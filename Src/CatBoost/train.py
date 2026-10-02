import sys
from pathlib import Path
import warnings
import joblib
import pandas as pd
import numpy as np
import catboost as cb

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
    Clean infinite values (inf, -inf) and NaNs for GPU Pool compatibility.
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

    # Ensure integer type for CatBoost categorical features
    for c in cat_cols:
        df[c] = df[c].astype(int)

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


def train_catboost():
    """
    Configure hyperparameters, set cat_features and sample weights for WMAE,
    train CatBoost with GPU acceleration (CUDA) and save checkpoint.
    """
    print("=" * 80)
    print(" Starting CatBoost model training (Time series forecasting with GPU)")
    print("=" * 80)

    config = load_config()
    checkpoint_dir = get_path(config["paths"]["models"]["catboost"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and split data
    df = load_data()
    data = split_time_series(df)

    X_train, y_train, w_train = data["X_train"], data["y_train"], data["w_train"]
    X_val, y_val, w_val = data["X_val"], data["y_val"], data["w_val"]
    cat_cols = data["cat_cols"]
    feature_cols = data["feature_cols"]

    # 2. Build CatBoost Pool objects
    print("\n[3/4] Preparing CatBoost pool and configuring device...")
    train_pool = cb.Pool(X_train, y_train, weight=w_train, cat_features=cat_cols)
    val_pool = cb.Pool(X_val, y_val, weight=w_val, cat_features=cat_cols)

    # 3. Configure hyperparameters
    # On CatBoost GPU, loss_function="RMSE" with sample weights and eval_metric="MAE" converges cleanly
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
        print("      Initializing CatBoost with GPU (CUDA acceleration)...")
        gpu_params = base_params.copy()
        gpu_params["task_type"] = "GPU"
        model = cb.CatBoostRegressor(**gpu_params)

        # Quick test on small slice
        test_pool = cb.Pool(X_train.iloc[:50], y_train.iloc[:50], weight=w_train[:50], cat_features=cat_cols)
        _ = model.fit(test_pool, eval_set=test_pool, verbose=False)
        print("      [SUCCESS] Enabled hardware GPU acceleration for CatBoost!")
        model = cb.CatBoostRegressor(**gpu_params)
    except Exception as e:
        print(f"      [NOTE] Could not initialize CatBoost on GPU ({e}).")
        print("      [SWITCH] Falling back to multi-threaded CPU mode (thread_count=-1)...")
        cpu_params = base_params.copy()
        cpu_params["task_type"] = "CPU"
        cpu_params["thread_count"] = -1
        model = cb.CatBoostRegressor(**cpu_params)
        used_device = "CPU"

    # 4. Train model with early stopping
    print(f"\n[4/4] Starting CatBoost training on {used_device} (Early stopping 50 rounds)...")
    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=100
    )

    best_iter = model.get_best_iteration()
    print(f"\n[COMPLETED] Training stopped at optimal iteration: {best_iter}")

    # Preliminary evaluation on Validation set
    val_preds = model.predict(val_pool)
    val_metrics = evaluate_all(y_val, val_preds, data["val_df"]['IsHoliday'])
    print("\n" + format_terminal_report("CatBoost (Validation set - Preliminary)", val_metrics))

    # Save model checkpoint
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
    print(f"[CHECKPOINT] Model and metadata saved to: {checkpoint_file}")

    # Save native CatBoost binary model (.cbm)
    try:
        cbm_file = checkpoint_dir / "catboost_model.cbm"
        model.save_model(str(cbm_file))
        print(f"[CBM MODEL] Saved CatBoost binary model to: {cbm_file}")
    except Exception as e:
        print(f"[WARNING] Could not save .cbm file: {e}")

    return model, val_metrics


if __name__ == "__main__":
    train_catboost()
