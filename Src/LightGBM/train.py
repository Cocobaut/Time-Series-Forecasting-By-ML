import sys
from pathlib import Path
import pandas as pd
import joblib
import lightgbm as lgb
from lightgbm import LGBMRegressor

# Thêm BASE_DIR vào sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from Config import load_config, get_path

def load_data() -> pd.DataFrame:
    """
    Tải dữ liệu đã tiền xử lý từ config.
    """
    config = load_config()
    data_path = get_path(config["paths"]["files"]["processed_train_parquet"])
    df = pd.read_parquet(data_path)

    return df
def split_time_series(df: pd.DataFrame):
    """
    Chia dữ liệu thành train, validation Q4/2011 và holdout từ năm 2012.
    """
    config = load_config()
    validation_start = pd.Timestamp(config["data"]["validation_start"])
    test_start = pd.Timestamp(config["data"]["test_start"])

    if validation_start >= test_start:
        raise ValueError("validation_start phải nằm trước test_start.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(
        ["Date", "Store", "Dept"]
    ).reset_index(drop=True)

    train_df = df[df["Date"] < validation_start].copy()
    val_df = df[
        (df["Date"] >= validation_start) & (df["Date"] < test_start)
    ].copy()
    test_df = df[df["Date"] >= test_start].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Train, validation hoặc holdout rỗng; kiểm tra ngày trong config và dữ liệu."
        )

    print("\n===== TIME SERIES SPLIT =====")
    print(f"Validation start: {validation_start}")
    print(f"Holdout start: {test_start}")
    print(
        f"Train period: "
        f"{train_df['Date'].min()} -> "
        f"{train_df['Date'].max()}"
    )
    print(
        f"Validation period: "
        f"{val_df['Date'].min()} -> "
        f"{val_df['Date'].max()}"
    )
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Holdout shape: {test_df.shape}")

    return train_df, val_df, test_df

def train_lightgbm():
    """
    Cấu hình siêu tham số, thiết lập sample_weight cho WMAE, huấn luyện LightGBM và lưu checkpoint.
    """
    config = load_config()
    checkpoint_dir = get_path(config["paths"]["models"]["lightgbm"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Huấn luyện LGBMRegressor với early stopping và lưu model vào checkpoint_dir
    df = load_data()

    # Split theo thời gian
    train_df, val_df, _ = split_time_series(df)

    # Target
    target = "Weekly_Sales"

    # Features
    drop_columns = [
        "Weekly_Sales",
        "Date"
    ]

    X_train = train_df.drop(
        columns=drop_columns
    )
    y_train = train_df[target]

    X_val = val_df.drop(
        columns=drop_columns
    )
    y_val = val_df[target]

    # WMAE weights
    train_weight = train_df["IsHoliday"].map(
        {0: 1.0, 1: 5.0}
    ).to_numpy()

    val_weight = val_df["IsHoliday"].map(
        {0: 1.0, 1: 5.0}
    ).to_numpy()

    # Model
    model = LGBMRegressor(
        objective="regression",
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(
        X_train,
        y_train,
        sample_weight=train_weight,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[val_weight],
        categorical_feature=["Store", "Dept", "Type"],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )

    # Save
    model_path = checkpoint_dir / "lightgbm_model.pkl"

    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    train_lightgbm()
