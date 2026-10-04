import sys
from pathlib import Path
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
IMPROVE_DIR = CURRENT_DIR.parent
BASE_DIR = IMPROVE_DIR.parent
sys.path.append(str(BASE_DIR))


def build_p3_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Tạo các đặc trưng nâng cao giai đoạn P3:
    1. Thống kê nhóm lịch sử theo cặp (Store, Dept) được tính HOÀN TOÀN từ tập Train (chống rò rỉ tương lai).
    2. Tỷ lệ tương quan chu kỳ năm: sales_ratio_to_store_dept_mean = sales_lag_52 / store_dept_mean.
    3. Cờ nhị phân nhận biết giai đoạn phát sinh dữ liệu khuyến mãi: is_markdown_available.
    4. Mật độ khuyến mãi theo quy mô cửa hàng: markdown_to_size_ratio = Total_MarkDown / Size.
    """
    print("\n[P3 Feature Engineering] Tính toán thống kê nhóm lịch sử (Store, Dept) từ tập Train...")

    # 1. Thống kê theo cặp (Store, Dept) từ Train
    store_dept_stats = train_df.groupby(["Store", "Dept"])["Weekly_Sales"].agg(
        store_dept_mean="mean",
        store_dept_median="median",
        store_dept_std="std",
        store_dept_max="max",
        store_dept_min="min"
    ).reset_index()

    # Thống kê fallback theo Dept và Store để bù các cặp chưa từng xuất hiện
    dept_stats = train_df.groupby("Dept")["Weekly_Sales"].agg(dept_mean="mean").reset_index()
    overall_mean = float(train_df["Weekly_Sales"].mean())

    def transform_split(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()

        # Merge thống kê cặp
        df_out = df_out.merge(store_dept_stats, on=["Store", "Dept"], how="left")
        df_out = df_out.merge(dept_stats, on=["Dept"], how="left")

        # Xử lý missing cho các cặp mới
        df_out["store_dept_mean"] = df_out["store_dept_mean"].fillna(df_out["dept_mean"]).fillna(overall_mean)
        df_out["store_dept_median"] = df_out["store_dept_median"].fillna(df_out["store_dept_mean"])
        df_out["store_dept_std"] = df_out["store_dept_std"].fillna(0.0)
        df_out["store_dept_max"] = df_out["store_dept_max"].fillna(df_out["store_dept_mean"])
        df_out["store_dept_min"] = df_out["store_dept_min"].fillna(0.0)
        df_out.drop(columns=["dept_mean"], inplace=True)

        # 2. Tỷ lệ doanh số cùng kỳ so với trung bình gian hàng
        df_out["sales_ratio_to_store_dept_mean"] = df_out["sales_lag_52"] / (df_out["store_dept_mean"] + 1.0)

        # 3. Đặc trưng xử lý bất đối xứng dữ liệu Markdown
        df_out["markdown_to_size_ratio"] = df_out["Total_MarkDown"] / (df_out["Size"] + 1.0)
        df_out["is_markdown_available"] = (df_out["Total_MarkDown"] > 0).astype(int)

        # Làm sạch inf / -inf
        num_cols = df_out.select_dtypes(include=np.number).columns
        df_out[num_cols] = df_out[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return df_out

    train_p3 = transform_split(train_df)
    val_p3 = transform_split(val_df)
    test_p3 = transform_split(test_df)

    new_features = [
        "store_dept_mean", "store_dept_median", "store_dept_std", "store_dept_max", "store_dept_min",
        "sales_ratio_to_store_dept_mean", "markdown_to_size_ratio", "is_markdown_available"
    ]
    print(f"      Đã bổ sung thành công {len(new_features)} đặc trưng mới: {new_features}")
    return train_p3, val_p3, test_p3, new_features
