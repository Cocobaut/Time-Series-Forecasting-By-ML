import sys
from pathlib import Path
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
IMPROVE_DIR = CURRENT_DIR.parent
BASE_DIR = IMPROVE_DIR.parent
sys.path.append(str(BASE_DIR))

from Metric.metrics import calculate_wmae


def identify_surge_departments(train_df: pd.DataFrame, surge_threshold: float = 1.35) -> list:
    """
    Xác định các phòng ban (Dept) có doanh số bùng nổ trong tuần Lễ Tạ Ơn (Tuần 47)
    dựa HOÀN TOÀN trên dữ liệu Train (trước tháng 10/2011) để chống rò rỉ tương lai.
    """
    train_df = train_df.copy()
    weeks = pd.to_datetime(train_df["Date"]).dt.isocalendar().week

    non_holiday_mean = train_df[train_df["IsHoliday"] == 0].groupby("Dept")["Weekly_Sales"].mean()
    thanksgiving_mean = train_df[weeks == 47].groupby("Dept")["Weekly_Sales"].mean()

    surge_depts = []
    for dept in thanksgiving_mean.index:
        if dept in non_holiday_mean and non_holiday_mean[dept] > 0:
            ratio = thanksgiving_mean[dept] / non_holiday_mean[dept]
            if ratio >= surge_threshold:
                surge_depts.append(dept)

    return sorted(surge_depts)


def apply_holiday_multiplier(
    preds: np.ndarray,
    df: pd.DataFrame,
    alpha: float = 1.05,
    surge_depts: list = None,
    target_weeks: list = None
) -> np.ndarray:
    """
    Áp dụng hệ số bù đỉnh ngày lễ (Holiday Post-Processing Multiplier):
    preds_final = preds * alpha đối với các tuần và các phòng ban bùng nổ mục tiêu.
    """
    preds_out = np.asarray(preds, dtype=float).copy()
    weeks = pd.to_datetime(df["Date"]).dt.isocalendar().week.to_numpy()
    depts = df["Dept"].to_numpy()

    if target_weeks is None:
        target_weeks = [47, 51]

    if surge_depts is not None and len(surge_depts) > 0:
        mask = np.isin(weeks, target_weeks) & np.isin(depts, surge_depts)
    else:
        mask = np.isin(weeks, target_weeks)

    preds_out[mask] = preds_out[mask] * alpha
    return preds_out
