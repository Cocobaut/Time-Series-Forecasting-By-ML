"""
BASELINE TUYẾN TÍNH: RIDGE / LASSO VỚI LAG FEATURES

Mô hình tuyến tính có điều chuẩn L2 (Ridge) và L1 (Lasso) trên cùng bộ đặc trưng tiền xử lý
(lag, rolling, lịch, cờ lễ hội, biến ngoại sinh) + one-hot Store / Dept / Type:
  - Học trên thang tương đối y / s (s = doanh số TB của chuỗi Store×Dept trên Train), winsorize + chuẩn hóa
    đặc trưng số, cờ thiếu lịch sử (chi tiết trong build_design),
    trọng số mẫu WMAE (x5 tuần lễ) x s, zero-clipping dự báo.
  - Hệ số điều chuẩn alpha chọn theo WMAE trên Validation Q4/2011 (cùng giao thức chọn siêu tham số
    với các mô hình GBDT); Holdout Test 2012 chỉ dùng để báo cáo.
Mô hình tất định nên không cần nhiều seed (std = 0).

Đầu ra: results/01_baseline/linear_results.txt, linear_results.json

Cách chạy:
    python Src/Baseline/linear_baseline.py
    python Src/Baseline/linear_baseline.py --quick
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from Config import get_result_dir
from Metric.metrics import evaluate_all, compute_mase_scale, format_metric_report
from Src.Experiments.common import (
    setup_console, load_processed_train, static_split, holiday_weights, sample_series, save_json, markdown_table,
)
from Src.Utils.system_profiler import Timer

RESULT_DIR = get_result_dir("baseline")
ONE_HOT_COLS = ["Store", "Dept", "Type"]
DROP_COLS = ["Weekly_Sales", "Date"]
# Alpha áp lên target tương đối y / s (s = quy mô chuỗi, xem series_scale)
ALPHA_GRID = {"Ridge": [0.1, 1.0, 10.0, 100.0, 1000.0], "Lasso": [1e-3, 1e-2, 1e-1]}
WINSOR_Q = (0.001, 0.999)
LONG_LAGS = ["sales_lag_51", "sales_lag_52", "sales_lag_53"]
SHORT_LAGS = ["sales_lag_1", "sales_lag_2", "sales_lag_4", "sales_lag_8"]


def series_scale(train_df):
    """Quy mô từng chuỗi s = doanh số trung bình (Store, Dept) trên Train (tối thiểu 1 USD)."""
    s_map = train_df.groupby(["Store", "Dept"])["Weekly_Sales"].mean().clip(lower=1.0)
    fallback = float(s_map.median())

    def scale_of(df):
        s = df.set_index(["Store", "Dept"]).index.map(s_map).to_numpy(dtype=float)
        return np.nan_to_num(s, nan=fallback)
    return scale_of


def build_design(train_df, val_df, test_df):
    """
    Ma trận thiết kế cho mô hình tuyến tính trên thang TƯƠNG ĐỐI (y / s, lag / s):
      - Hiệu ứng mùa lễ của Walmart mang tính nhân theo quy mô phòng ban; trên thang USD, mô hình cộng tính
        áp cùng một mức tăng (vd. weeks_to_thanksgiving) cho mọi phòng ban nên sai rất nặng ở Q4.
        Chia target và lag/rolling cho quy mô chuỗi s đưa các chuỗi về cùng thang.
      - preprocess.py điền -1 cho lag/rolling chưa đủ lịch sử (khoảng nửa tập Train với lag 51-53): coi là thiếu,
        thay bằng trung bình (0 sau chuẩn hóa) và thêm cờ miss_lag_long / miss_lag_short.
      - Winsorize về phân vị [0.1%, 99.9%] của Train: sales_growth_ratio_4 có ngoại lai tới -8.8e12.
    GBDT không cần các bước này vì chỉ chia ngưỡng. Trả về thêm vector quy mô s của từng tập.
    """
    num_cols = [c for c in train_df.columns if c not in DROP_COLS + ONE_HOT_COLS]
    hist_cols = [c for c in num_cols if "lag" in c or "rolling" in c]
    scale_of = series_scale(train_df)

    def mask_placeholders(df):
        out = df[num_cols].copy()
        out[hist_cols] = out[hist_cols].mask(out[hist_cols] == -1.0).div(scale_of(df), axis=0)
        return out

    train_num = mask_placeholders(train_df)
    lower, upper = train_num.quantile(WINSOR_Q[0]), train_num.quantile(WINSOR_Q[1])
    scaler = StandardScaler().fit(train_num.clip(lower, upper, axis=1))

    def transform(df):
        raw = mask_placeholders(df)
        num = pd.DataFrame(scaler.transform(raw.clip(lower, upper, axis=1)), columns=num_cols, index=df.index).fillna(0.0)
        num["miss_lag_long"] = raw[LONG_LAGS].isna().any(axis=1).astype(float)
        num["miss_lag_short"] = raw[SHORT_LAGS].isna().any(axis=1).astype(float)
        dummies = pd.get_dummies(df[ONE_HOT_COLS].astype(str), prefix=ONE_HOT_COLS, dtype=float)
        return pd.concat([num, dummies], axis=1)

    X_tr = transform(train_df)
    cols = X_tr.columns
    X_val = transform(val_df).reindex(columns=cols, fill_value=0.0)
    X_te = transform(test_df).reindex(columns=cols, fill_value=0.0)
    scales = (scale_of(train_df), scale_of(val_df), scale_of(test_df))
    return X_tr, X_val, X_te, num_cols, scales


def main():
    setup_console()
    parser = argparse.ArgumentParser(description="Baseline tuyến tính Ridge / Lasso với lag features")
    parser.add_argument("--quick", action="store_true", help="Chạy thử nhanh trên 10%% chuỗi")
    args = parser.parse_args()

    df = load_processed_train()
    if args.quick:
        df = sample_series(df, 0.1, seed=0)
    train_df, val_df, test_df = static_split(df)
    X_tr, X_val, X_te, num_cols, (s_tr, s_val, s_te) = build_design(train_df, val_df, test_df)
    y_rel = train_df["Weekly_Sales"].to_numpy() / s_tr
    # Trọng số x s để bình phương sai số tương đối xấp xỉ sai số theo USD (chuỗi lớn vẫn quan trọng hơn)
    w_tr = holiday_weights(train_df) * s_tr
    mase_scale = compute_mase_scale(train_df)
    print(f"[Linear] Ma trận thiết kế: {X_tr.shape[0]:,} dòng x {X_tr.shape[1]} cột "
          f"({len(num_cols)} đặc trưng số + one-hot Store/Dept/Type)")

    results = {}
    for name, cls in [("Ridge", Ridge), ("Lasso", Lasso)]:
        search = []
        best = None
        for alpha in ALPHA_GRID[name]:
            model = cls(alpha=alpha, max_iter=2000, tol=1e-4, selection="random", random_state=0) if name == "Lasso" else cls(alpha=alpha)
            with Timer() as t:
                model.fit(X_tr, y_rel, sample_weight=w_tr)
            p_val = np.clip(model.predict(X_val) * s_val, 0.0, None)
            val_m = evaluate_all(val_df["Weekly_Sales"], p_val, val_df["IsHoliday"], mase_scale=mase_scale)
            search.append({"alpha": alpha, "val_wmae": val_m["WMAE"], "fit_time_s": t.elapsed})
            print(f"   {name:<5} alpha={alpha:<7} | Val WMAE {val_m['WMAE']:>10,.2f} | {t.elapsed:,.1f}s")
            if best is None or val_m["WMAE"] < best[1]["WMAE"]:
                best = (model, val_m, alpha, t.elapsed)

        model, val_m, alpha, fit_time = best
        with Timer() as t_pred:
            p_te = np.clip(model.predict(X_te) * s_te, 0.0, None)
        test_m = evaluate_all(test_df["Weekly_Sales"], p_te, test_df["IsHoliday"], mase_scale=mase_scale)
        coef = pd.Series(model.coef_, index=X_tr.columns)  # hệ số trên thang tương đối y / s
        top = coef.abs().sort_values(ascending=False).head(10)
        results[name] = {
            "best_alpha": alpha, "val": val_m, "test": test_m, "alpha_search": search,
            "fit_time_s": fit_time, "inference_time_s": t_pred.elapsed,
            "n_parameters": int(np.sum(model.coef_ != 0) + 1), "n_features": int(X_tr.shape[1]),
            "top_coefficients": {k: float(coef[k]) for k in top.index},
        }

    suffix = "_quick" if args.quick else ""
    save_json({"quick": args.quick, "results": results}, RESULT_DIR / f"linear_results{suffix}.json")

    rows = []
    for name, r in results.items():
        for split in ["val", "test"]:
            m = r[split]
            rows.append([name, r["best_alpha"], "Validation Q4/2011" if split == "val" else "Holdout Test 2012",
                         f"{m['WMAE']:,.2f}", f"{m['MAE']:,.2f}", f"{m['RMSE']:,.2f}", f"{m['MASE']:.4f}",
                         r["n_parameters"]])
    table = markdown_table(["Mô hình", "Alpha", "Tập", "WMAE", "MAE", "RMSE", "MASE", "Hệ số khác 0"], rows,
                           [":---", "---:", ":---", "---:", "---:", "---:", "---:", "---:"])
    reports = []
    for name, r in results.items():
        reports.append(format_metric_report(f"{name} (alpha={r['best_alpha']}) - Validation", r["val"]))
        reports.append(format_metric_report(f"{name} (alpha={r['best_alpha']}) - Test 2012", r["test"]))
        reports.append(" Top-10 hệ số (theo |hệ số|, đặc trưng đã chuẩn hóa):\n" + "\n".join(
            f"   {k:<28} {v:>12,.2f}" for k, v in r["top_coefficients"].items()) + "\n")
    text = ("LINEAR BASELINE RESULTS (Ridge / Lasso với lag features)\n"
            "========================================================\n"
            "Alpha chọn theo WMAE trên Validation; Test 2012 chỉ dùng để báo cáo.\n\n"
            + table + "\n\n" + "\n".join(reports))
    (RESULT_DIR / f"linear_results{suffix}.txt").write_text(text, encoding="utf-8")
    print("\n" + table)
    print(f"\n[Linear] Đã lưu: {RESULT_DIR / f'linear_results{suffix}.txt'}")


if __name__ == "__main__":
    main()
