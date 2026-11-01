"""
TÁI LẬP CHUỖI CẢI TIẾN (Benchmark -> ML Baseline -> P1 -> P2 -> P3) TRÊN NHIỀU SEED

Dữ liệu cho BẢNG 1 của Src/Evaluation/generate_benchmark_tables.py. Mỗi seed huấn luyện lại:
  - ML Baseline : LightGBM L2 (Src/LightGBM/train.py) và XGBoost L1 (Src/XGBoost/train.py)
  - P1          : LightGBM L1 loss + Zero-Clipping (Src/Improvement/P1)
  - P2          : Weighted Duo Blend (XGB + LGB L2, trọng số tối ưu WMAE trên Val) + Holiday Multiplier (Src/Improvement/P2)
  - P3          : LightGBM + Group Stats + siêu tham số Optuna (checkpoints/p3_lightgbm_optuna/best_params.json),
                  Super Blend với XGBoost + Holiday Multiplier (Src/Improvement/P3)
  - Benchmark   : Seasonal Naive (lag 52) và Moving Average 4 tuần — tất định, std = 0.
Giữ nguyên siêu tham số gốc của từng giai đoạn; chỉ thay seed. Lưu ý: cấu hình gốc của LightGBM
L2/L1 không bật bagging; phần lớn phương sai giữa các seed đến từ XGBoost (subsample 0.8) và tính không tất định của huấn luyện đa luồng.

Đầu ra: results/07_improvement/a_p1_p3/improvement_multiseed.json, .md
        predictions_seed<seed đầu tiên>.parquet (dự báo từng dòng Train/Val/Test — đầu vào cho dự báo phân cấp)

Cách chạy:
    python Src/Experiments/run_improvement_multiseed.py
    python Src/Experiments/run_improvement_multiseed.py --quick
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from Config import get_result_dir, get_checkpoint_dir
from Metric.metrics import evaluate_all, compute_mase_scale
from Src.Experiments.common import (
    HARDWARE_INFO_PATH, setup_console, load_processed_train, static_split, holiday_weights,
    sample_series, mean_std, format_mean_std, save_json, markdown_table,
)
from Src.Experiments import model_zoo
from Src.Utils.system_profiler import Timer, save_hardware_info
from Src.Improvement.P2.blending import weighted_blend, find_optimal_weights
from Src.Improvement.P2.holiday_multiplier import identify_surge_departments, apply_holiday_multiplier
from Src.Improvement.P3.features_p3 import build_p3_features

RESULT_DIR = get_result_dir("improvement_multiseed")
P3_PARAMS_PATH = get_checkpoint_dir("p3_lightgbm_optuna") / "best_params.json"
DEFAULT_SEEDS = [42, 123, 2024]
CAT_COLS = ["Store", "Dept", "Type"]
DROP_COLS = ["Weekly_Sales", "Date"]
METRIC_KEYS = ["WMAE", "MAE", "RMSE", "MASE", "MAPE (%)", "R2"]

# Tham số gốc của từng giai đoạn (giữ nguyên như các script train ban đầu)
LGB_L2_PARAMS = {"objective": "regression", "n_estimators": 5000, "learning_rate": 0.03, "num_leaves": 31, "verbose": -1}
LGB_L1_PARAMS = {**LGB_L2_PARAMS, "objective": "regression_l1", "metric": "l1"}
XGB_PARAMS = {"objective": "reg:absoluteerror", "eval_metric": "mae", "tree_method": "hist", "enable_categorical": True,
              "n_estimators": 2000, "learning_rate": 0.05, "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.8}
P3_FIXED = {"objective": "regression_l1", "metric": "l1", "n_estimators": 4000, "subsample_freq": 1, "verbose": -1}

# Tên & mô tả các dòng trong Bảng 1
STAGES = [
    ("benchmark_snaive", "Benchmark Gốc", "Seasonal Naive (lag 52)"),
    ("benchmark_ma4", "Benchmark Gốc", "Moving Average 4 tuần"),
    ("ml_lgb_l2", "ML Baseline", "LightGBM mặc định (L2 loss)"),
    ("ml_xgb_l1", "ML Baseline", "XGBoost mặc định (L1 loss)"),
    ("p1", "Giai đoạn P1", "LightGBM L1 Loss + Zero-Clipping"),
    ("p2", "Giai đoạn P2", "Weighted Duo Blend + Holiday Multiplier"),
    ("p3_lgb", "Giai đoạn P3", "LightGBM đơn + Group Stats + Optuna"),
    ("p3", "Giai đoạn P3", "Group Stats + Optuna + Super Blend + Multiplier"),
]


def train_predict(model_name, params, seed, device, train_df, val_df, test_df, feature_cols, levels, es_rounds):
    """Huấn luyện một mô hình (ES trên Val), trả về dự báo (Train, Val, Test) đã zero-clip + thời gian fit."""
    p = model_zoo.build_params(model_name, params, seed, device)
    X_tr = model_zoo.prepare_X(model_name, train_df, feature_cols, CAT_COLS, levels)
    X_val = model_zoo.prepare_X(model_name, val_df, feature_cols, CAT_COLS, levels)
    X_te = model_zoo.prepare_X(model_name, test_df, feature_cols, CAT_COLS, levels)
    with Timer() as t:
        model = model_zoo.fit_model(model_name, p, X_tr, train_df["Weekly_Sales"], holiday_weights(train_df),
                                    X_val, val_df["Weekly_Sales"], holiday_weights(val_df), CAT_COLS, es_rounds)
    return (model_zoo.predict(model, X_tr), model_zoo.predict(model, X_val), model_zoo.predict(model, X_te)), t.elapsed


def run_seed(seed, df, p3_params, quick, devices, save_predictions=False):
    """Chạy toàn bộ chuỗi cải tiến cho một seed."""
    train_df, val_df, test_df = static_split(df)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    levels = model_zoo.category_levels(df, CAT_COLS)
    cap = {"n_estimators": 300} if quick else {}

    print(f"\n----- SEED {seed} -----")
    preds, times = {}, {}

    # Benchmark (tất định)
    # Mỗi phần tử preds[key] = (dự báo Train, dự báo Val, dự báo Test)
    preds["benchmark_snaive"] = tuple(np.clip(d["sales_lag_52"].to_numpy(), 0, None) for d in (train_df, val_df, test_df))
    preds["benchmark_ma4"] = tuple(np.clip(d["sales_rolling_mean_4"].to_numpy(), 0, None) for d in (train_df, val_df, test_df))

    # ML Baseline
    preds["ml_lgb_l2"], times["ml_lgb_l2"] = train_predict("lightgbm", {**LGB_L2_PARAMS, **cap}, seed, devices["lightgbm"],
                                                         train_df, val_df, test_df, feature_cols, levels, 100)
    print(f"   LightGBM L2 xong ({times['ml_lgb_l2']:,.1f}s)")
    preds["ml_xgb_l1"], times["ml_xgb_l1"] = train_predict("xgboost", {**XGB_PARAMS, **cap}, seed, devices["xgboost"],
                                                         train_df, val_df, test_df, feature_cols, levels, 50)
    print(f"   XGBoost L1 xong ({times['ml_xgb_l1']:,.1f}s)")

    # P1
    preds["p1"], times["p1"] = train_predict("lightgbm", {**LGB_L1_PARAMS, **cap}, seed, devices["lightgbm"],
                                             train_df, val_df, test_df, feature_cols, levels, 100)
    print(f"   P1 LightGBM L1 xong ({times['p1']:,.1f}s)")

    y_val = val_df["Weekly_Sales"].to_numpy()
    val_holiday = val_df["IsHoliday"].to_numpy()
    surge_depts = identify_surge_departments(train_df, surge_threshold=1.35)

    def blend_with_multiplier(pred_a, pred_b, init):
        """Trọng số tối ưu WMAE trên Val (idx 1), áp dụng cho cả Train/Val/Test + Holiday Multiplier tuần 47."""
        w = find_optimal_weights([pred_a[1], pred_b[1]], y_val, val_holiday, init_weights=init)
        out = tuple(
            apply_holiday_multiplier(weighted_blend([pred_a[i], pred_b[i]], w), d,
                                     alpha=1.02, surge_depts=surge_depts, target_weeks=[47])
            for i, d in enumerate((train_df, val_df, test_df))
        )
        return out, w

    # P2
    preds["p2"], w_p2 = blend_with_multiplier(preds["ml_xgb_l1"], preds["ml_lgb_l2"], [0.75, 0.25])
    times["p2"] = times["ml_xgb_l1"] + times["ml_lgb_l2"]

    # P3
    train_p3, val_p3, test_p3, _ = build_p3_features(train_df, val_df, test_df)
    p3_cols = [c for c in train_p3.columns if c not in DROP_COLS]
    lgb_p3, t_p3 = train_predict("lightgbm", {**P3_FIXED, **p3_params, **cap}, seed, devices["lightgbm"],
                                 train_p3, val_p3, test_p3, p3_cols, levels, 100)
    print(f"   P3 LightGBM Optuna xong ({t_p3:,.1f}s)")
    preds["p3_lgb"], times["p3_lgb"] = lgb_p3, t_p3
    preds["p3"], w_p3 = blend_with_multiplier(preds["ml_xgb_l1"], lgb_p3, [0.6, 0.4])
    times["p3"] = times["ml_xgb_l1"] + t_p3

    mase_scale = compute_mase_scale(train_df)
    metrics = {}
    for key, _, _ in STAGES:
        _, pv, pt = preds[key]
        metrics[key] = {
            "val": evaluate_all(y_val, pv, val_holiday, mase_scale=mase_scale),
            "test": evaluate_all(test_df["Weekly_Sales"], pt, test_df["IsHoliday"], mase_scale=mase_scale),
            "fit_time_s": times.get(key, 0.0),
        }
        print(f"   {key:<17} Val WMAE {metrics[key]['val']['WMAE']:>9,.2f} | Test WMAE {metrics[key]['test']['WMAE']:>9,.2f}")

    # Lưu dự báo từng dòng Train (trong mẫu) / Val / Test — đầu vào cho Hierarchical Reconciliation
    if save_predictions:
        frames = []
        for idx, (split_name, split_df) in enumerate([("train", train_df), ("val", val_df), ("test", test_df)]):
            out = split_df[["Store", "Dept", "Date", "IsHoliday", "Weekly_Sales"]].copy()
            out["split"] = split_name
            for key, _, _ in STAGES:
                out[f"pred_{key}"] = preds[key][idx]
            frames.append(out)
        path = RESULT_DIR / f"predictions_seed{seed}{'_quick' if quick else ''}.parquet"
        pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
        print(f"   Đã lưu dự báo từng dòng: {path.name}")

    return {"seed": seed, "metrics": metrics, "mase_scale": mase_scale,
            "blend_weights": {"p2": w_p2.tolist(), "p3": w_p3.tolist()}, "n_surge_depts": len(surge_depts)}


def summarize(runs):
    """Mean ± std của mọi metric theo từng giai đoạn."""
    out = {}
    for key, _, _ in STAGES:
        out[key] = {
            split: {m: mean_std([r["metrics"][key][split][m] for r in runs]) for m in METRIC_KEYS}
            for split in ["val", "test"]
        }
        out[key]["fit_time_s"] = mean_std([r["metrics"][key]["fit_time_s"] for r in runs])
    return out


def to_markdown(result):
    """Bảng mean ± std cho cả Val và Test."""
    parts = []
    for split, title in [("val", "Validation (Q4/2011)"), ("test", "Holdout Test (2012)")]:
        rows = []
        for key, phase, idea in STAGES:
            s = result["summary"][key][split]
            rows.append([phase, idea, format_mean_std(s["WMAE"]), format_mean_std(s["MAE"]),
                         format_mean_std(s["RMSE"]), format_mean_std(s["MASE"], ".3f")])
        parts.append(f"### {title}\n\n" + markdown_table(
            ["Phiên bản", "Ý tưởng", "WMAE", "MAE", "RMSE", "MASE"], rows,
            [":---", ":---", "---:", "---:", "---:", "---:"]))
    return (f"# Chuỗi cải tiến trên {len(result['seeds'])} seed {result['seeds']} (Mean ± Std)\n\n"
            + "\n\n".join(parts) + "\n")


def main():
    setup_console()
    parser = argparse.ArgumentParser(description="Tái lập Benchmark -> P1 -> P2 -> P3 trên nhiều seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--quick", action="store_true", help="Chạy thử nhanh: 10%% chuỗi, tối đa 300 cây")
    args = parser.parse_args()

    if not P3_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Chưa có {P3_PARAMS_PATH}. Hãy chạy Src/Improvement/P3/run_p3.py trước.")
    with open(P3_PARAMS_PATH, encoding="utf-8") as f:
        p3_params = json.load(f)["params"]

    df = load_processed_train()
    if args.quick:
        df = sample_series(df, 0.1, seed=0)

    hardware = save_hardware_info(HARDWARE_INFO_PATH)
    devices = {"lightgbm": "cpu", "xgboost": model_zoo.resolve_device("xgboost", "auto")}

    # Chỉ lưu dự báo từng dòng của seed đầu tiên (dùng cho dự báo phân cấp)
    runs = [run_seed(seed, df, p3_params, args.quick, devices, save_predictions=(i == 0))
            for i, seed in enumerate(args.seeds)]
    result = {"seeds": args.seeds, "quick": args.quick, "devices": devices, "p3_params": p3_params,
              "stages": [{"key": k, "phase": p, "idea": i} for k, p, i in STAGES],
              "runs": runs, "summary": summarize(runs), "hardware": hardware}
    suffix = "_quick" if args.quick else ""
    save_json(result, RESULT_DIR / f"improvement_multiseed{suffix}.json")
    md = to_markdown(result)
    (RESULT_DIR / f"improvement_multiseed{suffix}.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
