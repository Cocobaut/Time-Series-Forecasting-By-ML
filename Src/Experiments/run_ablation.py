"""
ABLATION STUDY THEO NHÓM ĐẶC TRƯNG + FEATURE IMPORTANCE

Bóc tách đóng góp của từng nhóm đặc trưng bằng cách thêm dần (cumulative) vào cùng một mô hình
XGBoost L1 (tham số của ML Baseline trong Bảng 1), chạy trên nhiều seed (mean ± std):
    C1  Lags           : định danh (Store, Dept, Type, Size, IsHoliday) + lag 1/2/4/8/51/52/53
    C2  + Rolling      : + rolling mean/std 4/8/52, growth ratio, sales/size
    C3  + Calendar     : + Year/Month/Week/Quarter/DayOfYear, sin/cos, cờ 4 kỳ lễ, weeks_to_thanksgiving
    C4  + Exogenous    : + Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment (= bộ đặc trưng đầy đủ)
    C5  + Group Stats  : + 8 đặc trưng thống kê nhóm Store×Dept của P3 (tính từ Train)
Feature importance (gain & số lần split) lấy từ mô hình C5 của seed đầu tiên.

Đầu ra: results/06_ablation/ablation_results.{json,md,txt}, feature_importance.csv, feature_importance.png

Cách chạy:
    python Src/Experiments/run_ablation.py
    python Src/Experiments/run_ablation.py --quick
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from Config import get_result_dir
from Metric.metrics import evaluate_all, compute_mase_scale
from Src.Experiments.common import (
    setup_console, load_processed_train, static_split, holiday_weights, sample_series,
    mean_std, format_mean_std, save_json, markdown_table,
)
from Src.Experiments import model_zoo
from Src.Experiments.run_improvement_multiseed import XGB_PARAMS, CAT_COLS
from Src.Utils.system_profiler import Timer
from Src.Improvement.P3.features_p3 import build_p3_features

RESULT_DIR = get_result_dir("ablation")
DEFAULT_SEEDS = [42, 123, 2024]
METRIC_KEYS = ["WMAE", "MAE", "RMSE", "MASE"]

GROUPS = {
    "id": ["Store", "Dept", "Type", "Size", "IsHoliday"],
    "lags": ["sales_lag_1", "sales_lag_2", "sales_lag_4", "sales_lag_8", "sales_lag_51", "sales_lag_52", "sales_lag_53"],
    "rolling": ["sales_rolling_mean_4", "sales_rolling_std_4", "sales_rolling_mean_8", "sales_rolling_mean_52",
                "sales_growth_ratio_4", "sales_to_size_ratio"],
    "calendar": ["Year", "Month", "Week", "Quarter", "DayOfYear", "Week_sin", "Week_cos", "Month_sin", "Month_cos",
                 "is_super_bowl", "is_labor_day", "is_thanksgiving", "is_christmas", "is_pre_christmas_peak",
                 "weeks_to_thanksgiving"],
    "exogenous": ["Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
                  "Total_MarkDown", "Has_MarkDown", "CPI", "Unemployment"],
}
CONFIGS = [
    ("C1", "Lags", ["id", "lags"]),
    ("C2", "+ Rolling", ["id", "lags", "rolling"]),
    ("C3", "+ Calendar & holiday flags", ["id", "lags", "rolling", "calendar"]),
    ("C4", "+ Exogenous (bộ đầy đủ)", ["id", "lags", "rolling", "calendar", "exogenous"]),
    ("C5", "+ Target Group Stats (P3)", ["id", "lags", "rolling", "calendar", "exogenous", "group_stats"]),
]


def plot_importance(imp: pd.DataFrame, path: Path, top: int = 15):
    """Biểu đồ thanh ngang Top-N đặc trưng theo gain (%)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * top + 1.2))
    ax.barh(data["feature"], data["gain_pct"], color="#2a6fdb")
    for y, v in enumerate(data["gain_pct"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Tỷ trọng gain (%)")
    ax.set_title(f"Top-{top} đặc trưng quan trọng nhất — XGBoost L1 (cấu hình C5)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    setup_console()
    parser = argparse.ArgumentParser(description="Ablation study theo nhóm đặc trưng + feature importance")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--quick", action="store_true", help="Chạy thử nhanh: 10%% chuỗi, tối đa 300 cây")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args()

    df = load_processed_train()
    if args.quick:
        df = sample_series(df, 0.1, seed=0)
    train_df, val_df, test_df = static_split(df)
    train_df, val_df, test_df, group_stats = build_p3_features(train_df, val_df, test_df)
    groups = {**GROUPS, "group_stats": group_stats}
    levels = model_zoo.category_levels(df, CAT_COLS)
    device = model_zoo.resolve_device("xgboost", args.device)
    mase_scale = compute_mase_scale(train_df)
    params = {**XGB_PARAMS, **({"n_estimators": 300} if args.quick else {})}

    runs = {key: [] for key, _, _ in CONFIGS}
    importance = None
    for key, label, group_names in CONFIGS:
        cols = [c for g in group_names for c in groups[g]]
        X_tr = model_zoo.prepare_X("xgboost", train_df, cols, CAT_COLS, levels)
        X_val = model_zoo.prepare_X("xgboost", val_df, cols, CAT_COLS, levels)
        X_te = model_zoo.prepare_X("xgboost", test_df, cols, CAT_COLS, levels)
        for seed in args.seeds:
            p = model_zoo.build_params("xgboost", params, seed, device)
            with Timer() as t:
                model = model_zoo.fit_model("xgboost", p, X_tr, train_df["Weekly_Sales"], holiday_weights(train_df),
                                            X_val, val_df["Weekly_Sales"], holiday_weights(val_df), CAT_COLS, 50)
            val_m = evaluate_all(val_df["Weekly_Sales"], model_zoo.predict(model, X_val), val_df["IsHoliday"],
                                 mase_scale=mase_scale)
            test_m = evaluate_all(test_df["Weekly_Sales"], model_zoo.predict(model, X_te), test_df["IsHoliday"],
                                  mase_scale=mase_scale)
            runs[key].append({"seed": seed, "val": val_m, "test": test_m, "fit_time_s": t.elapsed,
                              "best_iteration": model_zoo.best_iteration("xgboost", model)})
            print(f" {key} {label:<28} seed {seed:>5} | {len(cols):>2} đặc trưng | Val WMAE {val_m['WMAE']:>9,.2f} "
                  f"| Test WMAE {test_m['WMAE']:>9,.2f} | {t.elapsed:,.1f}s")
            if key == "C5" and importance is None:
                booster = model.get_booster()
                gain = booster.get_score(importance_type="total_gain")
                weight = booster.get_score(importance_type="weight")
                importance = pd.DataFrame({"feature": cols})
                importance["gain"] = importance["feature"].map(gain).fillna(0.0)
                importance["split_count"] = importance["feature"].map(weight).fillna(0).astype(int)
                importance["gain_pct"] = 100 * importance["gain"] / importance["gain"].sum()
                importance["split_pct"] = 100 * importance["split_count"] / max(importance["split_count"].sum(), 1)
                importance = importance.sort_values("gain_pct", ascending=False).reset_index(drop=True)

    summary = {key: {split: {m: mean_std([r[split][m] for r in runs[key]]) for m in METRIC_KEYS}
                     for split in ["val", "test"]} for key, _, _ in CONFIGS}
    for key in summary:
        summary[key]["fit_time_s"] = mean_std([r["fit_time_s"] for r in runs[key]])

    sfx = "_quick" if args.quick else ""
    result = {
        "model": "XGBoost L1 (ML Baseline params)", "device": device, "seeds": args.seeds, "quick": args.quick,
        "configs": [{"key": k, "label": l, "groups": g, "n_features": sum(len(groups[x]) for x in g)}
                    for k, l, g in CONFIGS],
        "runs": runs, "summary": summary,
        "feature_importance_top20": importance.head(20).to_dict(orient="records"),
    }
    save_json(result, RESULT_DIR / f"ablation_results{sfx}.json")
    importance.to_csv(RESULT_DIR / f"feature_importance{sfx}.csv", index=False)
    png = RESULT_DIR / f"feature_importance{sfx}.png"
    plot_importance(importance, png)

    rows, prev = [], None
    for c in result["configs"]:
        s = summary[c["key"]]
        delta = "" if prev is None else f"{s['val']['WMAE']['mean'] - prev:+,.2f}"
        prev = s["val"]["WMAE"]["mean"]
        rows.append([c["key"], c["label"], c["n_features"], format_mean_std(s["val"]["WMAE"]), delta,
                     format_mean_std(s["val"]["MASE"], ".3f"), format_mean_std(s["test"]["WMAE"]),
                     format_mean_std(s["test"]["MASE"], ".3f")])
    table = markdown_table(["Cấu hình", "Nhóm đặc trưng", "#Đặc trưng", "Val WMAE", "Δ Val WMAE", "Val MASE",
                            "Test WMAE", "Test MASE"], rows,
                           [":---", ":---", "---:", "---:", "---:", "---:", "---:", "---:"])
    imp_rows = [[i + 1, r["feature"], f"{r['gain_pct']:.2f}%", f"{r['split_pct']:.2f}%"]
                for i, r in importance.head(10).iterrows()]
    imp_table = markdown_table(["#", "Đặc trưng", "Gain (%)", "Split count (%)"], imp_rows,
                               ["---:", ":---", "---:", "---:"])
    md = (f"# Ablation study — XGBoost L1, {len(args.seeds)} seed {args.seeds} (mean ± std)\n\n"
          "Thêm dần từng nhóm đặc trưng; Δ = thay đổi Val WMAE so với cấu hình liền trước.\n\n" + table
          + "\n\n## Top-10 feature importance (cấu hình C5, seed đầu tiên)\n\n" + imp_table
          + f"\n\n![Feature importance]({png.name})\n")
    (RESULT_DIR / f"ablation_results{sfx}.md").write_text(md, encoding="utf-8")
    (RESULT_DIR / f"ablation_results{sfx}.txt").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
