"""
GIAI ĐOẠN 1 — ROLLING ORIGIN CROSS-VALIDATION (EXPANDING WINDOW)

Đánh giá mô hình trên 4 mốc dự báo (origin) liên tiếp, mỗi fold dự báo 13 tuần tiếp theo:
    Fold 1: Train <= 2011-04-30 -> Val 2011-05 .. 2011-07
    Fold 2: Train <= 2011-07-31 -> Val 2011-08 .. 2011-10
    Fold 3: Train <= 2011-10-31 -> Val 2011-11 .. 2012-01 (Q4 Holiday)
    Fold 4: Train <= 2012-01-31 -> Val 2012-02 .. 2012-04
(Các mốc lấy từ mục [[cv.folds]] trong Config/experiments/*.toml.)

Chống rò rỉ dữ liệu:
  - feature_mode = "horizon_safe" (mặc định): với MỖI fold, doanh số sau origin bị che (NaN) rồi
    lag/rolling được tái tính từ lưới thời gian liên tục bằng chính các hàm của preprocess.py;
    các đặc trưng có độ trễ < horizon (lag 1/2/4/8, rolling 4/8, ...) bị loại khỏi mô hình.
  - feature_mode = "one_step": dùng đặc trưng tiền xử lý sẵn (dự báo 1 bước, mỗi tuần biết doanh số
    tuần trước — cùng giao thức với P1/P2/P3).
  - Early stopping dùng 8 tuần CUỐI của phần train trong fold, không bao giờ chạm vào tập val của fold.
  - Mẫu số MASE tính lại từ dữ liệu train của từng fold.

Cách chạy:
    python Src/Evaluation/time_series_cv.py                       # cả 3 mô hình
    python Src/Evaluation/time_series_cv.py --configs lightgbm_tuned --quick
"""
import sys
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from Config import load_experiment_config, get_result_dir
from Metric.metrics import evaluate_all, compute_mase_scale
from Src.Experiments.common import (
    HARDWARE_INFO_PATH, setup_console, load_processed_train, clean_numeric, get_feature_columns, holiday_weights,
    sample_series, mean_std, format_mean_std, save_json, load_tuned_params, markdown_table,
)
from Src.Experiments import model_zoo
from Src.Utils.system_profiler import Timer, save_hardware_info

CV_RESULT_DIR = get_result_dir("cv")
DEFAULT_CONFIGS = ["lightgbm_tuned", "xgboost_tuned", "catboost_tuned"]


# =============================================================================
# 1. TÁI TẠO ĐẶC TRƯNG THEO TỪNG FOLD (KHÔNG RÒ RỈ)
# =============================================================================

def load_preprocess_module():
    """Nạp Data/Preprocess/preprocess.py theo đường dẫn file (Data/ không phải package Python)."""
    path = BASE_DIR / "Data" / "Preprocess" / "preprocess.py"
    spec = importlib.util.spec_from_file_location("walmart_preprocess", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_base_grid(pp) -> pd.DataFrame:
    """Lưới Store x Dept x Tuần liên tục + biến ngoại sinh + đặc trưng lịch (chưa có lag/rolling)."""
    train, test, stores, features = pp.load_raw_data()
    grid = pp.build_continuous_time_grid(train, test)
    df = pp.merge_exogenous_features(grid, stores, features)
    return pp.create_calendar_and_holiday_features(df)


def build_fold_features(pp, base_grid: pd.DataFrame, origin_end: pd.Timestamp) -> pd.DataFrame:
    """
    Che toàn bộ doanh số sau origin_end rồi tái tính lag/rolling: mọi đặc trưng của một dòng
    chỉ dùng thông tin đã biết tại thời điểm origin. Nhãn thật được giữ riêng để đánh giá.
    """
    df = base_grid.copy()
    df["_y_true"] = df["Weekly_Sales"]
    df.loc[df["Date"] > origin_end, "Weekly_Sales"] = np.nan
    df = pp.create_lag_and_rolling_features(df)
    df = df[df["Is_Train_Record"] == 1].copy()
    df["Weekly_Sales"] = df["_y_true"]
    df = df.drop(columns=["_y_true", "Is_Train_Record", "Is_Test_Record"])
    return clean_numeric(df)


# =============================================================================
# 2. CHẠY ROLLING ORIGIN CV CHO MỘT THÍ NGHIỆM
# =============================================================================

def run_cv_for_config(cfg: dict, fold_frames: dict, seed: int, quick: bool = False) -> dict:
    """Huấn luyện + đánh giá mô hình trên từng fold, trả về dict kết quả."""
    model_name = cfg["experiment"]["model"]
    cv_cfg = cfg["cv"]
    data_cfg = cfg.get("data", {})
    feat_cfg = cfg.get("features", {})
    feature_mode = cv_cfg.get("feature_mode", "horizon_safe")
    cat_cols = feat_cfg.get("categorical", ["Store", "Dept", "Type"])
    hw = data_cfg.get("holiday_weight", 5.0)
    seasonality = cfg.get("metrics", {}).get("mase_seasonality", 52)
    es_weeks = int(cv_cfg.get("early_stopping_weeks", 8))

    device = model_zoo.resolve_device(model_name, cfg.get("device", {}).get("type", "auto"))
    tuned = load_tuned_params(cfg)
    tuned_unified = tuned["params"] if tuned else None
    es_rounds = cfg.get("model", {}).get("fit", {}).get("early_stopping_rounds", 100)
    n_estimators = 300 if quick else None

    print("\n" + "=" * 80)
    print(f" ROLLING ORIGIN CV — {cfg['experiment']['name']} ({model_name}, device={device}, "
          f"feature_mode={feature_mode}, tuned={'yes' if tuned else 'no'})")
    print("=" * 80)

    fold_results = []
    for fold in cv_cfg["folds"]:
        train_end = pd.Timestamp(fold["train_end"])
        val_start, val_end = pd.Timestamp(fold["val_start"]), pd.Timestamp(fold["val_end"])
        df = fold_frames[(feature_mode, fold["train_end"])]

        train_full = df[df["Date"] <= train_end]
        val_df = df[(df["Date"] >= val_start) & (df["Date"] <= val_end)]
        es_cut = train_end - pd.Timedelta(weeks=es_weeks)
        fit_df = train_full[train_full["Date"] <= es_cut]
        es_df = train_full[train_full["Date"] > es_cut]

        feature_cols = get_feature_columns(df, feat_cfg, mode=feature_mode)
        levels = model_zoo.category_levels(df, cat_cols)
        X_fit = model_zoo.prepare_X(model_name, fit_df, feature_cols, cat_cols, levels)
        X_es = model_zoo.prepare_X(model_name, es_df, feature_cols, cat_cols, levels)
        X_val = model_zoo.prepare_X(model_name, val_df, feature_cols, cat_cols, levels)

        params = model_zoo.build_params(model_name, cfg["model"]["params"], seed, device,
                                        tuned_unified, n_estimators=n_estimators)
        with Timer() as t_fit:
            model = model_zoo.fit_model(
                model_name, params,
                X_fit, fit_df["Weekly_Sales"], holiday_weights(fit_df, hw),
                X_es, es_df["Weekly_Sales"], holiday_weights(es_df, hw),
                cat_cols, early_stopping_rounds=es_rounds,
            )
        with Timer() as t_pred:
            preds = model_zoo.predict(model, X_val, data_cfg.get("clip_negative", True))

        scale = compute_mase_scale(train_full, seasonality=seasonality)
        metrics = evaluate_all(val_df["Weekly_Sales"], preds, val_df["IsHoliday"], mase_scale=scale)
        n_weeks = val_df["Date"].nunique()
        fold_results.append({
            "fold": fold["name"],
            "train_end": fold["train_end"],
            "val_start": fold["val_start"],
            "val_end": fold["val_end"],
            "n_train_rows": int(len(fit_df)), "n_es_rows": int(len(es_df)), "n_val_rows": int(len(val_df)),
            "n_val_weeks": int(n_weeks),
            "mase_scale": scale,
            "best_iteration": model_zoo.best_iteration(model_name, model),
            "fit_time_s": t_fit.elapsed, "inference_time_s": t_pred.elapsed,
            "metrics": metrics,
        })
        print(f" {fold['name']:<22} | {n_weeks:>2} tuần | WMAE {metrics['WMAE']:>10,.2f} | MAE {metrics['MAE']:>10,.2f} "
              f"| RMSE {metrics['RMSE']:>10,.2f} | MASE {metrics['MASE']:.4f} | fit {t_fit.elapsed:,.1f}s")

    summary = {k: mean_std([f["metrics"][k] for f in fold_results]) for k in ["WMAE", "MAE", "RMSE", "MASE"]}
    return {
        "experiment": cfg["experiment"]["name"], "model": model_name, "device": device, "seed": seed,
        "feature_mode": feature_mode, "tuned_params": tuned_unified, "quick": quick,
        "folds": fold_results, "summary": summary,
    }


def cv_markdown(result: dict) -> str:
    """Bảng kết quả CV: Fold | MAE | RMSE | MASE | WMAE."""
    rows = []
    for f in result["folds"]:
        m = f["metrics"]
        rows.append([f["fold"], f"{f['val_start']} → {f['val_end']}", f"{m['MAE']:,.2f}", f"{m['RMSE']:,.2f}",
                     f"{m['MASE']:.4f}", f"{m['WMAE']:,.2f}", f["best_iteration"], f"{f['fit_time_s']:,.1f}"])
    s = result["summary"]
    rows.append(["**Mean ± Std**", "", format_mean_std(s["MAE"]), format_mean_std(s["RMSE"]),
                 format_mean_std(s["MASE"], ".4f"), format_mean_std(s["WMAE"]), "", ""])
    header = (f"### Rolling Origin CV — {result['experiment']} "
              f"(feature_mode = `{result['feature_mode']}`, seed = {result['seed']}, device = {result['device']})\n\n")
    return header + markdown_table(
        ["Fold", "Khoảng Val", "MAE", "RMSE", "MASE", "WMAE", "Best iter", "Fit (s)"], rows,
        [":---", ":---", "---:", "---:", "---:", "---:", "---:", "---:"],
    )


def main():
    setup_console()
    parser = argparse.ArgumentParser(description="Rolling Origin (Expanding Window) Time-Series CV")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="Tên file trong Config/experiments")
    parser.add_argument("--seed", type=int, default=None, help="Seed dùng cho CV (mặc định: seed đầu tiên trong config)")
    parser.add_argument("--feature-mode", choices=["horizon_safe", "one_step"], default=None,
                        help="Ghi đè cv.feature_mode trong config")
    parser.add_argument("--quick", action="store_true", help="Chạy thử nhanh: 10%% chuỗi, tối đa 300 cây")
    args = parser.parse_args()

    configs = [load_experiment_config(c) for c in args.configs]
    if args.feature_mode:
        for cfg in configs:
            cfg["cv"]["feature_mode"] = args.feature_mode

    # Chuẩn bị dữ liệu từng fold một lần, dùng chung cho mọi mô hình
    fold_frames = {}
    needed = {(cfg["cv"].get("feature_mode", "horizon_safe"), f["train_end"]) for cfg in configs for f in cfg["cv"]["folds"]}
    pp, base_grid, processed = None, None, None
    for mode, train_end in sorted(needed):
        if mode == "one_step":
            processed = processed if processed is not None else load_processed_train()
            fold_frames[(mode, train_end)] = processed
        else:
            if base_grid is None:
                pp = load_preprocess_module()
                base_grid = build_base_grid(pp)
            print(f"\n[CV] Tái tính lag/rolling với origin = {train_end} (che doanh số tương lai)...")
            fold_frames[(mode, train_end)] = build_fold_features(pp, base_grid, pd.Timestamp(train_end))
    if args.quick:
        fold_frames = {k: sample_series(v, 0.1, seed=0) for k, v in fold_frames.items()}

    sfx = "_quick" if args.quick else ""
    save_hardware_info(HARDWARE_INFO_PATH)
    sections = []
    for cfg in configs:
        seed = args.seed if args.seed is not None else cfg["seeds"]["values"][0]
        result = run_cv_for_config(cfg, fold_frames, seed, quick=args.quick)
        save_json(result, CV_RESULT_DIR / f"cv_{cfg['experiment']['name']}{sfx}.json")
        md = cv_markdown(result)
        (CV_RESULT_DIR / f"cv_{cfg['experiment']['name']}{sfx}.md").write_text(md + "\n", encoding="utf-8")
        sections.append(md)
        print("\n" + md)

    # Tổng hợp từ mọi file cv_<thí nghiệm>{sfx}.md hiện có (giữ kết quả của mô hình không chạy lần này)
    sections = [p.read_text(encoding="utf-8").strip()
                for name in DEFAULT_CONFIGS + [c for c in args.configs if c not in DEFAULT_CONFIGS]
                if (p := CV_RESULT_DIR / f"cv_{name}{sfx}.md").exists()]
    summary_path = CV_RESULT_DIR / f"cv_summary{sfx}.md"
    summary_path.write_text("# Rolling Origin Cross-Validation\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"\n[CV] Đã lưu kết quả vào: {CV_RESULT_DIR}")


if __name__ == "__main__":
    main()
