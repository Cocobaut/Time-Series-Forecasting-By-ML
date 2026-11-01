"""
GIAI ĐOẠN 2 — THỰC NGHIỆM ĐA SEED + ĐO PHẦN CỨNG / THỜI GIAN / SỐ THAM SỐ

Với mỗi thí nghiệm trong Config/experiments (mặc định LightGBM, XGBoost, CatBoost):
  - Huấn luyện lại trên >= 3 seed (seeds.values = [42, 123, 2024]) với cùng split thời gian,
  - Dùng bộ siêu tham số tốt nhất từ Src/Experiments/tune.py nếu có (ngược lại: tham số trong config),
  - Đo fit_time / inference_time (time.perf_counter), số cây / lá / node split, kích thước mô hình,
  - Báo cáo WMAE, MAE, RMSE, MASE, MAPE, R2 trên Validation (Q4/2011) và Holdout Test (2012) dạng mean ± std.

Đầu ra:
  results/05_multiseed/<thí nghiệm>/multiseed_results.json, multiseed_summary.md   (từng thí nghiệm)
  results/05_multiseed/multiseed_summary.md, .json; results/hardware_info.json        (tổng hợp)

Cách chạy:
    python Src/Experiments/run_multiseed.py
    python Src/Experiments/run_multiseed.py --configs xgboost_tuned --seeds 42 123 2024 7 99
    python Src/Experiments/run_multiseed.py --quick
"""
import sys
import argparse
from pathlib import Path
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from Config import load_experiment_config, get_result_dir
from Metric.metrics import evaluate_all, compute_mase_scale
from Src.Experiments.common import (
    HARDWARE_INFO_PATH, setup_console, load_processed_train, static_split, get_feature_columns,
    holiday_weights, sample_series, mean_std, format_mean_std, multiseed_dir, load_tuned_params,
    save_json, load_json, markdown_table,
)
from Src.Experiments import model_zoo
from Src.Utils.system_profiler import Timer, count_model_parameters, save_hardware_info, format_hardware_report

DEFAULT_CONFIGS = ["lightgbm_tuned", "xgboost_tuned", "catboost_tuned"]
METRIC_KEYS = ["WMAE", "MAE", "RMSE", "MASE", "MAPE (%)", "R2"]
COST_KEYS = ["fit_time_s", "inference_time_s", "best_iteration", "n_parameters"]


def run_experiment(cfg: dict, df, seeds: list, quick: bool = False) -> dict:
    """Huấn luyện + đánh giá một thí nghiệm trên nhiều seed."""
    model_name = cfg["experiment"]["model"]
    data_cfg = cfg.get("data", {})
    feat_cfg = cfg.get("features", {})
    cat_cols = feat_cfg.get("categorical", ["Store", "Dept", "Type"])
    hw = data_cfg.get("holiday_weight", 5.0)
    clip = data_cfg.get("clip_negative", True)
    seasonality = cfg.get("metrics", {}).get("mase_seasonality", 52)
    es_rounds = cfg.get("model", {}).get("fit", {}).get("early_stopping_rounds", 100)

    device = model_zoo.resolve_device(model_name, cfg.get("device", {}).get("type", "auto"))
    tuned = load_tuned_params(cfg)
    tuned_unified = tuned["params"] if tuned else None

    train_df, val_df, test_df = static_split(df, data_cfg)
    feature_cols = get_feature_columns(df, feat_cfg)
    levels = model_zoo.category_levels(df, cat_cols)
    X_tr = model_zoo.prepare_X(model_name, train_df, feature_cols, cat_cols, levels)
    X_val = model_zoo.prepare_X(model_name, val_df, feature_cols, cat_cols, levels)
    X_te = model_zoo.prepare_X(model_name, test_df, feature_cols, cat_cols, levels)
    w_tr, w_val = holiday_weights(train_df, hw), holiday_weights(val_df, hw)
    mase_scale = compute_mase_scale(train_df, seasonality=seasonality)

    print("\n" + "=" * 80)
    print(f" MULTI-SEED — {cfg['experiment']['name']} ({model_name}, device={device}, "
          f"params={'tuned' if tuned else 'config'}, seeds={seeds})")
    print("=" * 80)

    runs = []
    for seed in seeds:
        params = model_zoo.build_params(model_name, cfg["model"]["params"], seed, device, tuned_unified,
                                        n_estimators=300 if quick else None)
        with Timer() as t_fit:
            model = model_zoo.fit_model(model_name, params, X_tr, train_df["Weekly_Sales"], w_tr,
                                        X_val, val_df["Weekly_Sales"], w_val, cat_cols, es_rounds)
        with Timer() as t_val:
            p_val = model_zoo.predict(model, X_val, clip)
        with Timer() as t_te:
            p_te = model_zoo.predict(model, X_te, clip)

        val_m = evaluate_all(val_df["Weekly_Sales"], p_val, val_df["IsHoliday"], mase_scale=mase_scale)
        test_m = evaluate_all(test_df["Weekly_Sales"], p_te, test_df["IsHoliday"], mase_scale=mase_scale)
        param_stats = count_model_parameters(model, model_name)
        run = {
            "seed": seed,
            "val": val_m, "test": test_m,
            "fit_time_s": t_fit.elapsed,
            "inference_time_s": t_val.elapsed + t_te.elapsed,
            "inference_rows": int(len(X_val) + len(X_te)),
            "best_iteration": model_zoo.best_iteration(model_name, model),
            "model_stats": param_stats,
            "n_parameters": param_stats["n_parameters"],
        }
        runs.append(run)
        print(f" seed {seed:>5} | Val WMAE {val_m['WMAE']:>9,.2f} MASE {val_m['MASE']:.4f} | "
              f"Test WMAE {test_m['WMAE']:>9,.2f} MASE {test_m['MASE']:.4f} | fit {t_fit.elapsed:,.1f}s | "
              f"iter {run['best_iteration']} | params {run['n_parameters']:,}")

    summary = {
        "val": {k: mean_std([r["val"][k] for r in runs]) for k in METRIC_KEYS},
        "test": {k: mean_std([r["test"][k] for r in runs]) for k in METRIC_KEYS},
        "cost": {k: mean_std([r[k] for r in runs]) for k in COST_KEYS},
    }
    result = {
        "experiment": cfg["experiment"]["name"], "model": model_name, "device": device,
        "params_source": "tuned" if tuned else "config", "tuned_params": tuned_unified,
        "seeds": seeds, "quick": quick, "mase_scale": mase_scale, "mase_seasonality": seasonality,
        "n_features": len(feature_cols), "n_train_rows": int(len(train_df)),
        "runs": runs, "summary": summary,
    }
    out_dir = multiseed_dir(cfg)
    suffix = "_quick" if quick else ""
    save_json(result, out_dir / f"multiseed_results{suffix}.json")
    (out_dir / f"multiseed_summary{suffix}.md").write_text(experiment_markdown(result) + "\n", encoding="utf-8")
    return result


def experiment_markdown(result: dict) -> str:
    """Bảng chi tiết từng seed + dòng mean ± std của một thí nghiệm."""
    rows = []
    for r in result["runs"]:
        rows.append([r["seed"], f"{r['val']['WMAE']:,.2f}", f"{r['val']['MASE']:.4f}",
                     f"{r['test']['WMAE']:,.2f}", f"{r['test']['MAE']:,.2f}", f"{r['test']['RMSE']:,.2f}",
                     f"{r['test']['MASE']:.4f}", f"{r['fit_time_s']:,.1f}", r["best_iteration"], f"{r['n_parameters']:,}"])
    s = result["summary"]
    rows.append(["**Mean ± Std**", format_mean_std(s["val"]["WMAE"]), format_mean_std(s["val"]["MASE"], ".4f"),
                 format_mean_std(s["test"]["WMAE"]), format_mean_std(s["test"]["MAE"]),
                 format_mean_std(s["test"]["RMSE"]), format_mean_std(s["test"]["MASE"], ".4f"),
                 format_mean_std(s["cost"]["fit_time_s"], ",.1f"), format_mean_std(s["cost"]["best_iteration"], ",.0f"),
                 format_mean_std(s["cost"]["n_parameters"], ",.0f")])
    return (f"### {result['experiment']} — {result['model']} ({result['device']}, tham số: {result['params_source']})\n\n"
            + markdown_table(["Seed", "Val WMAE", "Val MASE", "Test WMAE", "Test MAE", "Test RMSE", "Test MASE",
                              "Fit (s)", "Best iter", "Số tham số"], rows,
                             [":---"] + ["---:"] * 9))


def summary_markdown(results: list, hardware_text: str) -> str:
    """Bảng tổng hợp mean ± std cho mọi mô hình."""
    rows = []
    for r in results:
        s = r["summary"]
        rows.append([r["model"], r["params_source"], len(r["seeds"]),
                     format_mean_std(s["val"]["WMAE"]), format_mean_std(s["test"]["WMAE"]),
                     format_mean_std(s["test"]["MAE"]), format_mean_std(s["test"]["RMSE"]),
                     format_mean_std(s["test"]["MASE"], ".4f"),
                     format_mean_std(s["cost"]["fit_time_s"], ",.1f"),
                     f"{s['cost']['inference_time_s']['mean']:.2f}",
                     format_mean_std(s["cost"]["n_parameters"], ",.0f")])
    table = markdown_table(
        ["Mô hình", "Tham số", "#Seed", "Val WMAE", "Test WMAE", "Test MAE", "Test RMSE", "Test MASE",
         "Fit time (s)", "Inference (s)", "Số tham số"],
        rows, [":---", ":---:", ":---:"] + ["---:"] * 8,
    )
    details = "\n\n".join(experiment_markdown(r) for r in results)
    return ("# Kết quả thực nghiệm đa seed (Mean ± Std, ddof = 1)\n\n"
            "Số tham số = tổng số lá (giá trị dự báo) + tổng số node split (cặp đặc trưng/ngưỡng). "
            "MASE dùng mẫu số Seasonal Naive (m = 52) tính trên tập train.\n\n"
            "## Tổng hợp\n\n" + table + "\n\n## Chi tiết từng seed\n\n" + details +
            "\n\n## Cấu hình phần cứng\n\n```\n" + hardware_text + "\n```\n")


def main():
    setup_console()
    parser = argparse.ArgumentParser(description="Multi-seed experiments (mean ± std) cho các mô hình GBDT")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Ghi đè seeds.values trong config")
    parser.add_argument("--quick", action="store_true", help="Chạy thử nhanh: 10%% chuỗi, tối đa 300 cây")
    args = parser.parse_args()

    df = load_processed_train()
    if args.quick:
        df = sample_series(df, 0.1, seed=0)

    out_root = get_result_dir("multiseed")
    hardware = save_hardware_info(HARDWARE_INFO_PATH)
    hardware_text = format_hardware_report(hardware)
    print(hardware_text)

    results = []
    for name in args.configs:
        cfg = load_experiment_config(name)
        seeds = args.seeds or cfg["seeds"]["values"]
        if len(seeds) < 3:
            print(f"[CẢNH BÁO] Chỉ có {len(seeds)} seed — yêu cầu tối thiểu 3 seed để báo cáo mean ± std.")
        results.append(run_experiment(cfg, df, seeds, quick=args.quick))

    suffix = "_quick" if args.quick else ""
    # Gộp kết quả đã có của các thí nghiệm không chạy lần này (giữ thứ tự DEFAULT_CONFIGS)
    if not args.quick:
        for name in DEFAULT_CONFIGS:
            if name in args.configs:
                continue
            prev = load_json(multiseed_dir(load_experiment_config(name)) / "multiseed_results.json")
            if prev:
                results.append(prev)
        order = {n: i for i, n in enumerate(DEFAULT_CONFIGS)}
        results.sort(key=lambda r: order.get(r["experiment"], len(order)))
    save_json({"results": results, "hardware": hardware}, out_root / f"multiseed_summary{suffix}.json")
    md = summary_markdown(results, hardware_text)
    (out_root / f"multiseed_summary{suffix}.md").write_text(md, encoding="utf-8")
    print("\n" + md.split("## Chi tiết")[0])
    print(f"[Multi-seed] Đã lưu kết quả vào: {out_root}")


if __name__ == "__main__":
    main()
