"""
GIAI ĐOẠN 3 — TUNING SIÊU THAM SỐ VỚI NGÂN SÁCH ĐỒNG NHẤT

Mọi mô hình được tuning bằng Optuna (TPESampler, cùng sampler seed) với CÙNG:
  - ngân sách: tuning.n_trials (50) HOẶC tuning.timeout_minutes (30 phút), điều kiện nào đến trước,
  - tập validation thời gian: Q4/2011 (train < 2011-10-01),
  - hàm mục tiêu: WMAE trên validation (dự báo đã zero-clip),
  - không gian siêu tham số chung trong [tuning.search_space] của Config/experiments/*.toml.

Đầu ra (mỗi thí nghiệm): results/03_tuning/<thí nghiệm>/best_params.json, trials.csv
Tổng hợp: results/03_tuning/tuning_comparison.md (+ .json)

Cách chạy:
    python Src/Experiments/tune.py                               # cả 3 mô hình, ngân sách đầy đủ
    python Src/Experiments/tune.py --configs xgboost_tuned --n-trials 5 --quick
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

import optuna

from Config import load_experiment_config, get_result_dir
from Metric.metrics import calculate_wmae
from Src.Experiments.common import (
    setup_console, load_processed_train, static_split, get_feature_columns,
    holiday_weights, sample_series, tuning_dir, save_json, load_json, markdown_table,
)
from Src.Experiments import model_zoo
from Src.Utils.system_profiler import Timer, get_hardware_info

DEFAULT_CONFIGS = ["lightgbm_tuned", "xgboost_tuned", "catboost_tuned"]


def tune_experiment(cfg: dict, df: pd.DataFrame, n_trials: int = None, timeout_minutes: float = None,
                    quick: bool = False) -> dict:
    """Chạy Optuna cho một thí nghiệm và lưu bộ siêu tham số tốt nhất."""
    model_name = cfg["experiment"]["model"]
    tcfg = cfg["tuning"]
    data_cfg = cfg.get("data", {})
    feat_cfg = cfg.get("features", {})
    cat_cols = feat_cfg.get("categorical", ["Store", "Dept", "Type"])
    hw = data_cfg.get("holiday_weight", 5.0)
    clip = data_cfg.get("clip_negative", True)

    n_trials = n_trials or tcfg.get("n_trials", 50)
    timeout_minutes = timeout_minutes or tcfg.get("timeout_minutes", 30)
    n_estimators = 200 if quick else tcfg.get("n_estimators", 1500)
    es_rounds = tcfg.get("early_stopping_rounds", 50)
    device = model_zoo.resolve_device(model_name, cfg.get("device", {}).get("type", "auto"))
    seed = cfg["seeds"]["values"][0]

    train_df, val_df, _ = static_split(df, data_cfg)
    feature_cols = get_feature_columns(df, feat_cfg)
    levels = model_zoo.category_levels(df, cat_cols)
    X_tr = model_zoo.prepare_X(model_name, train_df, feature_cols, cat_cols, levels)
    X_val = model_zoo.prepare_X(model_name, val_df, feature_cols, cat_cols, levels)
    y_tr, y_val = train_df["Weekly_Sales"], val_df["Weekly_Sales"]
    w_tr, w_val = holiday_weights(train_df, hw), holiday_weights(val_df, hw)
    val_holiday = val_df["IsHoliday"].to_numpy()

    print("\n" + "=" * 80)
    print(f" OPTUNA TUNING — {cfg['experiment']['name']} ({model_name}, device={device})")
    print(f" Ngân sách: {n_trials} trials hoặc {timeout_minutes} phút | tối đa {n_estimators} cây/trial")
    print(f" Không gian: {tcfg['search_space']}")
    print("=" * 80)

    def objective(trial):
        unified = model_zoo.suggest_unified_params(trial, tcfg["search_space"])
        params = model_zoo.build_params(model_name, cfg["model"]["params"], seed, device,
                                        unified, n_estimators=n_estimators)
        with Timer() as t:
            model = model_zoo.fit_model(model_name, params, X_tr, y_tr, w_tr, X_val, y_val, w_val,
                                        cat_cols, early_stopping_rounds=es_rounds)
        preds = model_zoo.predict(model, X_val, clip)
        score = calculate_wmae(y_val, preds, val_holiday)
        trial.set_user_attr("best_iteration", model_zoo.best_iteration(model_name, model))
        trial.set_user_attr("fit_time_s", t.elapsed)
        print(f"   Trial {trial.number:>3} | WMAE {score:>10,.2f} | iter {trial.user_attrs['best_iteration']:>5} "
              f"| {t.elapsed:,.1f}s | {unified}")
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=tcfg.get("sampler_seed", 42)))
    with Timer() as t_total:
        study.optimize(objective, n_trials=n_trials, timeout=timeout_minutes * 60, show_progress_bar=False)

    best = study.best_trial
    # Chế độ --quick ghi ra thư mục riêng để không ghi đè kết quả tuning thật
    out_dir = tuning_dir(cfg, quick)
    result = {
        "experiment": cfg["experiment"]["name"],
        "model": model_name,
        "device": device,
        "metric": tcfg.get("metric", "WMAE"),
        "best_val_wmae": float(best.value),
        "best_trial": int(best.number),
        "best_iteration": int(best.user_attrs.get("best_iteration", 0)),
        "params": best.params,
        "library_params": model_zoo.map_unified_params(model_name, best.params, device),
        "budget": {"n_trials": n_trials, "timeout_minutes": timeout_minutes,
                   "n_estimators_per_trial": n_estimators, "early_stopping_rounds": es_rounds},
        "n_trials_completed": len(study.trials),
        "total_time_s": t_total.elapsed,
        "search_space": tcfg["search_space"],
        "quick": quick,
    }
    save_json(result, out_dir / "best_params.json")
    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
    print(f"\n[Tuning] Best WMAE = {best.value:,.2f} (trial {best.number}) | {len(study.trials)} trials "
          f"trong {t_total.elapsed / 60:,.1f} phút -> {out_dir / 'best_params.json'}")
    return result


def write_comparison(results: list, suffix: str = ""):
    """Bảng so sánh kết quả tuning giữa các mô hình + bảng không gian siêu tham số."""
    rows = []
    for r in results:
        params = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r["params"].items())
        rows.append([r["model"], r["device"], f"{r['best_val_wmae']:,.2f}", r["n_trials_completed"],
                     f"{r['total_time_s'] / 60:,.1f}", r["best_iteration"], params])
    table = markdown_table(
        ["Mô hình", "Thiết bị", "Best Val WMAE", "Số trial", "Thời gian (phút)", "Best iter", "Siêu tham số tốt nhất"],
        rows, [":---", ":---:", "---:", "---:", "---:", "---:", ":---"],
    )

    space_names = sorted({k for r in results for k in r["search_space"]})
    space_rows = []
    for name in space_names:
        row = [f"`{name}`"]
        for r in results:
            spec = r["search_space"].get(name)
            row.append(f"[{spec['low']}, {spec['high']}]{' (log)' if spec.get('log') else ''}" if spec else "—")
        space_rows.append(row)
    space_table = markdown_table(["Siêu tham số"] + [r["model"] for r in results], space_rows,
                                 [":---"] + [":---:"] * len(results))

    budget = results[0]["budget"]
    md = (
        "# So sánh Hyperparameter Tuning (ngân sách đồng nhất)\n\n"
        f"- Ngân sách mỗi mô hình: **{budget['n_trials']} trials Optuna** hoặc **{budget['timeout_minutes']} phút** "
        f"(điều kiện nào đến trước), tối đa {budget['n_estimators_per_trial']} cây/trial, "
        f"early stopping {budget['early_stopping_rounds']} vòng.\n"
        "- Cùng tập validation thời gian Q4/2011, cùng hàm mục tiêu WMAE, cùng TPESampler(seed).\n\n"
        "## Kết quả\n\n" + table + "\n\n## Không gian siêu tham số\n\n" + space_table + "\n\n"
        "> CatBoost: `max_depth` → `depth`, `reg_lambda` → `l2_leaf_reg`; không có L1 trên lá nên `reg_alpha` "
        "được lấy mẫu nhưng không dùng; `colsample_bytree` → `rsm` chỉ áp dụng trên CPU.\n"
    )
    out_dir = get_result_dir("tuning")
    (out_dir / f"tuning_comparison{suffix}.md").write_text(md, encoding="utf-8")
    save_json({"results": results, "hardware": get_hardware_info()}, out_dir / f"tuning_comparison{suffix}.json")
    print(f"\n[Tuning] Đã lưu bảng so sánh: {out_dir / ('tuning_comparison' + suffix + '.md')}")


def main():
    setup_console()
    parser = argparse.ArgumentParser(description="Optuna tuning với ngân sách đồng nhất cho các mô hình GBDT")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--n-trials", type=int, default=None, help="Ghi đè tuning.n_trials")
    parser.add_argument("--timeout-minutes", type=float, default=None, help="Ghi đè tuning.timeout_minutes")
    parser.add_argument("--quick", action="store_true", help="Chạy thử nhanh: 10%% chuỗi, 200 cây/trial")
    args = parser.parse_args()

    df = load_processed_train()
    if args.quick:
        df = sample_series(df, 0.1, seed=0)

    results = []
    for name in args.configs:
        cfg = load_experiment_config(name)
        results.append(tune_experiment(cfg, df, args.n_trials, args.timeout_minutes, args.quick))

    # Gộp với kết quả tuning đã có của các mô hình không chạy lần này
    for name in DEFAULT_CONFIGS:
        if name in args.configs or args.quick:
            continue
        cfg = load_experiment_config(name)
        prev = load_json(tuning_dir(cfg) / "best_params.json")
        if prev:
            results.append(prev)
    write_comparison(results, suffix="_quick" if args.quick else "")


if __name__ == "__main__":
    main()
