"""
Lớp giao tiếp thống nhất cho 3 mô hình GBDT (LightGBM, XGBoost, CatBoost):
- chọn thiết bị (GPU/CPU), ghép siêu tham số + seed,
- chuẩn bị ma trận đặc trưng đúng kiểu dữ liệu của từng thư viện,
- huấn luyện với early stopping + sample weight WMAE, dự báo,
- ánh xạ không gian siêu tham số chung (Config/experiments/*.toml) sang tên riêng của thư viện.
"""
import warnings
import numpy as np
import pandas as pd

SUPPORTED_MODELS = ("lightgbm", "xgboost", "catboost")

_SEED_KEY = {"lightgbm": "random_state", "xgboost": "random_state", "catboost": "random_seed"}
_N_ESTIMATORS_KEY = {"lightgbm": "n_estimators", "xgboost": "n_estimators", "catboost": "iterations"}
_DEVICE_CACHE = {}


def _check(model_name: str):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Mô hình không hỗ trợ: {model_name}. Chọn một trong {SUPPORTED_MODELS}")


# =============================================================================
# THIẾT BỊ HUẤN LUYỆN
# =============================================================================

def _device_params(model_name: str, device: str) -> dict:
    if model_name == "lightgbm":
        return {"device": "gpu"} if device == "gpu" else {"device": "cpu", "n_jobs": -1}
    if model_name == "xgboost":
        return {"device": "cuda"} if device == "gpu" else {"device": "cpu", "n_jobs": -1}
    return {"task_type": "GPU"} if device == "gpu" else {"task_type": "CPU", "thread_count": -1}


def resolve_device(model_name: str, requested: str = "auto") -> str:
    """
    Trả về "gpu" hoặc "cpu". Với "auto": thử huấn luyện nhanh trên GPU, lỗi thì lùi về CPU.
    Kết quả được cache theo mô hình.
    """
    _check(model_name)
    requested = (requested or "auto").lower()
    if requested in ("cpu", "gpu"):
        return requested
    if model_name in _DEVICE_CACHE:
        return _DEVICE_CACHE[model_name]

    X = pd.DataFrame({"a": np.random.rand(64), "b": np.random.rand(64)})
    y = X["a"] * 3.0
    device = "cpu"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model_name == "lightgbm":
                from lightgbm import LGBMRegressor
                LGBMRegressor(n_estimators=2, verbose=-1, device="gpu").fit(X, y)
            elif model_name == "xgboost":
                import xgboost as xgb
                xgb.XGBRegressor(n_estimators=2, device="cuda", tree_method="hist").fit(X, y)
            else:
                import catboost as cb
                cb.CatBoostRegressor(iterations=2, task_type="GPU", verbose=0, allow_writing_files=False).fit(X, y)
        device = "gpu"
    except Exception:
        device = "cpu"
    _DEVICE_CACHE[model_name] = device
    return device


# =============================================================================
# SIÊU THAM SỐ
# =============================================================================

def map_unified_params(model_name: str, unified: dict, device: str = "cpu") -> dict:
    """
    Ánh xạ siêu tham số dùng tên chung sang tên riêng của thư viện:
      learning_rate | num_leaves | max_depth | subsample | colsample_bytree | reg_alpha | reg_lambda
    """
    _check(model_name)
    out = {}
    for key, value in unified.items():
        if model_name == "lightgbm":
            out[key] = value
            if key == "subsample":
                out["subsample_freq"] = 1
        elif model_name == "xgboost":
            out["max_leaves" if key == "num_leaves" else key] = value
        else:  # catboost
            if key == "learning_rate":
                out["learning_rate"] = value
            elif key == "max_depth":
                out["depth"] = int(value)
            elif key == "subsample":
                out["bootstrap_type"] = "Bernoulli"
                out["subsample"] = value
            elif key == "reg_lambda":
                out["l2_leaf_reg"] = value
            elif key == "colsample_bytree":
                if device == "cpu":
                    out["rsm"] = value
            # reg_alpha (L1 trên lá) và num_leaves không có tương đương trong CatBoost (cây đối xứng)
    return out


def suggest_unified_params(trial, search_space: dict) -> dict:
    """Lấy mẫu một bộ siêu tham số (tên chung) từ không gian tìm kiếm trong config bằng Optuna."""
    params = {}
    for name, spec in search_space.items():
        low, high = spec["low"], spec["high"]
        if isinstance(low, int) and isinstance(high, int) and not spec.get("log", False) and "step" not in spec:
            params[name] = trial.suggest_int(name, low, high)
        else:
            params[name] = trial.suggest_float(name, float(low), float(high), log=bool(spec.get("log", False)))
    return params


def build_params(model_name: str, base_params: dict, seed: int, device: str,
                 tuned_unified: dict = None, n_estimators: int = None) -> dict:
    """Ghép tham số cố định trong config + tham số đã tuning + seed + thiết bị."""
    _check(model_name)
    params = dict(base_params)
    if tuned_unified:
        params.update(map_unified_params(model_name, tuned_unified, device))
    if n_estimators is not None:
        params[_N_ESTIMATORS_KEY[model_name]] = int(n_estimators)
    params[_SEED_KEY[model_name]] = int(seed)
    params.update(_device_params(model_name, device))
    if model_name == "catboost":
        params["allow_writing_files"] = False      # không sinh thư mục catboost_info/
        if device == "gpu":
            params.pop("rsm", None)
    return params


# =============================================================================
# DỮ LIỆU, HUẤN LUYỆN, DỰ BÁO
# =============================================================================

def category_levels(df: pd.DataFrame, cat_cols: list) -> dict:
    """Tập giá trị của từng cột phân loại (cố định để train/val/test có cùng mã category)."""
    return {c: sorted(df[c].unique().tolist()) for c in cat_cols if c in df.columns}


def prepare_X(model_name: str, df: pd.DataFrame, feature_cols: list, cat_cols: list, levels: dict = None) -> pd.DataFrame:
    """Chuẩn bị ma trận đặc trưng đúng định dạng từng thư viện."""
    _check(model_name)
    X = df[feature_cols].copy()
    for c in cat_cols:
        if c not in X.columns:
            continue
        if model_name == "xgboost":
            cats = (levels or {}).get(c)
            X[c] = pd.Categorical(X[c], categories=cats) if cats is not None else X[c].astype("category")
        else:
            X[c] = X[c].astype(int)
    return X


def fit_model(model_name: str, params: dict, X_tr, y_tr, w_tr, X_es, y_es, w_es,
              cat_cols: list, early_stopping_rounds: int = 100):
    """Huấn luyện mô hình với sample weight WMAE và early stopping trên tập (X_es, y_es)."""
    _check(model_name)
    cat_cols = [c for c in cat_cols if c in X_tr.columns]

    if model_name == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_es, y_es)], eval_sample_weight=[w_es],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        return model

    if model_name == "xgboost":
        import xgboost as xgb
        model = xgb.XGBRegressor(**params, early_stopping_rounds=early_stopping_rounds)
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_es, y_es)], sample_weight_eval_set=[w_es], verbose=False)
        return model

    import catboost as cb
    train_pool = cb.Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_cols)
    es_pool = cb.Pool(X_es, y_es, weight=w_es, cat_features=cat_cols)
    model = cb.CatBoostRegressor(**params, early_stopping_rounds=early_stopping_rounds)
    model.fit(train_pool, eval_set=es_pool, use_best_model=True)
    return model


def predict(model, X, clip_negative: bool = True) -> np.ndarray:
    """Dự báo, tùy chọn chặn dưới 0 (Zero-Clipping)."""
    preds = np.asarray(model.predict(X), dtype=float)
    return np.clip(preds, 0.0, None) if clip_negative else preds


def best_iteration(model_name: str, model) -> int:
    """Số vòng lặp tối ưu sau early stopping."""
    if model_name == "lightgbm":
        return int(getattr(model, "best_iteration_", 0) or model.n_estimators)
    if model_name == "xgboost":
        it = getattr(model, "best_iteration", None)
        return int(it + 1) if it is not None else int(model.n_estimators)
    return int(model.get_best_iteration() + 1) if model.get_best_iteration() is not None else int(model.tree_count_)
