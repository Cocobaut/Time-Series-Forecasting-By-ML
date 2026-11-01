# So sánh Hyperparameter Tuning (ngân sách đồng nhất)

- Ngân sách mỗi mô hình: **50 trials Optuna** hoặc **30 phút** (điều kiện nào đến trước), tối đa 1500 cây/trial, early stopping 50 vòng.
- Cùng tập validation thời gian Q4/2011, cùng hàm mục tiêu WMAE, cùng TPESampler(seed).

## Kết quả

| Mô hình | Thiết bị | Best Val WMAE | Số trial | Thời gian (phút) | Best iter | Siêu tham số tốt nhất |
| :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| catboost | gpu | 2,636.53 | 50 | 17.5 | 426 | learning_rate=0.05131, max_depth=8, subsample=0.643, colsample_bytree=0.6877, reg_alpha=0.02998, reg_lambda=0.001174 |
| lightgbm | cpu | 2,108.84 | 34 | 30.7 | 1500 | learning_rate=0.05716, num_leaves=90, subsample=0.7948, colsample_bytree=0.6698, reg_alpha=0.03444, reg_lambda=0.001671 |
| xgboost | gpu | 1,723.67 | 39 | 30.6 | 1486 | learning_rate=0.02369, max_depth=10, subsample=0.8928, colsample_bytree=0.8395, reg_alpha=0.004208, reg_lambda=0.004207 |

## Không gian siêu tham số

| Siêu tham số | catboost | lightgbm | xgboost |
| :--- | :---: | :---: | :---: |
| `colsample_bytree` | [0.6, 1.0] | [0.6, 1.0] | [0.6, 1.0] |
| `learning_rate` | [0.01, 0.1] (log) | [0.01, 0.1] (log) | [0.01, 0.1] (log) |
| `max_depth` | [4, 8] | — | [4, 10] |
| `num_leaves` | — | [15, 127] | — |
| `reg_alpha` | [0.001, 10.0] (log) | [0.001, 10.0] (log) | [0.001, 10.0] (log) |
| `reg_lambda` | [0.001, 10.0] (log) | [0.001, 10.0] (log) | [0.001, 10.0] (log) |
| `subsample` | [0.6, 1.0] | [0.6, 1.0] | [0.6, 1.0] |

> CatBoost: `max_depth` → `depth`, `reg_lambda` → `l2_leaf_reg`; không có L1 trên lá nên `reg_alpha` được lấy mẫu nhưng không dùng; `colsample_bytree` → `rsm` chỉ áp dụng trên CPU.
