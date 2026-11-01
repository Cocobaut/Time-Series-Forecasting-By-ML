# 📊 BẢNG ĐỐI SÁNH TRỌNG TÂM (tự động sinh bởi `Src/Evaluation/generate_benchmark_tables.py`)

## 📊 BẢNG 1: Hướng 1 — Cải tiến Machine Learning & Ensembling (P1, P2, P3)

MASE dùng mẫu số Seasonal Naive (m = 52) tính trên tập train. Benchmark là tất định (std = 0); LightGBM L2/L1 gốc không bật bagging; độ lệch nhỏ giữa các seed của chúng đến từ tính không tất định của huấn luyện đa luồng, còn XGBoost có thêm subsample 0.8. Bảng 3 dùng dự báo của seed đầu tiên (42) nên số P3 có thể khác nhẹ so với mean ở Bảng 1.

**Validation Q4/2011** (3 seed: 42, 123, 2024; mean ± std, ddof = 1)

| Phiên bản thực nghiệm | Ý tưởng cải tiến trọng tâm | WMAE (USD) ↓ | MAE (USD) ↓ | RMSE (USD) ↓ | MASE ↓ | MAPE (%) ↓ | R² ↑ | Ghi chú & Đóng góp |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Benchmark Gốc | Seasonal Naive (lag 52) | 2,294.98 ± 0.00 | 2,162.98 ± 0.00 | 4,873.20 ± 0.00 | 1.099 ± 0.000 | 41.6 ± 0.0 | 0.9653 ± 0.0000 | Mốc chuẩn cơ sở: lặp lại doanh số cùng kỳ năm trước |
| Benchmark Gốc | Moving Average 4 tuần | 4,424.14 ± 0.00 | 3,165.86 ± 0.00 | 10,697.49 ± 0.00 | 1.609 ± 0.000 | 45.6 ± 0.0 | 0.8329 ± 0.0000 | Mốc chuẩn cơ sở: trung bình 4 tuần gần nhất |
| Linear Baseline | Ridge (alpha = 0.1) với lag features | 3,998.28 ± 0.00 | 3,060.36 ± 0.00 | 7,655.09 ± 0.00 | 1.556 ± 0.000 | 137.5 ± 0.0 | 0.9144 ± 0.0000 | Hồi quy L2 trên thang tương đối y / quy mô chuỗi, lag + one-hot |
| Linear Baseline | Lasso (alpha = 0.001) với lag features | 3,888.46 ± 0.00 | 2,796.00 ± 0.00 | 7,573.15 ± 0.00 | 1.421 ± 0.000 | 82.5 ± 0.0 | 0.9162 ± 0.0000 | Hồi quy L1 (chọn biến) trên thang tương đối y / quy mô chuỗi |
| ML Baseline | LightGBM mặc định (L2 loss) | 2,295.66 ± 31.38 | 1,875.89 ± 4.38 | 4,948.02 ± 74.16 | 0.953 ± 0.002 | 141.1 ± 24.1 | 0.9642 ± 0.0011 | Loss L2, tham số thô |
| ML Baseline | XGBoost mặc định (L1 loss) | 1,906.48 ± 54.88 | 1,688.64 ± 20.66 | 4,588.58 ± 138.61 | 0.858 ± 0.011 | 164.7 ± 42.6 | 0.9692 ± 0.0019 | Loss L1 + trọng số WMAE |
| Giai đoạn P1 | LightGBM L1 Loss + Zero-Clipping | 2,226.32 ± 10.27 | 1,884.91 ± 9.32 | 5,688.75 ± 184.06 | 0.958 ± 0.005 | 122.4 ± 7.1 | 0.9527 ± 0.0031 | Hàm mất mát L1 + chặn dự báo âm |
| Giai đoạn P2 | Weighted Duo Blend + Holiday Multiplier | 1,887.31 ± 51.29 | 1,659.43 ± 16.18 | 4,433.23 ± 101.99 | 0.843 ± 0.008 | 159.2 ± 38.3 | 0.9713 ± 0.0013 | Blend XGB + LGB (trọng số tối ưu WMAE) + bù đỉnh Thanksgiving α = 1.02 (w_XGB ≈ 0.84) |
| Giai đoạn P3 | LightGBM đơn + Group Stats + Optuna | 2,095.66 ± 43.55 | 1,818.70 ± 22.13 | 5,090.22 ± 98.06 | 0.924 ± 0.011 | 177.8 ± 5.4 | 0.9621 ± 0.0015 | Mô hình đơn: 8 đặc trưng thống kê nhóm + siêu tham số Optuna |
| **Giai đoạn P3 [BEST]** | Group Stats + Optuna + Super Blend + Multiplier | **1,856.28 ± 38.24** | **1,661.43 ± 16.39** | **4,542.16 ± 102.49** | **0.844 ± 0.008** | 168.2 ± 31.8 | 0.9699 ± 0.0014 | Optuna (TPE) + thống kê nhóm Store×Dept + Super Blend (w_XGB ≈ 0.70) |

**Holdout Test 2012** (3 seed: 42, 123, 2024; mean ± std, ddof = 1)

| Phiên bản thực nghiệm | Ý tưởng cải tiến trọng tâm | WMAE (USD) ↓ | MAE (USD) ↓ | RMSE (USD) ↓ | MASE ↓ | MAPE (%) ↓ | R² ↑ | Ghi chú & Đóng góp |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Benchmark Gốc | Seasonal Naive (lag 52) | 1,775.43 ± 0.00 | 1,768.31 ± 0.00 | 3,784.16 ± 0.00 | 0.899 ± 0.000 | 317.6 ± 0.0 | 0.9707 ± 0.0000 | Mốc chuẩn cơ sở: lặp lại doanh số cùng kỳ năm trước |
| Benchmark Gốc | Moving Average 4 tuần | 2,006.54 ± 0.00 | 1,941.02 ± 0.00 | 4,847.24 ± 0.00 | 0.987 ± 0.000 | 612.7 ± 0.0 | 0.9520 ± 0.0000 | Mốc chuẩn cơ sở: trung bình 4 tuần gần nhất |
| Linear Baseline | Ridge (alpha = 0.1) với lag features | 1,638.50 ± 0.00 | 1,641.54 ± 0.00 | 3,589.86 ± 0.00 | 0.834 ± 0.000 | 1494.2 ± 0.0 | 0.9737 ± 0.0000 | Hồi quy L2 trên thang tương đối y / quy mô chuỗi, lag + one-hot |
| Linear Baseline | Lasso (alpha = 0.001) với lag features | 1,564.00 ± 0.00 | 1,537.42 ± 0.00 | 3,445.95 ± 0.00 | 0.781 ± 0.000 | 267.0 ± 0.0 | 0.9757 ± 0.0000 | Hồi quy L1 (chọn biến) trên thang tương đối y / quy mô chuỗi |
| ML Baseline | LightGBM mặc định (L2 loss) | 1,300.67 ± 10.33 | 1,279.49 ± 9.90 | 2,711.35 ± 13.02 | 0.650 ± 0.005 | 659.2 ± 37.2 | 0.9850 ± 0.0001 | Loss L2, tham số thô |
| ML Baseline | XGBoost mặc định (L1 loss) | 1,280.85 ± 2.48 | 1,255.05 ± 4.83 | 2,769.68 ± 23.48 | 0.638 ± 0.002 | 1033.3 ± 141.5 | 0.9843 ± 0.0003 | Loss L1 + trọng số WMAE |
| Giai đoạn P1 | LightGBM L1 Loss + Zero-Clipping | 1,304.64 ± 11.30 | 1,283.12 ± 10.67 | 2,854.91 ± 34.23 | 0.652 ± 0.005 | 526.4 ± 45.5 | 0.9833 ± 0.0004 | Hàm mất mát L1 + chặn dự báo âm |
| **Giai đoạn P2 [BEST]** | Weighted Duo Blend + Holiday Multiplier | **1,264.39 ± 3.04** | **1,239.44 ± 4.91** | **2,717.99 ± 18.45** | **0.630 ± 0.002** | 974.8 ± 126.2 | 0.9849 ± 0.0002 | Blend XGB + LGB (trọng số tối ưu WMAE) + bù đỉnh Thanksgiving α = 1.02 (w_XGB ≈ 0.84) |
| Giai đoạn P3 | LightGBM đơn + Group Stats + Optuna | 1,327.43 ± 4.81 | 1,305.20 ± 6.23 | 2,868.39 ± 9.77 | 0.663 ± 0.003 | 712.9 ± 58.8 | 0.9832 ± 0.0001 | Mô hình đơn: 8 đặc trưng thống kê nhóm + siêu tham số Optuna |
| Giai đoạn P3 | Group Stats + Optuna + Super Blend + Multiplier | 1,268.98 ± 0.61 | 1,244.08 ± 1.54 | 2,751.90 ± 13.26 | 0.632 ± 0.001 | 941.4 ± 138.3 | 0.9845 ± 0.0001 | Optuna (TPE) + thống kê nhóm Store×Dept + Super Blend (w_XGB ≈ 0.70) |

---

## 📊 BẢNG 2: Hướng 2 — Hierarchical Reconciliation (tầng lá: XGBoost L1 (ML Baseline))

Cây 1 Total + 45 Store + 3,331 Store×Dept. Dự báo cơ sở Level 0/1 từ mô hình LightGBM tổng hợp riêng. W của WLS/MinT từ phần dư `train`. Dòng in đậm: phương pháp nhất quán có WMAE tầng lá thấp nhất.

**Holdout Test 2012**

| Phương pháp phân cấp | Coherency Gap (Store≠ΣDept / Total≠ΣStore) | WMAE L0 Total (USD) | WMAE L1 Store (USD) | WMAE L2 Store×Dept (USD) | Nhận xét |
| :--- | :---: | ---: | ---: | ---: | :--- |
| Base Forecasts (không hòa giải) | 1.89% / 0.63% | 1,386,746 | 43,777 | 1,283.20 | Dự báo độc lập từng tầng, vi phạm tính nhất quán |
| **Bottom-Up (BU)** | 0% (nhất quán) | **1,287,241** | **40,687** | **1,283.20** | Giữ nguyên tầng lá, cộng dồn lên Store/Total |
| Top-Down (TD, tỷ trọng lịch sử) | 0% (nhất quán) | 1,386,746 | 75,175 | 2,625.24 | Phân rã Total theo tỷ trọng lịch sử |
| OLS (W = I) | 0% (nhất quán) | 1,383,354 | 45,667 | 1,354.25 | Chiếu trực giao lên không gian nhất quán (W = I) |
| WLS structural (W = diag(S·1)) | 0% (nhất quán) | 1,291,287 | 41,511 | 1,304.66 | Trọng số theo số chuỗi lá dưới mỗi nút |
| WLS variance (W = diag(σ²)) | 0% (nhất quán) | 1,270,875 | 40,369 | 1,286.97 | Trọng số theo phương sai phần dư từng chuỗi |
| MinT-shrink (Schäfer–Strimmer) | 0% (nhất quán) | 1,257,529 | 40,112 | 1,287.14 | Hiệp phương sai phần dư co rút (Schäfer–Strimmer) |

**Validation Q4/2011**

| Phương pháp phân cấp | Coherency Gap (Store≠ΣDept / Total≠ΣStore) | WMAE L0 Total (USD) | WMAE L1 Store (USD) | WMAE L2 Store×Dept (USD) | Nhận xét |
| :--- | :---: | ---: | ---: | ---: | :--- |
| Base Forecasts (không hòa giải) | 4.07% / 1.05% | 3,163,248 | 96,569 | 1,969.27 | Dự báo độc lập từng tầng, vi phạm tính nhất quán |
| **Bottom-Up (BU)** | 0% (nhất quán) | **1,243,676** | **50,124** | **1,969.27** | Giữ nguyên tầng lá, cộng dồn lên Store/Total |
| Top-Down (TD, tỷ trọng lịch sử) | 0% (nhất quán) | 3,163,248 | 107,067 | 5,127.68 | Phân rã Total theo tỷ trọng lịch sử |
| OLS (W = I) | 0% (nhất quán) | 3,164,417 | 96,738 | 2,418.78 | Chiếu trực giao lên không gian nhất quán (W = I) |
| WLS structural (W = diag(S·1)) | 0% (nhất quán) | 2,525,903 | 72,483 | 2,155.38 | Trọng số theo số chuỗi lá dưới mỗi nút |
| WLS variance (W = diag(σ²)) | 0% (nhất quán) | 1,662,219 | 55,760 | 2,024.27 | Trọng số theo phương sai phần dư từng chuỗi |
| MinT-shrink (Schäfer–Strimmer) | 0% (nhất quán) | 1,687,023 | 56,791 | 2,038.99 | Hiệp phương sai phần dư co rút (Schäfer–Strimmer) |

---

## 📊 BẢNG 3: Hướng 3 — Kết hợp P3 Best Blend + Hierarchical Reconciliation

[FINAL] = cấu hình nhất quán có WMAE Validation tầng lá thấp nhất (chọn trên Val, báo cáo thêm Test).

| Cấu hình tích hợp | WMAE Val L2 | MAE Val L2 | MASE Val L2 | WMAE Test L2 | WMAE Test L1 Store | Coherency | Ý nghĩa |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :--- |
| 1. P3 Best Blend (Raw) | 1,897.26 | 1,677.24 | 0.852 | 1,268.54 | 43,777 | Lệch Store 3.81% | Dự báo độc lập từng tầng, vi phạm tính nhất quán |
| **2. P3 + Bottom-Up (BU) [FINAL]** | **1,897.26** | **1,677.24** | **0.852** | **1,268.54** | **40,837** | 100% nhất quán | Giữ nguyên tầng lá, cộng dồn lên Store/Total |
| 3. P3 + OLS (W = I) | 2,305.46 | 1,896.95 | 0.964 | 1,335.48 | 45,677 | 100% nhất quán | Chiếu trực giao lên không gian nhất quán (W = I) |
| 4. P3 + WLS structural (W = diag(S·1)) | 2,060.54 | 1,759.20 | 0.894 | 1,288.71 | 41,811 | 100% nhất quán | Trọng số theo số chuỗi lá dưới mỗi nút |
| 5. P3 + WLS variance (W = diag(σ²)) | 1,952.03 | 1,703.72 | 0.866 | 1,272.76 | 40,599 | 100% nhất quán | Trọng số theo phương sai phần dư từng chuỗi |
| 6. P3 + MinT-shrink (Schäfer–Strimmer) | 1,953.17 | 1,694.22 | 0.861 | 1,272.77 | 40,417 | 100% nhất quán | Hiệp phương sai phần dư co rút (Schäfer–Strimmer) |

---

### Bảng phụ A: Rolling Origin Cross-Validation (expanding window, 4 fold × 13 tuần; mean ± std qua fold)

| Mô hình | Feature mode | #Fold | MAE | RMSE | MASE | WMAE | WMAE từng fold |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: | :--- |
| catboost | horizon_safe | 4 | 2,204.03 ± 443.52 | 4,678.50 ± 1,309.21 | 1.072 ± 0.235 | 2,273.91 ± 526.09 | 2,352 / 1,850 / 2,987 / 1,905 |
| lightgbm | horizon_safe | 4 | 1,882.81 ± 292.84 | 4,680.80 ± 1,836.13 | 0.916 ± 0.164 | 1,963.67 ± 411.77 | 1,952 / 1,588 / 2,540 / 1,775 |
| xgboost | horizon_safe | 4 | 1,882.54 ± 279.81 | 4,795.91 ± 1,929.02 | 0.915 ± 0.148 | 1,982.46 ± 428.65 | 2,025 / 1,633 / 2,572 / 1,700 |

---

### Bảng phụ B: 3 mô hình GBDT — đa seed, thời gian huấn luyện & số tham số

Số tham số = tổng số lá + tổng số node split của toàn bộ cây.

| Mô hình | Thiết bị | Tham số | #Seed | Val WMAE | Test WMAE | Test MAE | Test RMSE | Test MASE | Fit time (s) | Inference (s) | Số tham số |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lightgbm | cpu | tuned | 3 | 2,121.34 ± 87.49 | 1,290.87 ± 4.64 | 1,269.50 ± 5.13 | 2,818.54 ± 11.59 | 0.645 ± 0.003 | 154.2 ± 49.4 | 5.93 | 586,225 ± 183,584 |
| xgboost | gpu | tuned | 3 | 1,785.02 ± 57.25 | 1,264.82 ± 5.31 | 1,237.98 ± 4.76 | 2,751.49 ± 20.51 | 0.629 ± 0.002 | 72.9 ± 9.9 | 0.62 | 2,154,599 ± 263,964 |
| catboost | gpu | tuned | 3 | 2,696.62 ± 52.79 | 1,453.25 ± 22.96 | 1,434.26 ± 19.64 | 3,055.42 ± 34.94 | 0.729 ± 0.010 | 18.0 ± 3.8 | 0.25 | 120,296 ± 28,153 |

---

### Bảng phụ C: Tuning Optuna với ngân sách đồng nhất (50 trials / 30 phút)

| Mô hình | Best Val WMAE | Số trial | Thời gian (phút) | Siêu tham số tốt nhất |
| :--- | ---: | ---: | ---: | :--- |
| catboost | 2,636.53 | 50 | 17.5 | learning_rate=0.0513, max_depth=8, subsample=0.643, colsample_bytree=0.688, reg_alpha=0.03, reg_lambda=0.00117 |
| lightgbm | 2,108.84 | 34 | 30.7 | learning_rate=0.0572, num_leaves=90, subsample=0.795, colsample_bytree=0.67, reg_alpha=0.0344, reg_lambda=0.00167 |
| xgboost | 1,723.67 | 39 | 30.6 | learning_rate=0.0237, max_depth=10, subsample=0.893, colsample_bytree=0.839, reg_alpha=0.00421, reg_lambda=0.00421 |

---

### Bảng phụ D: Độ nhạy của MinT theo nguồn phần dư ước lượng W (WMAE trên Holdout Test 2012)

W: Train = phần dư trong mẫu 52 tuần cuối tập Train (mặc định, giống MinT gốc); W: Val = phần dư ngoài mẫu của Validation Q4/2011. Base và Bottom-Up không dùng W nên giống nhau ở hai cột; chỉ báo cáo Test vì với W từ Val, kết quả trên Val là trong mẫu.

| Tầng lá | Phương pháp | L0 (W: Train) | L1 (W: Train) | L2 (W: Train) | L0 (W: Val) | L1 (W: Val) | L2 (W: Val) | λ shrink (Train / Val) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| XGBoost L1 (ML Baseline) | Base Forecasts (không hòa giải) | 1,386,746 | 43,777 | 1,283.20 | 1,386,746 | 43,777 | 1,283.20 | — |
| XGBoost L1 (ML Baseline) | Bottom-Up (BU) | 1,287,241 | 40,687 | 1,283.20 | 1,287,241 | 40,687 | 1,283.20 | — |
| XGBoost L1 (ML Baseline) | WLS variance (W = diag(σ²)) | 1,270,875 | 40,369 | 1,286.97 | 1,273,052 | 40,454 | 1,286.41 | — |
| XGBoost L1 (ML Baseline) | MinT-shrink (Schäfer–Strimmer) | 1,257,529 | 40,112 | 1,287.14 | 1,266,662 | 40,433 | 1,288.36 | 0.929 / 0.969 |
| P3 Best Blend | Base Forecasts (không hòa giải) | 1,386,746 | 43,777 | 1,268.54 | 1,386,746 | 43,777 | 1,268.54 | — |
| P3 Best Blend | Bottom-Up (BU) | 1,298,480 | 40,837 | 1,268.54 | 1,298,480 | 40,837 | 1,268.54 | — |
| P3 Best Blend | WLS variance (W = diag(σ²)) | 1,275,014 | 40,599 | 1,272.76 | 1,281,917 | 40,706 | 1,271.41 | — |
| P3 Best Blend | MinT-shrink (Schäfer–Strimmer) | 1,264,185 | 40,417 | 1,272.77 | 1,275,060 | 40,728 | 1,274.29 | 0.925 / 0.953 |

---

### Bảng phụ E: Ablation study theo nhóm đặc trưng (XGBoost L1 (ML Baseline params), 3 seed)

Δ = thay đổi Val WMAE so với cấu hình liền trước (âm = tốt hơn).

| Cấu hình | Nhóm đặc trưng (cộng dồn) | #Đặc trưng | Val WMAE | Δ Val WMAE | Val MASE | Test WMAE | Test MASE |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | Lags | 12 | 3,162.87 ± 42.88 | — | 1.191 ± 0.016 | 1,359.98 ± 6.20 | 0.676 ± 0.002 |
| C2 | + Rolling | 18 | 3,060.47 ± 32.80 | -102.39 | 1.147 ± 0.014 | 1,351.88 ± 1.94 | 0.671 ± 0.001 |
| C3 | + Calendar & holiday flags | 33 | 1,856.31 ± 43.36 | -1,204.16 | 0.825 ± 0.010 | 1,267.93 ± 2.95 | 0.630 ± 0.001 |
| C4 | + Exogenous (bộ đầy đủ) | 44 | 1,900.69 ± 48.32 | +44.37 | 0.854 ± 0.018 | 1,281.69 ± 3.66 | 0.638 ± 0.002 |
| C5 | + Target Group Stats (P3) | 52 | 1,838.52 ± 2.46 | -62.17 | 0.844 ± 0.005 | 1,311.60 ± 3.06 | 0.653 ± 0.001 |

**Top-10 feature importance** (cấu hình C5, theo total gain; hình: `results/06_ablation/feature_importance.png`)

| # | Đặc trưng | Gain (%) | Split count (%) |
| ---: | :--- | ---: | ---: |
| 1 | sales_lag_1 | 33.03% | 3.71% |
| 2 | Dept | 12.18% | 17.53% |
| 3 | Store | 9.96% | 16.83% |
| 4 | sales_rolling_mean_4 | 8.47% | 1.61% |
| 5 | store_dept_median | 8.32% | 1.06% |
| 6 | sales_growth_ratio_4 | 2.49% | 5.81% |
| 7 | sales_ratio_to_store_dept_mean | 1.58% | 3.21% |
| 8 | Week | 1.51% | 2.45% |
| 9 | sales_rolling_std_4 | 1.49% | 3.72% |
| 10 | store_dept_mean | 1.47% | 1.01% |

---

### Cấu hình phần cứng

| Thành phần | Chi tiết |
| :--- | :--- |
| CPU | 12th Gen Intel(R) Core(TM) i7-12700H (14 nhân / 20 luồng) |
| RAM | 31.71 GB |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB) |
| Hệ điều hành | Windows 10 (10.0.26300) |
| Python | 3.10.21 |
| Thư viện | numpy 2.2.6, pandas 2.3.3, sklearn 1.7.2, lightgbm 4.7.0, xgboost 3.2.0, catboost 1.2.10, optuna 5.0.0, scipy 1.15.3 |
