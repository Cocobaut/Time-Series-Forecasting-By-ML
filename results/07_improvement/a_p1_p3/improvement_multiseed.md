# Chuỗi cải tiến trên 3 seed [42, 123, 2024] (Mean ± Std)

### Validation (Q4/2011)

| Phiên bản | Ý tưởng | WMAE | MAE | RMSE | MASE |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Benchmark Gốc | Seasonal Naive (lag 52) | 2,294.98 ± 0.00 | 2,162.98 ± 0.00 | 4,873.20 ± 0.00 | 1.099 ± 0.000 |
| Benchmark Gốc | Moving Average 4 tuần | 4,424.14 ± 0.00 | 3,165.86 ± 0.00 | 10,697.49 ± 0.00 | 1.609 ± 0.000 |
| ML Baseline | LightGBM mặc định (L2 loss) | 2,295.66 ± 31.38 | 1,875.89 ± 4.38 | 4,948.02 ± 74.16 | 0.953 ± 0.002 |
| ML Baseline | XGBoost mặc định (L1 loss) | 1,906.48 ± 54.88 | 1,688.64 ± 20.66 | 4,588.58 ± 138.61 | 0.858 ± 0.011 |
| Giai đoạn P1 | LightGBM L1 Loss + Zero-Clipping | 2,226.32 ± 10.27 | 1,884.91 ± 9.32 | 5,688.75 ± 184.06 | 0.958 ± 0.005 |
| Giai đoạn P2 | Weighted Duo Blend + Holiday Multiplier | 1,887.31 ± 51.29 | 1,659.43 ± 16.18 | 4,433.23 ± 101.99 | 0.843 ± 0.008 |
| Giai đoạn P3 | LightGBM đơn + Group Stats + Optuna | 2,095.66 ± 43.55 | 1,818.70 ± 22.13 | 5,090.22 ± 98.06 | 0.924 ± 0.011 |
| Giai đoạn P3 | Group Stats + Optuna + Super Blend + Multiplier | 1,856.28 ± 38.24 | 1,661.43 ± 16.39 | 4,542.16 ± 102.49 | 0.844 ± 0.008 |

### Holdout Test (2012)

| Phiên bản | Ý tưởng | WMAE | MAE | RMSE | MASE |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Benchmark Gốc | Seasonal Naive (lag 52) | 1,775.43 ± 0.00 | 1,768.31 ± 0.00 | 3,784.16 ± 0.00 | 0.899 ± 0.000 |
| Benchmark Gốc | Moving Average 4 tuần | 2,006.54 ± 0.00 | 1,941.02 ± 0.00 | 4,847.24 ± 0.00 | 0.987 ± 0.000 |
| ML Baseline | LightGBM mặc định (L2 loss) | 1,300.67 ± 10.33 | 1,279.49 ± 9.90 | 2,711.35 ± 13.02 | 0.650 ± 0.005 |
| ML Baseline | XGBoost mặc định (L1 loss) | 1,280.85 ± 2.48 | 1,255.05 ± 4.83 | 2,769.68 ± 23.48 | 0.638 ± 0.002 |
| Giai đoạn P1 | LightGBM L1 Loss + Zero-Clipping | 1,304.64 ± 11.30 | 1,283.12 ± 10.67 | 2,854.91 ± 34.23 | 0.652 ± 0.005 |
| Giai đoạn P2 | Weighted Duo Blend + Holiday Multiplier | 1,264.39 ± 3.04 | 1,239.44 ± 4.91 | 2,717.99 ± 18.45 | 0.630 ± 0.002 |
| Giai đoạn P3 | LightGBM đơn + Group Stats + Optuna | 1,327.43 ± 4.81 | 1,305.20 ± 6.23 | 2,868.39 ± 9.77 | 0.663 ± 0.003 |
| Giai đoạn P3 | Group Stats + Optuna + Super Blend + Multiplier | 1,268.98 ± 0.61 | 1,244.08 ± 1.54 | 2,751.90 ± 13.26 | 0.632 ± 0.001 |
