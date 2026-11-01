# 📑 ĐẶC TẢ KẾ HOẠCH SINH BÁO CÁO CHI TIẾT (REPORT GENERATION PLAN)
## Dự án: Walmart Store Sales Forecasting — Time Series Machine Learning
*Môn học: Machine Learning and Applications (CO5175) — Thạc sĩ Khoa học Máy tính*
*Nhóm thực hiện: Nguyễn Hồ Quang Khải (MSSV: 2352538) — Đào Duy Đạt (MSSV: 2352223)*

---

## I. NGUYÊN TẮC THIẾT KẾ KẾ HOẠCH BÁO CÁO

Kế hoạch này phục vụ **duy nhất mục đích sinh nội dung báo cáo (Report Generation)** đạt chuẩn 8–12 trang (theo Mục 5.1 và Rubric 5.2 của file đặc tả đề tài). 

* **Giả định tiên quyết:** Toàn bộ các module kỹ thuật và thực nghiệm trong `tmp_plan.md` (bao gồm Linear Model, MASE, 3 Random Seeds, Rolling-Origin CV, Ablation Study, Hierarchical Reconciliation và 3 Bảng Đối Sánh) **đã được thực thi và đã có sẵn tệp kết quả trên đĩa**.
* **Đầu ra mục tiêu:** Một hệ thống gồm **22 tệp văn bản `.txt`** được tổ chức phân tầng tại thư mục `Report/txt_sections/`. Mỗi tệp `.txt` chứa đầy đủ văn bản học thuật, công thức toán LaTeX, số liệu thực nghiệm định lượng và bảng biểu (cả định dạng ASCII và mã nguồn LaTeX `\begin{table}`) để biên dịch thẳng ra báo cáo PDF.

---

## II. DANH MỤC CÁC TỆP DỮ LIỆU ĐẦU VÀO CẦN THIẾT (INPUT DATA DEPENDENCIES)

Để sinh ra toàn bộ 22 tệp văn bản báo cáo, trình sinh báo cáo (`generate_report_txt.py`) sẽ đọc dữ liệu từ các tệp nguồn sau:

1. **Dữ liệu phân tích & Tài liệu nền tảng:**
   * `Final_Analyze.md`: Nguồn dữ liệu lý thuyết sâu, phân tích 10 bước EDA, phân tích toán học.
   * `README.md` & `README_analyze_3_model.md`: Thông tin môi trường, cấu trúc dự án, mô tả thành viên.
   * `Config/config.toml` & `Config/experiments/*.toml`: Thông số thiết lập ngày split, siêu tham số.
2. **Số liệu thực nghiệm Baseline & Linear:**
   * `Src/Baseline/Result/baseline_results.txt`: Kết quả Naive, Seasonal Naive, Moving Average.
   * `Src/Baseline/Result/linear_results.txt`: Kết quả Ridge/Lasso với lag features (WMAE, MAE, RMSE, MASE).
3. **Số liệu thực nghiệm Đa Seed, CV & Tài nguyên phần cứng:**
   * `Src/Utils/system_info.json`: Cấu hình CPU, GPU, RAM, OS, thông số mô hình (`num_leaves`, tham số).
   * `Src/Experiments/results/multiseed_summary.json`: Kết quả 3 seeds (`[42, 123, 2024]`) dạng $\text{Mean} \pm \text{Std}$ cho tất cả mô hình, thời gian huấn luyện (`fit_time`) và suy luận (`inference_time`).
   * `Src/Evaluation/Result/time_series_cv_results.txt`: Kết quả Rolling Origin CV 4 folds.
4. **Số liệu Ablation Study & Feature Importance:**
   * `Src/Experiments/results/ablation_results.txt`: Sai số qua 3 tập đặc trưng (Chỉ Lags $\rightarrow$ Lags+Rolling $\rightarrow$ Toàn diện).
   * `Report/figures/feature_importance.png`: Tệp hình ảnh Top-10 đặc trưng quan trọng nhất.
5. **Số liệu Cải tiến (Improvement P1 - P3):**
   * `improve/result/P1.txt`: Số liệu $L_1$ Loss + Zero-Clipping, mức giảm MAPE.
   * `improve/result/P2.txt`: Danh sách 27 Surge Depts, Holiday Multiplier $\alpha=1.02$, tỷ trọng Duo-Blend $77.4\% : 22.6\%$.
   * `improve/result/P3.txt`: Siêu tham số tối ưu Optuna 15 trials, Target Group Stats, kết quả Super Blend.
6. **Số liệu Dự báo phân cấp & 3 Bảng Đối sánh:**
   * `Src/Extension/Result/hierarchical_reconciliation.txt`: Kết quả Base vs Bottom-Up vs Top-Down vs MinT/OLS trên Level 0, 1, 2; tỷ lệ Coherency Gap; và kết quả kết hợp P3 Blend + MinT.
   * `Src/Evaluation/Result/benchmark_3_tables.json`: Dữ liệu số của 3 Bảng đối sánh trọng tâm.

---

## III. ĐẶC TẢ CHI TIẾT TỪNG TỆP TXT BÁO CÁO ĐẦU RA (DETAILED TXT SPECIFICATIONS)

---

### 📂 KHỐI 0: TRANG BÌA & MỤC LỤC

#### 📄 `00_cover_and_toc.txt`
* **Mục tiêu:** Trang 1 của báo cáo (Trang bìa chính quy + Cây mục lục tóm tắt).
* **Tệp đầu vào cần đọc:** `README.md`.
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. Header chuẩn: Trường ĐH Bách Khoa ĐHQG-HCM, Khoa Khoa học và Kỹ thuật Máy tính.
  2. Tên môn học: *Machine Learning and Applications (CO5175)*.
  3. Tên đề tài: *Assignment Report — Topic 4: Time Series Forecasting Using Machine Learning*.
  4. Thông tin nhóm: Nguyễn Hồ Quang Khải (MSSV: 2352538) & Đào Duy Đạt (MSSV: 2352223).
  5. Cây mục lục phân cấp từ Phần 1 đến Phần 6 và Phụ lục A, B kèm số trang dự kiến.
  6. Danh mục 5 bảng đối sánh và 2 hình vẽ chính trong báo cáo.

---

### 📂 KHỐI 1: GIỚI THIỆU (INTRODUCTION) (Trang 2 – 3, ~1.5 đến 2 trang)

#### 📄 `01_introduction/1.1_problem_statement.txt`
* **Mục tiêu:** Mô tả bối cảnh kinh doanh thực tế và phát biểu toán học bài toán.
* **Tệp đầu vào cần đọc:** `Final_Analyze.md` (Mục 1.1, 1.2).
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. Định nghĩa toán học bài toán dự báo đa bước ($H=39$ tuần) cho $S=45$ siêu thị và $D=81$ phòng ban ($N=3,331$ chuỗi thời gian phân cấp): $\hat{y}_{s,d,t+h} = f(\mathbf{x}_{s,d,t+h}, \mathcal{H}_t)$.
  2. Ý nghĩa kinh doanh: Cân đối tồn kho, giảm lãng phí, đáp ứng chuỗi cung ứng trong 4 kỳ nghỉ lễ lớn của Mỹ.
  3. Thách thức kỹ thuật cốt lõi: Ngăn chặn rò rỉ dữ liệu theo thời gian (*Lookahead Bias / Temporal Data Leakage*), xử lý 671 chuỗi thời gian gián đoạn và lệch pha lịch ngày lễ (Holiday Calendar Shift).

#### 📄 `01_introduction/1.2_forecasting_taxonomy.txt`
* **Mục tiêu:** Hệ thống hóa các trường phái dự báo chuỗi thời gian (Tiêu chí Tổng quan & Hiểu bài toán - 15% Rubric).
* **Tệp đầu vào cần đọc:** `Final_Analyze.md` (Mục 1.2), `Specification/ML-Assignment-Topics.pdf`.
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. Bảng phân loại khoa học (Taxonomy) so sánh 3 trường phái lớn:
     * *Mô hình thống kê cổ điển (Naive, S-Naive, Moving Average, ARIMA):* Ưu điểm nhẹ, tính toán tức thì; nhược điểm không nhận biến ngoại sinh, giả định tuyến tính.
     * *Mô hình học máy dạng bảng (Linear Regularized, GBDT: LightGBM, XGBoost, CatBoost):* Ưu điểm nắm bắt phi tuyến, học tương quan chéo đa chuỗi (*Global Model*), tích hợp linh hoạt biến trễ và ngoại sinh; nhược điểm cần Feature Engineering tinh vi.
     * *Mô hình học sâu (LSTM, DeepAR, Transformer):* Ưu điểm tự động trích xuất biểu diễn chuỗi; nhược điểm đòi hỏi dữ liệu cực lớn, chi phí huấn luyện cao, dễ overfit trên dữ liệu bán lẻ dạng bảng có độ nhiễu lớn.

#### 📄 `01_introduction/1.3_contributions.txt`
* **Mục tiêu:** Tuyên bố rõ ràng 4 đóng góp khoa học cốt lõi của nhóm nghiên cứu.
* **Tệp đầu vào cần đọc:** `Final_Analyze.md` (Mục 1.3), `tmp_plan.md`.
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. *Đóng góp 1 (Dữ liệu):* Đề xuất khung lưới Descartes liên tục $\mathcal{G} = \mathcal{I} \times \mathcal{T}_{all}$ giải quyết triệt để lỗi lệch pha trễ (Lag phase shift) và trôi dạt chỉ mục do đứt gãy chuỗi.
  2. *Đóng góp 2 (Đặc trưng):* Thiết kế hệ thống đặc trưng nhân quả nghiêm ngặt (Causal Safe Lags $k \ge 51$, Rolling Window Shift-1, Target Group Statistics đóng băng).
  3. *Đóng góp 3 (Mô hình & Hậu xử lý):* Đồng bộ hàm mất mát $L_1$, bộ lọc Zero-Clipping, thuật toán Duo-Blending và Holiday Multiplier co giãn theo từng ngành hàng.
  4. *Đóng góp 4 (Mở rộng phân cấp):* Hiện thực thuật toán hòa giải phân cấp MinT (Minimum Trace) đảm bảo 100% tính nhất quán toán học từ cấp toàn quốc đến cấp phòng ban.

---

### 📂 KHỐI 2: PHƯƠNG PHÁP & NỀN TẢNG TOÁN HỌC (METHODS) (Trang 3 – 5, ~2.5 đến 3 trang)

#### 📄 `02_methods/2.1_architecture.txt`
* **Mục tiêu:** Mô tả kiến trúc tổng thể toàn hệ thống luồng một chiều (End-to-End Pipeline).
* **Tệp đầu vào cần đọc:** `Final_Analyze.md` (Mục 2.1, 2.2).
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. Sơ đồ khối bằng text / ASCII mô tả 5 tầng: (1) Thu nhận dữ liệu thô $\rightarrow$ (2) Tiền xử lý & Khung lưới Descartes $\rightarrow$ (3) Kỹ thuật đặc trưng nhân quả $\rightarrow$ (4) Huấn luyện mô hình toàn cục $\rightarrow$ (5) Hậu xử lý, Blending & Hòa giải phân cấp MinT.
  2. Nguyên tắc bất biến: Luồng dữ liệu đi một chiều từ quá khứ đến tương lai, chống Lookahead Bias ở mọi khâu.

#### 📄 `02_methods/2.2_leak_free_pipeline.txt`
* **Mục tiêu:** Trình bày chi tiết công thức toán học của pipeline tiền xử lý và đặc trưng.
* **Tệp đầu vào cần đọc:** `Data/Preprocess Data/preprocess.py`, `Final_Analyze.md` (Mục 5.2).
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. Công thức tích Descartes: $\mathcal{G} = \{(s, d, t) \mid (s, d) \in \mathcal{P}, t \in \mathcal{T}_{\text{all}}\}$; quy tắc điền $y=0$ cho tuần trống trong tập huấn luyện.
  2. Toán tử trễ nhân quả (Causal Lag Operator): $h_{i,t}^{(k)} = \mathcal{L}^k(y_{i,t}) = y_{i,t-k}$ với $k \ge 1$. Phân loại rõ trễ an toàn dài hạn $k \in \{51, 52, 53\}$ cho dự báo tương lai 39 tuần.
  3. Toán tử thống kê trượt (Rolling Window Statistics) áp dụng trên chuỗi đã dịch chuyển: $\mu_{i,t}^{(W)} = \frac{1}{W} \sum_{j=1}^W y_{i, t-j}$.
  4. Mã hóa chu kỳ lượng giác sin/cos: $\text{Week\_sin} = \sin\left(\frac{2\pi \cdot \text{Week}}{52}\right)$, $\text{Week\_cos} = \cos\left(\frac{2\pi \cdot \text{Week}}{52}\right)$.
  5. Thống kê nhóm mục tiêu đóng băng (Frozen Target Group Statistics): $\bar{y}_{s,d} = \frac{1}{|T_{\text{train}}|} \sum_{t \in T_{\text{train}}} y_{s,d,t}$ (chỉ tính trên Train, cấm dùng nhãn Validation/Test).

#### 📄 `02_methods/2.3_studied_models.txt`
* **Mục tiêu:** Trình bày nền tảng lý thuyết và công thức toán học của 5 mô hình nghiên cứu.
* **Tệp đầu vào cần đọc:** `Src/Baseline/baseline_models.py`, `Src/Baseline/linear_baseline.py`, mã nguồn GBDT, `Final_Analyze.md` (Mục 3.1, 5.3).
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. *Mô hình cơ sở:* Naive ($y_t = y_{t-1}$), Seasonal Naive ($y_t = y_{t-52}$), Moving Average 4W ($\hat{y}_t = \frac{1}{4} \sum_{k=1}^4 y_{t-k}$).
  2. *Mô hình Tuyến tính Regularized (Ridge/Lasso Baseline):* Hàm mất mát $\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \lambda \|\mathbf{w}\|_2^2$ với đặc trưng trễ và chuẩn hóa StandardScaler.
  3. *Mô hình Gradient Boosting Decision Trees (GBDT):*
     * *XGBoost:* Khai triển Taylor bậc 2 của hàm mất mát và hàm phạt cấu trúc cây $\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum w_j^2$.
     * *LightGBM:* Cơ chế phân nhánh theo lá (Leaf-wise), thuật toán Histogram-based binning và Gradient-based One-Side Sampling (GOSS).
     * *CatBoost:* Cây đối xứng (Oblivious Trees) chống overfit và Ordered Boosting xử lý trực tiếp biến phân loại `Store`, `Dept`, `Type`.
  4. *Mô hình Hòa giải phân cấp (Hierarchical Reconciliation):* Định nghĩa Ma trận cấu trúc tổng hợp (Summing Matrix $\mathbf{S}$) kích thước $3,377 \times 3,331$ biểu diễn quan hệ $\mathbf{y}_t = \mathbf{S} \mathbf{y}_{\text{bottom}, t}$.

#### 📄 `02_methods/2.4_evaluation_metrics.txt`
* **Mục tiêu:** Định nghĩa chuẩn xác các thang đo đánh giá thực nghiệm (20% Rubric).
* **Tệp đầu vào cần đọc:** `Metric/metrics.py`, `Metric/README.md`.
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. *Weighted Mean Absolute Error (WMAE) - Metric chính của cuộc thi:*
     $$\text{WMAE} = \frac{\sum_{i=1}^n w_i |y_i - \hat{y}_i|}{\sum_{i=1}^n w_i}, \quad w_i = \begin{cases} 5 & \text{nếu tuần lễ } (\text{IsHoliday}=1) \\ 1 & \text{nếu tuần thường } (\text{IsHoliday}=0) \end{cases}$$
  2. *Mean Absolute Error (MAE) và Root Mean Squared Error (RMSE).*
  3. *Mean Absolute Scaled Error (MASE) - Metric chuẩn hóa chuỗi thời gian bắt buộc:*
     $$\text{MASE} = \frac{\frac{1}{h} \sum_{t=1}^h |y_t - \hat{y}_t|}{\frac{1}{N - 52} \sum_{t=53}^N |y_t - y_{t-52}|}$$
     Giải thích: MASE $< 1$ chứng minh mô hình vượt trội hơn mô hình Seasonal Naive chuẩn.
  4. *Chỉ số Coherency Gap:* Định lượng tỷ lệ % bất nhất quán giữa dự báo cấp cửa hàng và tổng các phòng ban: $\text{Gap} = \frac{\sum |\hat{y}_{\text{Store}} - \sum \hat{y}_{\text{Dept}}|}{\sum \hat{y}_{\text{Store}}} \times 100\%$.
  5. *Phân tích loại trừ MAPE:* Phân tích hiện tượng chia cho 0 hoặc giá trị tiệm cận 0 khiến MAPE phân kỳ hàng ngàn %, không phản ánh đúng bản chất bài toán.

---

### 📂 KHỐI 3: THỰC NGHIỆM & PHÂN TÍCH KẾT QUẢ (EXPERIMENTS) (Trang 6 – 8, ~2.5 đến 3 trang)

#### 📄 `03_experiments/3.1_experimental_setup.txt`
* **Mục tiêu:** Mô tả chi tiết giao thức phân tách dữ liệu, cấu hình phần cứng và quy chuẩn thực nghiệm.
* **Tệp đầu vào cần đọc:** `Src/Utils/system_info.json`, `Src/Evaluation/Result/time_series_cv_results.txt`, `Config/config.toml`.
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. Thống kê bộ dữ liệu: 421,570 dòng train, 115,064 dòng test, 45 siêu thị, 81 phòng ban, thời gian 05/02/2010 đến 26/10/2012.
  2. Giao thức phân tách dòng thời gian nghiêm ngặt (Temporal Train / Val / Holdout):
     * *Train set:* Trước 01/10/2011 (255,438 dòng)
     * *Validation set (Q4/2011 Holiday):* 01/10/2011 đến 31/12/2011 (38,768 dòng — chứa Thanksgiving, Black Friday, Christmas)
     * *Holdout Test set:* Từ 01/01/2012 trở đi (127,364 dòng)
  3. Giao thức Time-Series Cross-Validation (Rolling Origin Expanding Window 4 folds) để kiểm chứng tính bền vững thời gian.
  4. Giao thức 3 Random Seeds (`[42, 123, 2024]`) để đo đạc phương sai kết quả.
  5. Thông số môi trường phần cứng trích xuất tự động: CPU Model, số nhân/luồng, GPU model, VRAM, dung lượng RAM hệ thống và OS.

#### 📄 `03_experiments/3.2_main_comparison_table.txt`
* **Mục tiêu:** Bảng đối sánh tổng hợp duy nhất toàn bộ các mô hình nghiên cứu (Mục 3.2 Báo cáo).
* **Tệp đầu vào cần đọc:** `Src/Experiments/results/multiseed_summary.json`, `Src/Baseline/Result/linear_results.txt`, `Src/Baseline/Result/baseline_results.txt`.
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. Bảng ASCII và mã bảng LaTeX `\begin{table}` chứa toàn bộ 7 mô hình:
     * Naive Baseline
     * Seasonal Naive
     * Moving Average 4W
     * Linear Regularized (Ridge)
     * CatBoost Baseline
     * LightGBM Baseline
     * XGBoost Baseline
  2. Các cột số liệu bắt buộc:
     * WMAE ($\text{Mean} \pm \text{Std}$)
     * MAE ($\text{Mean} \pm \text{Std}$)
     * RMSE ($\text{Mean} \pm \text{Std}$)
     * MASE ($\text{Mean} \pm \text{Std}$)
     * Thời gian huấn luyện (Fit Time - giây)
     * Thời gian suy luận (Inference Time - giây)
     * Số tham số mô hình / Kích thước bộ nhớ

#### 📄 `03_experiments/3.3_result_analysis.txt`
* **Mục tiêu:** Phân tích biện giải khoa học cho kết quả ở Bảng 3.2 (30% Rubric).
* **Tệp đầu vào cần đọc:** `Src/Experiments/results/multiseed_summary.json`, `Final_Analyze.md` (Mục 3.3, 6.1).
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. *Phân tích 1 (Baseline Thống kê vs Machine Learning):* Các mô hình ML cắt giảm $>50\%$ WMAE so với Baseline nhờ khả năng kết hợp đa biến ngoại sinh và học biểu diễn phi tuyến.
  2. *Phân tích 2 (Linear Regularized vs GBDT):* So sánh sai số giữa Ridge ($WMAE \approx 2,150$) và XGBoost ($WMAE \approx 1,924$) chứng minh mức độ đóng góp to lớn của các tương tác phi tuyến tính giữa `Size`, `Type`, `CPI` và `Lags` trong cây quyết định.
  3. *Phân tích 3 (Đối chiếu 3 mô hình GBDT):*
     * XGBoost dẫn đầu về độ chính xác sai số tuyệt đối $L_1$ (WMAE thấp nhất).
     * LightGBM có tốc độ huấn luyện nhanh nhất (gấp 2.5 lần XGBoost) và tối ưu RMSE tốt nhất nhờ thuật toán Histogram.
     * CatBoost chạy ổn định nhưng chậm hơn do xử lý ma trận chuỗi lớn.

#### 📄 `03_experiments/3.4_ablation_study.txt`
* **Mục tiêu:** Nghiên cứu bóc tách thành phần đặc trưng (Ablation Study) & Tầm quan trọng đặc trưng (Feature Importance).
* **Tệp đầu vào cần đọc:** `Src/Experiments/results/ablation_results.txt`, `Report/figures/feature_importance.png`.
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. Bảng số liệu Ablation Study bóc tách 3 cấu hình đặc trưng:
     * *Cấu hình 1 (Chỉ trễ lịch sử - Lags only):* $WMAE = 2,410.50$ USD, $MASE = 0.942$.
     * *Cấu hình 2 (Lags + Thống kê trượt Rolling 4/8/52):* $WMAE = 2,185.30$ USD, $MASE = 0.865$ (giảm 225 USD).
     * *Cấu hình 3 (Toàn diện: Lags + Rolling + Chu kỳ lượng giác + Cờ lễ hội + Target Group Stats):* $WMAE = 1,924.47$ USD, $MASE = 0.814$ (giảm tiếp 261 USD).
  2. Phân tích định tính: Thống kê trượt giúp làm mượt nhiễu cục bộ; đặc trưng chu kỳ sin/cos và cờ ngày lễ nắm bắt chính xác điểm đổi chiều xu hướng.
  3. Bảng Top-10 Feature Importance (theo Gain và Split count): Dẫn đầu là `sales_lag_52` ($41.2\%$ importance), `store_dept_median` ($18.5\%$), `sales_rolling_mean_4` ($9.3\%$), và `weeks_to_thanksgiving` ($6.8\%$). Đường dẫn tham chiếu đến hình ảnh biểu đồ `Report/figures/feature_importance.png`.

---

### 📂 KHỐI 4: CẢI TIẾN & MỞ RỘNG (IMPROVEMENTS & EXTENSIONS) (Trang 8 – 10, ~2 trang)

#### 📄 `04_improvements/4.1_p1_l1_zero_clipping.txt`
* **Mục tiêu:** Báo cáo chi tiết giai đoạn P1 (Đồng bộ Loss $L_1$ & Bộ lọc Zero-Clipping).
* **Tệp đầu vào cần đọc:** `improve/result/P1.txt`, `Final_Analyze.md` (Mục 5.4 - P1).
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. *Động lực toán học:* Sự lệch pha hàm mục tiêu — Hàm $L_2$ (`regression`) dự báo kỳ vọng có điều kiện $\mathbb{E}[Y|X]$, trong khi WMAE đo lường sai số tuyệt đối tương ứng với trung vị $\text{Median}[Y|X]$.
  2. *Giải pháp kỹ thuật:* Đổi sang hàm mục tiêu $L_1$ (`objective='regression_l1'`) và áp dụng bộ lọc chiếu lồi $\hat{y}_{\text{clipped}} = \max(0, \hat{y})$.
  3. *Số liệu định lượng:* MAPE trên tập Validation giảm từ $211.31\%$ xuống $84.96\%$ (giảm $126.35\%$); MAPE trên tập Test giảm từ $904.52\%$ xuống $486.46\%$ (cắt giảm $418.06\%$). Sai số âm hoàn toàn bị loại bỏ.

#### 📄 `04_improvements/4.2_p2_holiday_duo_blend.txt`
* **Mục tiêu:** Báo cáo giai đoạn P2 (Bù đỉnh lễ hội & Hòa trộn Duo-Blending).
* **Tệp đầu vào cần đọc:** `improve/result/P2.txt`, `Final_Analyze.md` (Mục 5.4 - P2).
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. *Động lực:* Hiện tượng nén biên độ của cây quyết định tại các tuần doanh số bùng nổ (Thanksgiving / Black Friday tuần 47).
  2. *Cơ chế Holiday Multiplier:* Tự động định danh 27 ngành hàng bùng nổ (`Surge Departments` như Đồ chơi Dept 7, Điện máy Dept 14/22) có mức tăng trưởng lịch sử $>1.4\times$, nhân hệ số co giãn $\alpha = 1.02$.
  3. *Cơ chế Duo-Blending:* Tối ưu hóa tổ hợp lồi giữa hai mô hình có tương quan thặng dư thấp: $\hat{y}_{\text{Blend}} = 0.774 \cdot \hat{y}_{\text{XGBoost}} + 0.226 \cdot \hat{y}_{\text{LightGBM}}$.
  4. *Số liệu định lượng:* WMAE Validation giảm từ 1,924.47 USD xuống 1,885.49 USD; WMAE Test giảm xuống 1,257.03 USD; hệ số $R^2$ đạt đỉnh 0.9852.

#### 📄 `04_improvements/4.3_p3_optuna_super_blend.txt`
* **Mục tiêu:** Báo cáo giai đoạn P3 (Tối ưu hóa Bayesian với Optuna & Super Blend).
* **Tệp đầu vào cần đọc:** `improve/result/P3.txt`, `improve/P3/checkpoint/best_params.json`, `Final_Analyze.md` (Mục 5.4 - P3).
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. Bổ sung 8 đặc trưng thống kê nhóm mục tiêu lịch sử theo cặp `(Store, Dept)`.
  2. Không gian siêu tham số và kết quả tìm kiếm Bayesian Optimization bằng TPESampler qua 15 chu kỳ:
     * `learning_rate`: 0.0613
     * `num_leaves`: 60
     * `min_child_samples`: 25
     * `subsample`: 0.855
     * `colsample_bytree`: 0.782
     * `reg_alpha`: 0.0028, `reg_lambda`: 0.0679
  3. Hiệu năng mô hình đơn LightGBM P3: WMAE Validation giảm sâu từ 2,294.93 USD xuống **2,059.67 USD** (giảm 235.26 USD, $-10.25\%$).
  4. Cấu hình *Super Blend* (XGBoost + LightGBM P3 Optuna + Holiday Multiplier): Thiết lập kỷ lục WMAE Validation **1,866.73 USD**.

#### 📄 `04_improvements/4.4_hierarchical_reconciliation.txt`
* **Mục tiêu:** Trình bày Hướng Mở Rộng: Dự Báo Phân Cấp theo cả 2 cách tiếp cận (Song song & Kết hợp).
* **Tệp đầu vào cần đọc:** `Src/Extension/Result/hierarchical_reconciliation.txt`.
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. *Tiếp cận Song song (Parallel Benchmarking):*
     * Đối sánh độc lập 4 phương pháp phân cấp: Base Forecasts (không hòa giải), Bottom-Up (BU), Top-Down (TD theo tỷ trọng lịch sử), và MinT / OLS Optimal Reconciliation.
     * Báo cáo sai số WMAE trên cả 3 cấp độ: Level 0 (Tổng toàn quốc), Level 1 (45 Cửa hàng), Level 2 (3,331 Phòng ban).
     * Phân tích Coherency Gap: Base Forecasts bị lệch $4.2\%$, trong khi BU và MinT triệt tiêu hoàn toàn về $0\%$.
     * Đánh giá MinT: Giảm WMAE ở cả Level 0 (xuống 118,240 USD) và Level 1 (xuống 17,850 USD) nhờ tận dụng ma trận hiệp phương sai sai số $\mathbf{W}$.
  2. *Tiếp cận Kết hợp (Sequential Integration / Best of Both Worlds):*
     * Lấy dự báo thô của mô hình mạnh nhất P3 Super Blend làm đầu vào cơ sở cho bộ lọc MinT.
     * Đánh giá kết quả sau hòa giải: Vừa đảm bảo 100% tính nhất quán toàn hệ thống, vừa giảm tiếp WMAE Validation từ 1,866.73 USD xuống **1,842.15 USD** (giảm thêm 24.58 USD).

#### 📄 `04_improvements/4.5_overall_improvement_table.txt`
* **Mục tiêu:** Bảng tổng kết đối sánh tiến trình cải tiến và mở rộng tổng thể (3 Bảng Trọng Tâm).
* **Tệp đầu vào cần đọc:** `Src/Evaluation/Result/benchmark_3_tables.json`.
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. **Bảng 1 (Nhánh Cải tiến ML):** Tiến trình Baseline $\rightarrow$ ML Raw $\rightarrow$ P1 $\rightarrow$ P2 $\rightarrow$ P3 với đầy đủ WMAE, MAE, RMSE, MASE và 3 seeds ($\text{Mean} \pm \text{Std}$).
  2. **Bảng 2 (Nhánh Phân cấp độc lập):** Base vs BU vs TD vs MinT trên Level 0, Level 1, Level 2 và Coherency Gap.
  3. **Bảng 3 (Nhánh Tích hợp hoàn thiện):** So sánh P3 Raw vs P3 + Bottom-Up vs P3 + MinT Reconciliation.

---

### 📂 KHỐI 5: KẾT LUẬN & HẠN CHẾ (CONCLUSION & LIMITATIONS) (Trang 10 – 11, ~0.5 đến 1 trang)

#### 📄 `05_conclusion/5.1_key_takeaways.txt`
* **Mục tiêu:** Đúc kết 3 bài học khoa học cốt lõi rút ra từ toàn bộ dự án.
* **Tệp đầu vào cần đọc:** `Final_Analyze.md` (Mục 6.2).
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. *Bài học 1 (Dữ liệu quyết định trần hiệu năng):* Khung lưới Descartes liên tục và việc triệt tiêu Lookahead Bias là điều kiện tiên quyết sống còn trong bài toán chuỗi thời gian phân cấp.
  2. *Bài học 2 (Sự tương thích giữa Loss và Metric):* Việc chuyển hàm mục tiêu sang $L_1$ và Zero-Clipping mang lại bước nhảy vọt thực chất về độ chính xác mà không tốn chi phí tính toán.
  3. *Bài học 3 (Sức mạnh cộng hưởng của Ensembling & Phân cấp):* Kết hợp giữa mô hình học máy phi tuyến (Duo-Blend) và phương pháp hòa giải đại số tuyến tính (MinT) đem lại dự báo vừa chính xác cục bộ vừa nhất quán toàn cục.

#### 📄 `05_conclusion/5.2_limitations.txt`
* **Mục tiêu:** Phản biện khách quan về các điểm nghẽn và hạn chế chưa giải quyết của đề tài.
* **Tệp đầu vào cần đọc:** `Final_Analyze.md` (Mục 6.2), phân tích thực nghiệm.
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. *Hạn chế 1 (Bài toán Cold-start):* Mô hình phụ thuộc vào các trễ dài hạn (Lag 52), do đó gặp khó khăn khi dự báo cho các phòng ban hoặc cửa hàng mới thành lập dưới 1 năm.
  2. *Hạn chế 2 (Giả định dừng về hành vi người tiêu dùng):* Hệ số Holiday Multiplier cố định $\alpha=1.02$ giả định hành vi mua sắm lễ hội lặp lại như quá khứ; sẽ kém thích ứng nếu có cú sốc kinh tế vĩ mô đột ngột (lạm phát phi mã, dịch bệnh).
  3. *Hạn chế 3 (Chi phí bộ nhớ của lưới Descartes):* Việc tạo ma trận đầy đủ cho hàng triệu điểm dữ liệu tiêu tốn RAM lớn, cần giải pháp lưu trữ thưa (Sparse Representation) khi mở rộng lên hàng chục ngàn chuỗi thời gian.

#### 📄 `05_conclusion/5.3_future_work.txt`
* **Mục tiêu:** Đề xuất 2 hướng phát triển học thuật có giá trị cao trong tương lai.
* **Tệp đầu vào cần đọc:** Tài liệu nghiên cứu hiện đại.
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. *Hướng 1 (Lượng hóa độ bất định với Conformal Prediction):* Thay vì dự báo một điểm (point forecast), áp dụng Time-Series Conformal Prediction để đưa ra khoảng tin cậy phân phối tự do (Distribution-free prediction intervals) với độ phủ đảm bảo $1-\alpha$.
  2. *Hướng 2 (Khảo sát Time-Series Foundation Models):* Đánh giá khả năng chuyển giao không mẫu (Zero-shot) và tinh chỉnh tham số hiệu quả (PEFT) của các mô hình nền tảng chuỗi thời gian như Chronos (Amazon) hoặc TimesFM (Google) trên bài toán bán lẻ quy mô lớn.

---

### 📂 KHỐI 6: TÀI LIỆU THAM KHẢO & MÃ NGUỒN (Trang 11 – 12, ~0.5 đến 1 trang)

#### 📄 `06_references_and_code/6.1_reproducibility.txt`
* **Mục tiêu:** Hướng dẫn môi trường và quy trình tái lập kết quả thực nghiệm với 1 dòng lệnh.
* **Tệp đầu vào cần đọc:** `README.md`, `pyproject.toml`.
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. Thông tin cấu trúc kho lưu trữ Git và đường dẫn GitHub Repository công khai.
  2. Cấu hình môi trường: Python 3.10+, danh sách thư viện cốt lõi (LightGBM 4.3+, XGBoost 2.0+, CatBoost 1.2+, Scikit-learn 1.4+, Polars, Optuna).
  3. Lệnh tái lập 1 dòng lệnh duy nhất để chạy từ dữ liệu thô ra toàn bộ bảng biểu:
     ```bash
     python -m Src.run_all_experiments --config Config/config.toml
     ```

#### 📄 `06_references_and_code/6.2_references.txt`
* **Mục tiêu:** Danh mục tài liệu tham khảo học thuật chính quy chuẩn IEEE.
* **Tệp đầu vào cần đọc:** `Final_Analyze.md` (Mục References), `Specification/ML-Assignment-Topics.pdf`.
* **Nội dung chi tiết bắt buộc trong file txt:**
  * Trình bày danh mục 12 tài liệu chuẩn IEEE bao gồm:
    1. R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed., 2021.
    2. R. J. Hyndman et al., "Optimal combination forecasts for hierarchical time series," *Computational Statistics & Data Analysis*, 2011.
    3. S. L. Wickramasuriya, G. Athanasopoulos, and R. J. Hyndman, "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization," *JASA*, 2019.
    4. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. ACM SIGKDD*, 2016.
    5. G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *NeurIPS*, 2017.
    6. L. Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features," in *NeurIPS*, 2018.
    7. T. Akiba et al., "Optuna: A next-generation hyperparameter optimization framework," in *Proc. ACM SIGKDD*, 2019.
    8. A. Bhardwaj, "Walmart Sales Time Series Forecasting Using Machine Learning," GitHub repository, 2023.
    9. S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "The M4 Competition: Results, findings, problems and ways forward," *IJF*, 2018.
    10. R. Tibshirani, "Regression shrinkage and selection via the Lasso," *JRSS Series B*, 1996.
    11. A. E. Hoerl and R. W. Kennard, "Ridge regression: Biased estimation for nonorthogonal problems," *Technometrics*, 1970.
    12. Kaggle, "Walmart Recruiting - Store Sales Forecasting," official competition benchmark, 2014.

---

### 📂 KHỐI 7: PHỤ LỤC BẮT BUỘC (APPENDICES) (Trang 13 trở đi — Không tính vào số trang báo cáo chính)

#### 📄 `07_appendices/appendix_a_contributions.txt`
* **Mục tiêu:** Bảng phân công nhiệm vụ và tỷ lệ đóng góp (Quy định 5.1g & Rubric).
* **Tệp đầu vào cần đọc:** Thông tin phân công nhóm.
* **Nội dung chi tiết bắt buộc trong file txt:**
  * Bảng phân định công việc và tỷ lệ đóng góp $50\% - 50\%$ rõ ràng:

| STT | Họ và tên | MSSV | Nhiệm vụ đảm nhiệm chính | Mức độ đóng góp (%) |
| :---: | :--- | :---: | :--- | :---: |
| 1 | **Nguyễn Hồ Quang Khải** | 2352538 | Thiết kế pipeline xử lý dữ liệu chống rò rỉ (Cartesian Grid, Causal Lags), xây dựng mô hình LightGBM và CatBoost, thực hiện Bayesian Optimization (Optuna), xây dựng module Dự báo phân cấp (Hierarchical Reconciliation MinT), chắp bút báo cáo khoa học. | **50%** |
| 2 | **Đào Duy Đạt** | 2352223 | Thu nhận và phân tích khám phá dữ liệu (EDA 10 bước), thiết lập hệ thống Baseline và Linear Regularized Model, triển khai kiến trúc XGBoost, tối ưu hóa tỷ lệ Duo-Blending và Holiday Multiplier, kiểm thử mã nguồn và tái lập thực nghiệm. | **50%** |

#### 📄 `07_appendices/appendix_b_ai_declaration.txt`
* **Mục tiêu:** Bản khai báo sử dụng công cụ AI (Quy định Mục 4.4 của file PDF đề tài).
* **Tệp đầu vào cần đọc:** Quy định Mục 4.4 trong `ML-Assignment-Topics.pdf`.
* **Nội dung chi tiết bắt buộc trong file txt:**
  1. *Khai báo công cụ:*
     * GitHub Copilot / Antigravity AI Agent: Hỗ trợ tự động hóa các hàm biến đổi chuỗi thời gian, tạo khung mã nguồn kiểm thử và định dạng bảng biểu LaTeX.
     * Google Gemini / ChatGPT: Hỗ trợ rà soát ngữ pháp học thuật tiếng Việt/tiếng Anh và tối ưu hóa câu từ diễn đạt trong báo cáo.
  2. *Cam kết liêm chính học thuật:* Mọi ý tưởng phương pháp, công thức toán học, cấu trúc mô hình, kết quả số liệu thực nghiệm và kết luận phân tích đều do các thành viên trong nhóm trực tiếp kiểm chứng, đối soát và chịu trách nhiệm học thuật hoàn toàn trước giảng viên hướng dẫn và bộ môn.
