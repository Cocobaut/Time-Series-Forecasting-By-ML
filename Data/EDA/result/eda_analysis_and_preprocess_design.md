# BÁO CÁO TOÀN DIỆN KẾT QUẢ PHÂN TÍCH EDA & CÁC QUYẾT ĐỊNH THIẾT KẾ TIỀN XỬ LÝ (PREPROCESS DESIGN)

Dự án: **Times series forecasting using machine learning**  
Bộ dữ liệu: **Walmart Recruiting - Store Sales Forecasting**  
Địa điểm lưu trữ artifacts: `Data/EDA/result/`

---

## 📌 TỔNG QUAN KẾT QUẢ 10 BƯỚC PHÂN TÍCH EDA & QUYẾT ĐỊNH PREPROCESS TƯƠNG ỨNG

### Bước 0: Kiểm tra giá trị khuyết thiếu (Missing Values / NA Analysis)
* **Kết quả phân tích EDA:**
  * `train.csv` (421,570 dòng), `stores.csv` (45 dòng), `test.csv` (115,064 dòng) **hoàn toàn sạch 100% không khuyết thiếu**.
  * `features.csv` (8,190 dòng) chứa 7 cột có giá trị NA:
    * `MarkDown1`: 4,158 dòng (50.77%)
    * `MarkDown2`: 5,269 dòng (64.33%)
    * `MarkDown3`: 4,577 dòng (55.89%)
    * `MarkDown4`: 4,726 dòng (57.70%)
    * `MarkDown5`: 4,140 dòng (50.55%)
    * `CPI`: 585 dòng (7.14% - tập trung ở các tuần cuối năm 2013 của tập Test)
    * `Unemployment`: 585 dòng (7.14% - tập trung ở các tuần cuối năm 2013 của tập Test)
* **Quyết định thiết kế Preprocess data:**
  1. Điền giá trị `0.0` cho `MarkDown1` đến `MarkDown5` vì NA biểu thị thời điểm không áp dụng chương trình khuyến mãi giảm giá.
  2. Tạo cờ tổng hợp: `Total_MarkDown = sum(MarkDown1..5)` và `Has_MarkDown = (Total_MarkDown > 0)`.
  3. Áp dụng kỹ thuật nội suy thời gian: Forward-fill (`ffill()`) kết hợp Backward-fill (`bfill()`) theo từng `Store` cho `CPI` và `Unemployment` để bảo toàn tính liên tục của chuỗi vĩ mô trên tập Test.

---

### Bước 1: Tính toàn vẹn thời gian & Rủi ro Re-indexing (Time Continuity & Cartesian Product)
* **Kết quả phân tích EDA:**
  * Train gồm **143 tuần** liên tục (2010-02-05 đến 2012-10-26). Test gồm **39 tuần** liên tục (2012-11-02 đến 2013-07-26).
  * Toàn bộ các phòng ban (`Dept`) trong Test đều có mặt trong Train (không có Dept mới hoàn toàn).
  * Lưới tuần đầy đủ: $3,331 	ext{ cặp (Store, Dept)} 	imes 143 	ext{ tuần} = \mathbf{476,333} 	ext{ dòng}$.
  * Dữ liệu thực tế trong `train.csv` chỉ có **$421,570$ dòng** $ightarrow$ Bị khuyết **$54,763$ dòng ($11.50\%$)**.
  * Phân loại 3,331 chuỗi:
    * $2,660$ cặp ($79.86\%$) liên tục đầy đủ 143 tuần.
    * $106$ cặp ($3.18\%$) mở muộn (`Late_Start`).
    * $102$ cặp ($3.06\%$) đóng sớm (`Early_End`).
    * $463$ cặp ($13.90\%$) bị gián đoạn đứt gãy giữa chừng (`Intermittent_Gaps`).
* **Quyết định thiết kế Preprocess data:**
  1. **Bắt buộc Re-indexing Cartesian Grid:** Ghép toàn bộ cặp `(Store, Dept)` với chuỗi ngày liên tục 182 tuần (Train + Test) trước khi tạo lag.
  2. Với các tuần bị khuyết giữa chừng trong giai đoạn hoạt động: Gán `Weekly_Sales = 0.0` (phòng ban mở cửa nhưng tuần đó không phát sinh doanh số).
  3. Cơ chế này đảm bảo khi thực hiện `shift(1)` hay `shift(52)`, dữ liệu luôn lùi đúng 1 tuần hoặc đúng 52 tuần lịch sử, loại bỏ triệt để hiện tượng lấy nhầm tuần.

---

### Bước 2: Phân phối Weekly_Sales, Doanh số âm & Bẫy Retransformation
* **Kết quả phân tích EDA:**
  * Có **$1,285$ dòng có doanh số âm** ($< 0$ USD, chiếm $0.30\%$), giá trị nhỏ nhất là $-4,988.94$ USD (do trả hàng, hủy đơn hoặc điều chỉnh sổ sách kiểm kê).
  * Có $73$ dòng có doanh số đúng bằng $0$ USD ($0.02\%$).
  * Phân vị 90%: $42,845$ USD; phân vị 99%: $106,479$ USD; phân vị 99.9%: $174,882$ USD.
  * Phân tích toán học: Biến đổi $\log(1+y)$ sẽ bị lỗi với số âm $\le -1$. Hơn nữa, huấn luyện trên không gian log sẽ tối ưu hóa sai số tương đối (RMSLE), khi nghịch đảo $\exp(\hat{y}) - 1$ sẽ bị lệch kỳ vọng (Retransformation Bias: $\mathbb{E}[y] 
e \exp(\mathbb{E}[\log y])$).
* **Quyết định thiết kế Preprocess data & Modeling:**
  1. **Không biến đổi log** trên biến mục tiêu `Weekly_Sales`. Giữ nguyên thang đo USD gốc.
  2. Cấu hình các mô hình học máy tối ưu trực tiếp hàm mục tiêu $L_1$ / MAE: LightGBM dùng `objective='regression_l1'`, CatBoost dùng `loss_function='MAE'`.
  3. Thiết lập bước hậu xử lý (Post-processing): Gán `np.clip(y_pred, 0, None)` sau khi dự báo để triệt tiêu các giá trị âm vô lý, giảm điểm phạt WMAE.

---

### Bước 3: Phân rã chuỗi thời gian: Xu hướng (Trend) & Chu kỳ mùa vụ (Seasonality)
* **Kết quả phân tích EDA:**
  * Xu hướng dài hạn (Trend) của tổng doanh số toàn hệ thống duy trì mức ổn định quanh ngưỡng 45 - 48 triệu USD/tuần.
  * Tính chu kỳ thời vụ (Seasonality 52 tuần) cực kỳ rõ nét: Hàng năm luôn có 2 đỉnh doanh số khổng lồ vào tháng 11 (Thanksgiving) và tháng 12 (trước Giáng Sinh).
* **Quyết định thiết kế Preprocess data:**
  1. Trích xuất nhóm **Calendar Features**: `Year`, `Month`, `Week`, `Quarter`, `DayOfYear`.
  2. Mã hóa chu kỳ tuần hoàn bằng hàm lượng giác sóng:
     $$	ext{Week\_sin} = \sin\left(rac{2\pi \cdot 	ext{Week}}{52}ight), \quad 	ext{Week\_cos} = \cos\left(rac{2\pi \cdot 	ext{Week}}{52}ight)$$
     $$	ext{Month\_sin} = \sin\left(rac{2\pi \cdot 	ext{Month}}{12}ight), \quad 	ext{Month\_cos} = \cos\left(rac{2\pi \cdot 	ext{Month}}{12}ight)$$

---

### Bước 4: Tác động Ngày lễ, Holiday Shift & Lệch pha LAG-52
* **Kết quả phân tích EDA:**
  * Doanh số trung bình theo kỳ lễ:
    * **Thanksgiving:** Đạt mức cao nhất toàn hệ thống với **$22,220.94$ USD** (tăng ~40% so với ngày thường $15,901$ USD).
    * **Super Bowl:** Đạt $16,378.00$ USD.
    * **Labor Day:** Đạt $15,881.69$ USD.
    * **Christmas:** Đạt $14,543.39$ USD.
  * Hiện tượng **Holiday Calendar Shift**: Cờ `IsHoliday = True` của Giáng Sinh rơi vào tuần 52, nhưng **đỉnh mua sắm thực tế luôn bùng nổ ở tuần 51** (ngay trước Giáng Sinh). Doanh số tuần 51 cao gần gấp đôi tuần 52!
  * Khảo sát ma trận ISO week: Thanksgiving năm 2010, 2011, 2012 rơi vào tuần 47, nhưng năm 2013 chuyển sang tuần 48.
* **Quyết định thiết kế Preprocess data:**
  1. Bóc tách riêng 4 cờ ngày lễ: `is_super_bowl`, `is_labor_day`, `is_thanksgiving`, `is_christmas`.
  2. Tạo cờ cao điểm mua sắm trước Giáng Sinh: `is_pre_christmas_peak` (tuần 51).
  3. Tạo đặc trưng khoảng cách thời gian: `weeks_to_thanksgiving = abs(Week - 47)`.
  4. Sử dụng `sales_lag_51`, `sales_lag_52`, `sales_lag_53` song song để bù trừ độ lệch pha ngày lễ giữa các năm.

---

### Bước 5: Phân cấp Cửa hàng và Phòng ban (Store & Dept Hierarchy)
* **Kết quả phân tích EDA:**
  * **Quy luật Pareto (80/20):** Top 5 phòng ban (Dept 92, 95, 38, 72, 90) đóng góp gần **$30\%$ tổng doanh thu** của toàn bộ chuỗi siêu thị.
  * Doanh số có sự phân hóa cực lớn giữa các cửa hàng: Store 20, 4, 14, 13 có doanh số trung bình cao nhất (> 28,000 USD/tuần), trong khi Store 33, 44, 5 chỉ đạt dưới 6,000 USD/tuần.
* **Quyết định thiết kế Preprocess data:**
  1. Giữ nguyên `Store` và `Dept` dưới dạng categorical features để mô hình GBDT học target encoding cục bộ.
  2. Tạo đặc trưng tương tác quy mô: Tỷ lệ giữa doanh số trung bình gần nhất với diện tích cửa hàng (`sales_to_size_ratio`).

---

### Bước 6: Khảo sát đặc tính Cửa hàng (Type & Size) từ stores.csv
* **Kết quả phân tích EDA:**
  * Loại A: 22 siêu thị Supercenter, diện tích sàn trung bình $182,231$ sqft, doanh số TB **$20,099.57$ USD**.
  * Loại B: 17 siêu thị quy mô vừa, diện tích sàn trung bình $101,819$ sqft, doanh số TB **$12,237.08$ USD**.
  * Loại C: 6 cửa hàng tiện lợi nhỏ, diện tích sàn trung bình $40,536$ sqft, doanh số TB **$9,519.53$ USD**.
  * Hệ số tương quan dương rõ rệt giữa diện tích (`Size`) và doanh số bán hàng.
* **Quyết định thiết kế Preprocess data:**
  1. Mã hóa Ordinal cho Loại cửa hàng: `Type = {'A': 1, 'B': 2, 'C': 3}`.
  2. Giữ nguyên `Size` làm biến ngoại sinh số học kết hợp tương tác với các biến doanh số.

---

### Bước 7: Đánh giá Khuyến mãi (MarkDown1 đến MarkDown5)
* **Kết quả phân tích EDA:**
  * Mốc thời gian bắt đầu ghi nhận dữ liệu MarkDown là **`2011-11-11`**. Trước thời điểm này 100% là NA do Walmart chưa thu thập dữ liệu này.
* **Quyết định thiết kế Preprocess data:**
  1. Điền giá trị `0.0` cho toàn bộ các ô NA của MarkDown.
  2. Tạo biến tổng `Total_MarkDown = MarkDown1 + MarkDown2 + MarkDown3 + MarkDown4 + MarkDown5`.
  3. Tạo biến cờ nhị phân `Has_MarkDown` để mô hình dễ dàng phân tách giai đoạn có khuyến mãi và không khuyến mãi.

---

### Bước 8: Kinh tế vĩ mô và Thời tiết (features.csv)
* **Kết quả phân tích EDA:**
  * Tương quan tuyến tính trực tiếp (Pearson/Spearman) giữa các biến vĩ mô đơn lẻ (`Temperature`, `Fuel_Price`, `CPI`, `Unemployment`) với `Weekly_Sales` dao động ở mức thấp (-0.00 đến -0.02).
  * Điều này chứng minh bản chất quan hệ là **phi tuyến tính** (ví dụ: nhiệt độ quá lạnh kích cầu áo ấm ở Dept may mặc, giá xăng tác động gián tiếp đến hành vi tích trữ).
* **Quyết định thiết kế Preprocess data:**
  1. Giữ nguyên các biến vĩ mô để mô hình học máy dạng cây (Tree-based) tự động chia nhánh phi tuyến và học tương tác bậc cao.
  2. Điền khuyết thiếu cho `CPI` và `Unemployment` bằng forward-fill/backward-fill theo Store.

---

### Bước 9: Tự tương quan (ACF/PACF) & Rủi ro Test Horizon khi tạo Lag
* **Kết quả phân tích EDA:**
  * Hệ số tự tương quan qua các bước trễ:
    * `Lag 1`: Corr = $0.338$ (Rủi ro thiếu nhãn trên Test)
    * `Lag 2`: Corr = $0.221$ (Rủi ro thiếu nhãn trên Test)
    * `Lag 4`: Corr = $0.174$ (Rủi ro thiếu nhãn trên Test)
    * `Lag 39`: Corr = $-0.138$ (An toàn cho Test)
    * `Lag 51`: Corr = $0.258$ (An toàn cho Test)
    * **`Lag 52`: Corr = 0.926 (CỰC KỲ CAO - An toàn 100% cho Test)**
    * `Lag 53`: Corr = $0.363$ (An toàn cho Test)
  * **Rủi ro kỹ thuật:** Tập Test kéo dài liên tục 39 tuần không có nhãn. Do đó `lag_1`, `lag_2` không tồn tại thực tế từ tuần thứ 2 của Test trở đi nếu không áp dụng Recursive forecasting.
* **Quyết định thiết kế Preprocess data:**
  1. Xây dựng bộ ba **Lag an toàn dài hạn**: `sales_lag_51`, `sales_lag_52`, `sales_lag_53` làm trụ cột chính.
  2. Tạo các **Rolling Window Statistics** có `shift(1)` chống Lookahead bias: `sales_rolling_mean_4`, `sales_rolling_std_4`, `sales_rolling_mean_8`, `sales_rolling_mean_52`.
  3. Tạo tỷ số tăng trưởng: `sales_growth_ratio_4 = sales_lag_1 / (sales_rolling_mean_4 + 1.0)`.

---

### Bước 10: Dịch chuyển phân phối giữa Train và Test (Train/Test Consistency)
* **Kết quả phân tích EDA:**
  * Điểm giao thoa thời gian giữa Train (`2012-10-26`) và Test (`2012-11-02`) liền mạch hoàn toàn.
  * Phân phối của `Fuel_Price` và `Temperature` giữa Train và Test rất tương đồng.
  * `CPI` có xu hướng tăng nhẹ theo lạm phát tự nhiên (Train: 171.58 $ightarrow$ Test: 177.31), `Unemployment` giảm nhẹ (Train: 8.00% $ightarrow$ Test: 6.88%).
* **Quyết định thiết kế Validation:**
  1. Áp dụng chiến lược chia tập kiểm định theo thời gian (**Time-based Holdout & TimeSeriesSplit**).
  2. Không bao giờ sử dụng K-Fold ngẫu nhiên để tránh vi phạm giả định trật tự thời gian.

---
Báo cáo được tổng hợp và xuất tự động từ module: `Data/EDA/eda.py`
