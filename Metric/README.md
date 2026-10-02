# Các Thang Đo Đánh Giá (Evaluation Metrics)

Thư mục này định nghĩa và giải thích chi tiết các thang đo (metrics) được sử dụng để đánh giá chất lượng dự báo trong dự án **Time Series Forecasting Using Machine Learning** trên bộ dữ liệu bán lẻ Walmart.

---

## 1. WMAE (Weighted Mean Absolute Error) - Thang đo chuẩn của Walmart Challenge

### Công thức:
$$\text{WMAE} = \frac{\sum_{i=1}^{n} w_i \cdot |y_i - \hat{y}_i|}{\sum_{i=1}^{n} w_i}$$

Trong đó:
* $n$: Tổng số mẫu kiểm tra trong tập test/validation.
* $y_i$: Doanh số thực tế của mẫu thứ $i$ (`Weekly_Sales`).
* $\hat{y}_i$: Doanh số dự đoán bởi mô hình.
* $w_i$: Trọng số của mẫu thứ $i$:
  * $w_i = 5$ nếu tuần đó là tuần có ngày lễ lớn (`IsHoliday = True`).
  * $w_i = 1$ nếu tuần đó là tuần bình thường (`IsHoliday = False`).

### Ý nghĩa kinh doanh & kỹ thuật:
* **Tầm quan trọng của ngày lễ:** Trong ngành bán lẻ, các dịp lễ lớn (Super Bowl, Labor Day, Thanksgiving/Black Friday, Giáng Sinh) đóng góp phần lớn lợi nhuận và doanh thu cả năm. Nếu dự báo thiếu hàng (stockout) hoặc thừa hàng (overstock) vào các tuần này, chi phí thiệt hại lớn gấp nhiều lần ngày thường.
* **Hình phạt sai số:** WMAE phạt sai số dự đoán vào tuần lễ **gấp 5 lần** so với tuần thường, buộc mô hình phải tập trung học tốt các đột biến thời vụ và tác động của ngày lễ.

---

## 2. MAE (Mean Absolute Error)

### Công thức:
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### Ý nghĩa:
* Thước đo sai số tuyệt đối trung bình giữa giá trị thực tế và giá trị dự báo.
* Ưu điểm lớn nhất là **dễ diễn giải trực tiếp** theo đơn vị tiền tệ gốc (USD). Ví dụ, MAE = 1,500 nghĩa là trung bình mỗi tuần dự đoán chênh lệch khoảng 1,500 USD.
* Không nhạy cảm thái quá với các điểm ngoại lai (outliers) so với RMSE.

---

## 3. RMSE (Root Mean Squared Error)

### Công thức:
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### Ý nghĩa:
* Căn bậc hai của sai số bình phương trung bình.
* Bằng việc bình phương sai số $(y_i - \hat{y}_i)^2$, RMSE **phạt rất nặng các dự báo sai lệch lớn**.
* Hữu ích khi doanh nghiệp đặc biệt muốn tránh những sai sót mang tính thảm họa (dự báo lệch hàng chục ngàn USD).

---

## 4. MAPE (Mean Absolute Percentage Error)

### Công thức:
$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

### Ý nghĩa:
* Sai số phần trăm tuyệt đối trung bình, thể hiện độ lệch dưới dạng phần trăm (%).
* Cho phép so sánh hiệu năng dự báo giữa các cửa hàng/phòng ban có quy mô doanh số chênh lệch nhau (ví dụ: cửa hàng doanh thu 1 triệu USD so với cửa hàng 50 ngàn USD).
* *Lưu ý:* Cần xử lý tránh chia cho 0 khi $y_i = 0$ hoặc doanh số âm do trả hàng.

---

## 5. R² Score (Coefficient of Determination)

### Công thức:
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

Trong đó $\bar{y}$ là giá trị trung bình thực tế của doanh số.

### Ý nghĩa:
* Đo lường tỷ lệ phần trăm phương sai của doanh số mà các đặc trưng và mô hình giải thích được.
* $R^2 = 1$: Dự báo hoàn hảo.
* $R^2 = 0$: Mô hình chỉ ngang bằng mức đoán bằng giá trị trung bình $\bar{y}$.
* $R^2 < 0$: Mô hình kém hơn cả việc đoán trung bình.

---

## Bảng so sánh tóm tắt

| Thang đo | Đơn vị | Độ nhạy với Outliers | Trọng tâm đánh giá |
| :--- | :--- | :--- | :--- |
| **WMAE** | USD | Trung bình (Tập trung ngày Lễ) | **Thang đo chính của cuộc thi Walmart** |
| **MAE** | USD | Thấp (Bền vững) | Độ lệch tiền tệ trung bình dễ hiểu |
| **RMSE** | USD | Rất cao | Phạt sai số lớn |
| **MAPE** | % | Trung bình | Tỷ lệ phần trăm sai lệch |
| **$R^2$** | Không thứ nguyên ($\le 1$) | Cao | Khả năng giải thích biến thiên của mô hình |
