import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np

# Đảm bảo UTF-8 stream trên Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# Thêm BASE_DIR vào sys.path để import Config
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

from Config import load_config, get_path


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Nạp 4 tệp dữ liệu thô từ thư mục Data/Origin/data:
    train.csv, test.csv, stores.csv, features.csv.
    """
    config = load_config()
    train_path = get_path(config["paths"]["files"]["train_csv"])
    test_path = get_path(config["paths"]["files"]["test_csv"])
    stores_path = get_path(config["paths"]["files"]["stores_csv"])
    features_path = get_path(config["paths"]["files"]["features_csv"])

    print("1. Loading 4 raw dataset tables from Data/Origin/data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    stores = pd.read_csv(stores_path)
    features = pd.read_csv(features_path)

    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])
    features['Date'] = pd.to_datetime(features['Date'])

    print(f"   + train.csv   : {train.shape[0]:,} rows")
    print(f"   + test.csv    : {test.shape[0]:,} rows")
    print(f"   + stores.csv  : {stores.shape[0]:,} rows")
    print(f"   + features.csv: {features.shape[0]:,} rows")

    return train, test, stores, features


def build_continuous_time_grid(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Giải quyết rủ ro từ EDA:
    - Tạo khung lưới thời gian đầy đủ (Cartesian Product Store x Dept x All_Dates)
    - để đảm bảo phép tính shift(1) và shift(52) không bao giờ bị lệch tuần do đứt gãy dữ liệu.
    """
    print("\n2. Building continuous time grid (re-indexing Cartesian grid)...")

    # Toàn bộ các mốc tuần liên tục từ Train (143 tuần) qua Test (39 tuần) = 182 tuần
    all_dates = pd.date_range(
        start=min(train['Date'].min(), test['Date'].min()),
        end=max(train['Date'].max(), test['Date'].max()),
        freq='W-FRI'  # Dữ liệu Walmart luôn kết thúc vào thứ Sáu hàng tuần
    )
    print(f"   - Total continuous weekly timestamps (train + test): {len(all_dates)} weeks")

    # Toàn bộ các cặp (Store, Dept) duy nhất xuất hiện ở cả Train và Test
    pairs_train = train[['Store', 'Dept']].drop_duplicates()
    pairs_test = test[['Store', 'Dept']].drop_duplicates()
    all_pairs = pd.concat([pairs_train, pairs_test]).drop_duplicates().reset_index(drop=True)
    print(f"   - Total unique (Store, Dept) pairs: {len(all_pairs):,}")

    # Tạo MultiIndex Cartesian Product
    grid = pd.MultiIndex.from_product(
        [all_pairs['Store'].unique(), all_pairs['Dept'].unique(), all_dates],
        names=['Store', 'Dept', 'Date']
    ).to_frame().reset_index(drop=True)

    # Lọc chỉ giữ các cặp (Store, Dept) thực sự tồn tại
    grid = grid.merge(all_pairs, on=['Store', 'Dept'], how='inner')
    print(f"   - Normalized full grid size: {len(grid):,} rows")

    # Đánh dấu nguồn gốc bản ghi (Train, Test hay Tuần khuyết giữa chừng)
    train_tagged = train.copy()
    train_tagged['Is_Train_Record'] = 1
    test_tagged = test.copy()
    test_tagged['Is_Test_Record'] = 1

    grid = grid.merge(train_tagged[['Store', 'Dept', 'Date', 'Weekly_Sales', 'Is_Train_Record']],
                      on=['Store', 'Dept', 'Date'], how='left')
    grid = grid.merge(test_tagged[['Store', 'Dept', 'Date', 'Is_Test_Record']],
                      on=['Store', 'Dept', 'Date'], how='left')

    grid['Is_Train_Record'] = grid['Is_Train_Record'].fillna(0).astype(int)
    grid['Is_Test_Record'] = grid['Is_Test_Record'].fillna(0).astype(int)

    # Với các tuần trong giai đoạn Train mà bị khuyết (lỗ hổng giữa chừng):
    # Theo kết luận EDA: điền Weekly_Sales = 0 (phòng ban không phát sinh doanh số tuần đó)
    train_period_mask = (grid['Date'] <= train['Date'].max()) & (grid['Is_Train_Record'] == 0)
    grid.loc[train_period_mask, 'Weekly_Sales'] = 0.0

    print("   -> Completed continuous time grid normalization to prevent lag phase shift!")
    return grid


def merge_exogenous_features(grid: pd.DataFrame, stores: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """
    Ghép 3 bảng thành một bảng dữ liệu tổng thể thống nhất:
    Grid + stores.csv + features.csv.
    """
    print("\n3. Merging exogenous features from stores.csv and features.csv...")

    # Nối với stores.csv theo 'Store'
    df = grid.merge(stores, on='Store', how='left')

    # Nối với features.csv theo ['Store', 'Date']
    # Loại bỏ IsHoliday từ features nếu grid đã có hoặc lấy chính xác từ features
    feat_cols = ['Store', 'Date', 'Temperature', 'Fuel_Price',
                 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
                 'CPI', 'Unemployment', 'IsHoliday']
    df = df.merge(features[feat_cols], on=['Store', 'Date'], how='left')

    # Xử lý các giá trị khuyết thiếu theo kết luận EDA
    print("   - Handling missing values according to EDA conclusions:")
    # 1. Điền 0 cho các cột khuyến mãi MarkDown1 - MarkDown5
    md_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in md_cols:
        df[col] = df[col].fillna(0.0)
    print("     * Filled 0 for MarkDown1 -> MarkDown5 (weeks without markdowns)")

    # 2. Tạo đặc trưng khuyến mãi tổng hợp
    df['Total_MarkDown'] = df[md_cols].sum(axis=1)
    df['Has_MarkDown'] = (df['Total_MarkDown'] > 0).astype(int)

    # 3. Điền khuyết thiếu cho CPI và Unemployment (thiếu ở các tuần cuối Test)
    # Áp dụng forward-fill rồi backward-fill theo từng Store
    df['CPI'] = df.groupby('Store')['CPI'].ffill().bfill()
    df['Unemployment'] = df.groupby('Store')['Unemployment'].ffill().bfill()
    print("     * Forward-filled interpolated CPI and Unemployment by store")

    # 4. Mã hóa biến Loại cửa hàng (Type)
    type_map = {'A': 1, 'B': 2, 'C': 3}
    df['Type'] = df['Type'].map(type_map).astype(int)

    # 5. Chuyển IsHoliday sang 0/1
    df['IsHoliday'] = df['IsHoliday'].astype(int)

    return df


def create_calendar_and_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    GIẢI QUYẾT RỦI RO 4 TỪ EDA:
    Tạo đặc trưng lịch tuần hoàn và bóc tách riêng 4 kỳ nghỉ lễ lớn
    để giải quyết hiện tượng lệch pha lịch (Holiday Calendar Shift).
    """
    print("\n4. Extracting calendar features and aligning holiday phase shifts...")

    # Đặc trưng lịch cơ bản
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Biến đổi sóng sin/cos tuần hoàn cho Week (chu kỳ 52 tuần) và Month (chu kỳ 12 tháng)
    df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52.0)
    df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52.0)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)

    # Bóc tách cờ riêng cho 4 kỳ lễ lớn theo phát hiện EDA
    # 1. Super Bowl (Tuần 5 hoặc 6)
    df['is_super_bowl'] = ((df['Week'].isin([5, 6])) & (df['IsHoliday'] == 1)).astype(int)

    # 2. Labor Day (Tuần 36)
    df['is_labor_day'] = ((df['Week'] == 36) & (df['IsHoliday'] == 1)).astype(int)

    # 3. Thanksgiving (Tuần 47 hoặc 48 - bùng nổ doanh số lớn nhất)
    df['is_thanksgiving'] = ((df['Week'].isin([47, 48])) & (df['IsHoliday'] == 1)).astype(int)

    # 4. Christmas (Tuần 52)
    df['is_christmas'] = ((df['Week'].isin([51, 52])) & (df['IsHoliday'] == 1)).astype(int)

    # 5. Tuần mua sắm cao điểm ngay trước Giáng Sinh (Tuần 51)
    df['is_pre_christmas_peak'] = (df['Week'] == 51).astype(int)

    # 6. Khoảng cách tuần đến đợt mua sắm Thanksgiving/Black Friday
    df['weeks_to_thanksgiving'] = df['Week'].apply(lambda w: abs(w - 47))

    return df


def create_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    GIẢI QUYẾT RỦI RO 1 & 2 TỪ EDA:
    Tạo các Lag và Rolling Statistics trên lưới đã Re-index liên tục theo thời gian.
    QUY TẮC SỐNG CÒN: Sắp xếp theo [Store, Dept, Date] và shift(1) để chống rò rỉ dữ liệu (Lookahead Bias).
    """
    print("\n5. Creating lag features and rolling window statistics...")

    df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    grouped = df.groupby(['Store', 'Dept'])['Weekly_Sales']

    # 1. Các Lag an toàn dài hạn (AN TOÀN TUYỆT ĐỐI CHO TẬP TEST HORIZON 39 TUẦN VÌ >= 39)
    # EDA chứng minh sales_lag_52 có tương quan cực cao: 0.926!
    # sales_lag_51 và 53 giúp xử lý hiện tượng lệch ngày Black Friday giữa các năm
    safe_test_lags = [51, 52, 53]
    for lag in safe_test_lags:
        df[f'sales_lag_{lag}'] = grouped.shift(lag)

    # 2. Các Lag ngắn hạn (phục vụ mô hình 1-step, recursive hoặc baseline)
    short_lags = [1, 2, 4, 8]
    for lag in short_lags:
        df[f'sales_lag_{lag}'] = grouped.shift(lag)

    # 3. Thống kê cửa sổ trượt (Rolling Window Statistics)
    # LƯU Ý: Phải shift(1) trước khi rolling để không làm lộ doanh số của chính tuần cần dự đoán!
    shifted_sales = grouped.shift(1)
    shifted_grouped = shifted_sales.groupby([df['Store'], df['Dept']])

    # Cửa sổ 4 tuần (1 tháng)
    df['sales_rolling_mean_4'] = shifted_grouped.rolling(window=4, min_periods=1).mean().reset_index(drop=True)
    df['sales_rolling_std_4'] = shifted_grouped.rolling(window=4, min_periods=1).std().fillna(0.0).reset_index(drop=True)

    # Cửa sổ 8 tuần (2 tháng)
    df['sales_rolling_mean_8'] = shifted_grouped.rolling(window=8, min_periods=1).mean().reset_index(drop=True)

    # Cửa sổ 52 tuần (1 năm)
    df['sales_rolling_mean_52'] = shifted_grouped.rolling(window=52, min_periods=1).mean().reset_index(drop=True)

    # Tỷ số tăng trưởng doanh số ngắn hạn so với trung bình 1 tháng gần nhất
    df['sales_growth_ratio_4'] = (df['sales_lag_1'] / (df['sales_rolling_mean_4'] + 1.0)).fillna(1.0)

    # Đặc trưng tương tác giữa quy mô cửa hàng và phòng ban
    df['sales_to_size_ratio'] = df['sales_rolling_mean_4'] / (df['Size'] + 1.0)

    # Điền giá trị cho các tuần đầu năm chưa có lịch sử lag (dùng -1 để mô hình cây nhận diện)
    lag_cols = [c for c in df.columns if 'lag' in c or 'rolling' in c]
    for c in lag_cols:
        df[c] = df[c].fillna(-1.0)

    return df


def preprocess_pipeline():
    """
    Hàm điều phối toàn bộ quy trình tiền xử lý dữ liệu và tạo bảng hoàn chỉnh.
    """
    print("=" * 80)
    print(" Starting comprehensive preprocessing & feature engineering pipeline")
    print("=" * 80)

    config = load_config()
    output_dir = get_path(config["paths"]["data"]["preprocess_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_parquet_path = get_path(config["paths"]["files"]["processed_train_parquet"])
    test_parquet_path = get_path(config["paths"]["files"]["processed_test_parquet"])
    train_csv_path = get_path(config["paths"]["files"]["processed_train_csv"])
    test_csv_path = get_path(config["paths"]["files"]["processed_test_csv"])

    # 1. Đọc dữ liệu gốc
    train, test, stores, features = load_raw_data()

    # 2. Xây dựng lưới thời gian liên tục chống rò rỉ và lệch pha lag
    grid = build_continuous_time_grid(train, test)

    # 3. Ghép 3 bảng dữ liệu
    df = merge_exogenous_features(grid, stores, features)

    # 4. Tạo Calendar và Holiday Shift features
    df = create_calendar_and_holiday_features(df)

    # 5. Tạo Lag và Rolling features trên chuỗi liên tục
    df = create_lag_and_rolling_features(df)

    # 6. Tách thành tập Train và Test thực tế
    print("\n6. Extracting final tables for train and test...")

    # Lọc lại đúng các dòng quan sát ban đầu của Train và Test
    processed_train = df[df['Is_Train_Record'] == 1].copy()
    processed_test = df[df['Is_Test_Record'] == 1].copy()

    # Loại bỏ các cột đánh dấu nội bộ tạm thời
    internal_cols = ['Is_Train_Record', 'Is_Test_Record']
    processed_train.drop(columns=internal_cols, inplace=True)
    processed_test.drop(columns=internal_cols + ['Weekly_Sales'], inplace=True)

    print(f"   + Processed train table: {processed_train.shape[0]:,} rows, {processed_train.shape[1]} columns")
    print(f"   + Processed test table : {processed_test.shape[0]:,} rows, {processed_test.shape[1]} columns")

    # 7. Lưu kết quả ra file Parquet tốc độ cao và CSV dự phòng
    print(f"\n7. Saving processed data to directory: {output_dir} ...")

    # Lưu tập Train
    try:
        processed_train.to_parquet(train_parquet_path, index=False)
        print(f"   [OK] Saved train Parquet: {train_parquet_path.name} ({train_parquet_path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"   [WARN] Could not save Parquet ({e}). Saving to CSV...")
    processed_train.to_csv(train_csv_path, index=False)
    print(f"   [OK] Saved train CSV: {train_csv_path.name}")

    # Lưu tập Test
    try:
        processed_test.to_parquet(test_parquet_path, index=False)
        print(f"   [OK] Saved test Parquet: {test_parquet_path.name} ({test_parquet_path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"   [WARN] Could not save Parquet ({e}). Saving to CSV...")
    processed_test.to_csv(test_csv_path, index=False)
    print(f"   [OK] Saved test CSV: {test_csv_path.name}")

    print("\n" + "=" * 80)
    print(" [SUCCESS] Preprocessing pipeline completed successfully!")
    print(f" List of generated features ({len(processed_train.columns)} columns):")
    print(f" {list(processed_train.columns)}")
    print("=" * 80)


if __name__ == "__main__":
    preprocess_pipeline()
