import sys
from pathlib import Path

# Đảm bảo UTF-8 stream trên Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
IMPROVE_DIR = CURRENT_DIR.parent
BASE_DIR = IMPROVE_DIR.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(CURRENT_DIR))

from train_p1_lightgbm import train_lightgbm_p1
from evaluate_p1 import evaluate_p1


def main():
    print("================================================================================")
    print(" BẮT ĐẦU QUY TRÌNH THỰC THI GIAI ĐOẠN P1: QUICK WINS")
    print("================================================================================")

    # 1. Huấn luyện mô hình cải tiến
    print("\n>>> BƯỚC 1: Huấn luyện LightGBM với objective='regression_l1' (L1 loss)...")
    train_lightgbm_p1()

    # 2. Đánh giá và xuất kết quả ra improve/result/P1.txt
    print("\n>>> BƯỚC 2: Đánh giá mô hình & áp dụng kỹ thuật Zero-clipping hậu xử lý...")
    evaluate_p1()

    print("\n================================================================================")
    print(" [THÀNH CÔNG] ĐÃ HOÀN TẤT GIAI ĐOẠN P1!")
    print(f" Kết quả chi tiết đã được ghi vào: {IMPROVE_DIR / 'result' / 'P1.txt'}")
    print("================================================================================")


if __name__ == "__main__":
    main()
