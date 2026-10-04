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

from train_p3 import train_p3_model
from evaluate_p3 import evaluate_p3


def main():
    print("================================================================================")
    print(" BẮT ĐẦU QUY TRÌNH THỰC THI GIAI ĐOẠN P3: ADVANCED OPTIMIZATION")
    print("================================================================================")

    # 1. Huấn luyện mô hình P3 nâng cao với Target Group Stats & Optuna
    print("\n>>> BƯỚC 1: Xây dựng đặc trưng P3 & Tối ưu siêu tham số tự động bằng Optuna...")
    train_p3_model(n_trials=8)

    # 2. Đánh giá toàn diện và kết hợp Super Blending
    print("\n>>> BƯỚC 2: Đánh giá hiệu năng P3 & Siêu kết hợp (Super Blending)...")
    evaluate_p3()

    print("\n================================================================================")
    print(" [THÀNH CÔNG] ĐÃ HOÀN TẤT GIAI ĐOẠN P3!")
    print(f" Báo cáo chi tiết đã được ghi vào: {IMPROVE_DIR / 'result' / 'P3.txt'}")
    print("================================================================================")


if __name__ == "__main__":
    main()
