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

from evaluate_p2 import evaluate_p2


def main():
    print("================================================================================")
    print(" BẮT ĐẦU QUY TRÌNH THỰC THI GIAI ĐOẠN P2: POST-PROCESSING & BLENDING")
    print("================================================================================")

    # Đánh giá Blending và Holiday Multiplier
    evaluate_p2()

    print("\n================================================================================")
    print(" [THÀNH CÔNG] ĐÃ HOÀN TẤT GIAI ĐOẠN P2!")
    print(f" Kết quả chi tiết đã được ghi vào: {IMPROVE_DIR / 'result' / 'P2.txt'}")
    print("================================================================================")


if __name__ == "__main__":
    main()
