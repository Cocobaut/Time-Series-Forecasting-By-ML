import json
import os
import shutil
import zipfile
from pathlib import Path
import sys

# Đảm bảo mã hóa UTF-8 trên Windows console để tránh UnicodeEncodeError
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def setup_kaggle_credentials():
    """Tu dong thiet lap thong tin xac thuc tu file kaggle.json."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    local_kaggle_json = base_dir / "kaggle.json"
    user_kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    # Copy sang ~/.kaggle neu can va nap bien moi truong
    if local_kaggle_json.exists():
        user_kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_kaggle_json, user_kaggle_json)
        try:
            with open(local_kaggle_json, 'r', encoding='utf-8') as f:
                creds = json.load(f)
                os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
                os.environ["KAGGLE_KEY"] = creds.get("key", "")
                print(f"[OK] Da nap tai khoan Kaggle: {creds.get('username')}")
        except Exception as e:
            print(f"[WARN] Loi khi doc kaggle.json: {e}")
    elif user_kaggle_json.exists():
        try:
            with open(user_kaggle_json, 'r', encoding='utf-8') as f:
                creds = json.load(f)
                os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
                os.environ["KAGGLE_KEY"] = creds.get("key", "")
                print(f"[OK] Da nap tai khoan Kaggle tu ~/.kaggle: {creds.get('username')}")
        except Exception as e:
            print(f"[WARN] Loi khi doc ~/.kaggle/kaggle.json: {e}")
    else:
        print("[WARN] Khong tim thay file kaggle.json!")

def download_data():
    origin_dir = Path(__file__).resolve().parent
    data_dir = origin_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    setup_kaggle_credentials()

    print("\n=== BAT DAU TAI DU LIEU CUOC THI WALMART ===")
    import kagglehub

    try:
        path = kagglehub.competition_download('walmart-recruiting-store-sales-forecasting')
        print(f"\n[OK] Duong dan cache du lieu: {path}")

        # Sao chep va giai nen vao Data/Origin/data
        print("[...] Dang dong bo va giai nen du lieu vao Data/Origin/data...")
        for file in Path(path).iterdir():
            dest_file = data_dir / file.name
            if file.is_file():
                shutil.copy2(file, dest_file)
                if dest_file.suffix == '.zip':
                    print(f"     -> Dang giai nen: {dest_file.name} ...")
                    with zipfile.ZipFile(dest_file, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)

        print(f"\n[THANH CONG] Toan bo du lieu da san sang tai: {data_dir}")
        print("\nDanh sach cac file hien co:")
        for f in sorted(data_dir.iterdir()):
            if f.is_file() and f.suffix in ['.csv', '.zip']:
                print(f" - {f.name:25} ({f.stat().st_size / (1024*1024):6.2f} MB)")

    except Exception as e:
        error_msg = str(e)
        print(f"\n[LOI KHI TAI]: {error_msg}")
        if "401" in error_msg or "permission" in error_msg.lower() or "rules" in error_msg.lower():
            print("\n" + "="*70)
            print("(!) BUOC QUAN TRONG DE MO KHOA QUYEN TAI DU LIEU:")
            print("Tai khoan Kaggle cua ban can bam Chap nhan Dieu khoan (Rules) cua cuoc thi.")
            print("1. Mo trinh duyet va truy cap:")
            print("   https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/rules")
            print("2. Dang nhap tai khoan 'quangkhinguynh'")
            print("3. Cuon xuong cuoi trang va bam nut: 'I Understand and Accept'")
            print("4. Sau khi bam xong, chay lai lenh:")
            print("   python \"Data/Origin/download_data.py\"")
            print("="*70 + "\n")

if __name__ == "__main__":
    download_data()
