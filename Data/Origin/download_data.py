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

    if local_kaggle_json.exists():
        user_kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_kaggle_json, user_kaggle_json)

        try:
            with open(local_kaggle_json, 'r', encoding='utf-8') as f:
                creds = json.load(f)
                os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
                os.environ["KAGGLE_KEY"] = creds.get("key", "")
                print(f"[SUCCESS] Loaded Kaggle account: {creds.get('username')}")
        except Exception as e:
            print(f"[WARN] Error reading kaggle.json: {e}")

    elif user_kaggle_json.exists():
        try:
            with open(user_kaggle_json, 'r', encoding='utf-8') as f:
                creds = json.load(f)
                os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
                os.environ["KAGGLE_KEY"] = creds.get("key", "")
                print(f"[OK] Loaded Kaggle account from ~/.kaggle: {creds.get('username')}")
        except Exception as e:
            print(f"[WARN] Error reading ~/.kaggle/kaggle.json: {e}")

    else:
        print("[WARN] kaggle.json file not found!")

def download_data():
    origin_dir = Path(__file__).resolve().parent
    data_dir = origin_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    setup_kaggle_credentials()

    print("\n=== Starting Walmart competition data download ===")
    import kagglehub

    try:
        path = kagglehub.competition_download('walmart-recruiting-store-sales-forecasting')
        
        print(f"\n[SUCCESS] Data cache path: {path}")

        print("[...] Syncing and extracting data into Data/Origin/data...")

        for file in Path(path).iterdir():
            dest_file = data_dir / file.name
            if file.is_file():
                shutil.copy2(file, dest_file)
                if dest_file.suffix == '.zip':
                    print(f"     -> Extracting: {dest_file.name} ...")
                    with zipfile.ZipFile(dest_file, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)

        print(f"\n[SUCCESS] All data files are ready at: {data_dir}")

        print("\nList of available files:")
        for f in sorted(data_dir.iterdir()):
            if f.is_file() and f.suffix in ['.csv', '.zip']:
                print(f" - {f.name:25} ({f.stat().st_size / (1024*1024):6.2f} MB)")

    except Exception as e:
        error_msg = str(e)
        print(f"\n[DOWNLOAD ERROR]: {error_msg}")
        if "401" in error_msg or "permission" in error_msg.lower() or "rules" in error_msg.lower():
            print("\n" + "="*70)
            print("(!) Important step to unlock download permission:")
            print("Your Kaggle account needs to accept the competition rules.")
            print("1. Open your browser and visit:")
            print("   https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/rules")
            print("2. Log in with your Kaggle account")
            print("3. Scroll to the bottom and click: 'I Understand and Accept'")
            print("4. After accepting, re-run this command:")
            print("   python \"Data/Origin/download_data.py\"")
            print("="*70 + "\n")

if __name__ == "__main__":
    download_data()
