try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"

def load_config() -> dict:
    """Đọc tệp cấu hình config.toml và trả về dict cấu hình."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {CONFIG_PATH}")
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)

def get_path(relative_str: str) -> Path:
    """Chuyển đổi đường dẫn tương đối từ config thành Path object tuyệt đối an toàn."""
    return BASE_DIR / relative_str
