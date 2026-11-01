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

def get_result_dir(key: str, *subdirs: str) -> Path:
    """
    Thư mục kết quả của một giai đoạn trong results/ (khóa trong [paths.results] của config.toml,
    ví dụ "tuning", "multiseed", "hierarchical"), tự tạo nếu chưa có.
    """
    path = get_path(load_config()["paths"]["results"][key]).joinpath(*subdirs)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_checkpoint_dir(key: str) -> Path:
    """Thư mục trọng số mô hình trong checkpoints/ (khóa trong [paths.checkpoints]), tự tạo nếu chưa có."""
    path = get_path(load_config()["paths"]["checkpoints"][key])
    path.mkdir(parents=True, exist_ok=True)
    return path

EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"

def load_experiment_config(name_or_path) -> dict:
    """
    Đọc file cấu hình riêng của một thí nghiệm trong Config/experiments/.
    Chấp nhận tên ngắn ("lightgbm_tuned"), tên file ("lightgbm_tuned.toml") hoặc đường dẫn đầy đủ.
    """
    path = Path(name_or_path)
    if not path.suffix:
        path = path.with_suffix(".toml")
    if not path.is_absolute() and not path.exists():
        path = EXPERIMENTS_DIR / path.name
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình thí nghiệm: {path}")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg.setdefault("experiment", {})["config_path"] = str(path)
    return cfg

def list_experiment_configs() -> list:
    """Liệt kê toàn bộ file cấu hình thí nghiệm hiện có."""
    return sorted(EXPERIMENTS_DIR.glob("*.toml"))
