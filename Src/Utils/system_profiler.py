import os
import sys
import json
import time
import ctypes
import shutil
import platform
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime


# =============================================================================
# 1. THÔNG TIN PHẦN CỨNG & MÔI TRƯỜNG
# =============================================================================

def _run_command(cmd: list, timeout: float = 10.0) -> str:
    """Chạy lệnh hệ thống và trả về stdout (chuỗi rỗng nếu lỗi)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _get_cpu_name() -> str:
    """Tên chip CPU theo từng hệ điều hành."""
    system = platform.system()
    if system == "Windows":
        name = _run_command(["powershell", "-NoProfile", "-Command",
                             "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"])
        if name:
            return name
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif system == "Darwin":
        name = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    return platform.processor() or "Unknown CPU"


def _get_physical_cores():
    """Số nhân vật lý (None nếu không xác định được)."""
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except ImportError:
        pass
    system = platform.system()
    if system == "Windows":
        out = _run_command(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"])
        return int(out) if out.isdigit() else None
    if system == "Darwin":
        out = _run_command(["sysctl", "-n", "hw.physicalcpu"])
        return int(out) if out.isdigit() else None
    if system == "Linux":
        try:
            cores = set()
            phys_id = core_id = None
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("physical id"):
                        phys_id = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":")[1].strip()
                        cores.add((phys_id, core_id))
            return len(cores) or None
        except Exception:
            return None
    return None


def _get_total_ram_gb():
    """Tổng RAM hệ thống (GB)."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        pass
    system = platform.system()
    if system == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return round(stat.ullTotalPhys / (1024 ** 3), 2)
    elif system == "Darwin":
        out = _run_command(["sysctl", "-n", "hw.memsize"])
        if out.isdigit():
            return round(int(out) / (1024 ** 3), 2)
    elif hasattr(os, "sysconf"):
        try:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3), 2)
        except (ValueError, OSError):
            pass
    return None


def _get_gpus() -> list:
    """Danh sách GPU NVIDIA (tên, VRAM, driver) qua nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return []
    out = _run_command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({
                "name": parts[0],
                "vram_gb": round(float(parts[1]) / 1024, 2) if parts[1].replace(".", "").isdigit() else None,
                "driver": parts[2],
            })
    return gpus


def _get_library_versions() -> dict:
    """Phiên bản các thư viện học máy chính."""
    versions = {}
    for lib in ["numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost", "optuna", "scipy"]:
        try:
            module = __import__(lib)
            versions[lib] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not installed"
    return versions


def get_hardware_info() -> dict:
    """
    Tự động phát hiện thông tin phần cứng & môi trường chạy thí nghiệm:
    CPU (tên chip, số nhân/luồng), GPU (tên card, VRAM), RAM, hệ điều hành, phiên bản thư viện.
    """
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": {
            "name": _get_cpu_name(),
            "physical_cores": _get_physical_cores(),
            "logical_threads": os.cpu_count(),
        },
        "ram_gb": _get_total_ram_gb(),
        "gpus": _get_gpus(),
        "libraries": _get_library_versions(),
    }


def format_hardware_report(info: dict) -> str:
    """Định dạng thông tin phần cứng thành đoạn văn bản dễ đọc."""
    gpu_lines = [f"   - {g['name']} | VRAM {g['vram_gb']} GB | Driver {g['driver']}" for g in info["gpus"]] \
        or ["   - Không phát hiện GPU NVIDIA"]
    libs = ", ".join(f"{k}={v}" for k, v in info["libraries"].items())
    lines = [
        "=" * 80,
        " CẤU HÌNH PHẦN CỨNG & MÔI TRƯỜNG THỰC NGHIỆM",
        "=" * 80,
        f" - Thời điểm   : {info['timestamp']}",
        f" - Hệ điều hành: {info['os']} [{info['machine']}]",
        f" - Python      : {info['python']}",
        f" - CPU         : {info['cpu']['name']} "
        f"({info['cpu']['physical_cores']} nhân / {info['cpu']['logical_threads']} luồng)",
        f" - RAM         : {info['ram_gb']} GB",
        " - GPU         :",
        *gpu_lines,
        f" - Thư viện    : {libs}",
        "=" * 80,
    ]
    return "\n".join(lines)


def save_hardware_info(path) -> dict:
    """Ghi thông tin phần cứng ra file JSON và trả về dict."""
    info = get_hardware_info()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


# =============================================================================
# 2. BỘ ĐẾM THỜI GIAN
# =============================================================================

class Timer:
    """
    Bộ đếm thời gian chính xác dựa trên time.perf_counter():
        with Timer() as t:
            model.fit(...)
        print(t.elapsed)
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        return False


# =============================================================================
# 3. BỘ ĐẾM THAM SỐ MÔ HÌNH (GBDT)
# =============================================================================

def _file_size_kb(save_fn, suffix: str) -> float:
    """Lưu mô hình ra file tạm để đo kích thước (KB)."""
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        save_fn(tmp_path)
        return round(os.path.getsize(tmp_path) / 1024, 2)
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _count_lightgbm(model) -> dict:
    booster = model.booster_ if hasattr(model, "booster_") else model
    dump = booster.dump_model()
    tree_info = dump["tree_info"]
    n_leaves = int(sum(t["num_leaves"] for t in tree_info))
    n_splits = int(sum(t["num_leaves"] - 1 for t in tree_info))
    return {
        "n_trees": len(tree_info),
        "n_leaves": n_leaves,
        "n_split_nodes": n_splits,
        "num_leaves_param": getattr(model, "num_leaves", None),
        "model_size_kb": _file_size_kb(lambda p: booster.save_model(p), ".txt"),
    }


def _count_xgboost(model) -> dict:
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    trees = booster.get_dump(dump_format="json")
    n_leaves = sum(t.count('"leaf"') for t in trees)
    n_splits = sum(t.count('"split"') for t in trees)
    raw_json = booster.save_raw(raw_format="json")
    return {
        "n_trees": len(trees),
        "n_leaves": int(n_leaves),
        "n_split_nodes": int(n_splits),
        "max_depth_param": getattr(model, "max_depth", None),
        "booster_json_kb": round(len(raw_json) / 1024, 2),
    }


def _count_catboost(model) -> dict:
    leaf_counts = model.get_tree_leaf_counts()
    n_leaves = int(sum(leaf_counts))
    # Cây đối xứng (oblivious): mỗi tầng 1 split dùng chung -> số split = depth = log2(leaves)
    n_splits = int(sum(int(c).bit_length() - 1 for c in leaf_counts))
    return {
        "n_trees": int(model.tree_count_),
        "n_leaves": n_leaves,
        "n_split_nodes": n_splits,
        "model_size_kb": _file_size_kb(lambda p: model.save_model(p), ".cbm"),
    }


def count_model_parameters(model, model_name: str = None) -> dict:
    """
    Đếm "số tham số" của mô hình cây tăng cường:
      n_parameters = n_leaves (giá trị lá) + n_split_nodes (cặp feature/ngưỡng chia).
    - LightGBM: tổng số cây, tổng số lá, tổng số node split.
    - XGBoost: số cây, max_depth, kích thước booster JSON.
    - CatBoost: tree_count_, kích thước mô hình.
    """
    name = (model_name or type(model).__module__).lower()
    if "lightgbm" in name or "lgbm" in name:
        stats = _count_lightgbm(model)
    elif "xgboost" in name or "xgb" in name:
        stats = _count_xgboost(model)
    elif "catboost" in name:
        stats = _count_catboost(model)
    else:
        raise ValueError(f"Không hỗ trợ đếm tham số cho mô hình: {name}")
    stats["n_parameters"] = int(stats["n_leaves"] + stats["n_split_nodes"])
    return stats


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(format_hardware_report(get_hardware_info()))
