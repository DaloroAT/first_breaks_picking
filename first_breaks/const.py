import os
from os import environ
from pathlib import Path
from sys import platform

from first_breaks import is_linux, is_macos, is_windows


def get_cache_folder() -> Path:
    if is_linux():
        return Path(environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "first_breaks_picking"
    elif is_macos():
        return Path.home() / "Library" / "Caches" / "first_breaks_picking"
    elif is_windows():
        return Path.home() / ".cache" / "first_breaks_picking"
    else:
        raise ValueError(f"Unexpected platform {platform}.")


PROJECT_ROOT = Path(__file__).parent.parent
CACHE_FOLDER = get_cache_folder()

DEMO_SGY_PATH = CACHE_FOLDER / "real_gather.sgy"
DEMO_SGY_URL = "https://raw.githubusercontent.com/DaloroAT/first_breaks_picking/main/data/real_gather.sgy"
DEMO_SGY_HASH = "92fe2992b57d69c6f572c672f63960cf"


MODEL_ONNX_PATH = CACHE_FOLDER / "fb.onnx"
MODEL_ONNX_URL = "https://oml.daloroserver.com/download/seis/fb.onnx"
MODEL_ONNX_HASH = "7e39e017b01325180e36885eccaeb17a"
MODEL_ONNX_HASHES = [
    # MODEL_ONNX_HASH,
    "afc03594f49b88ea61b5cf6ba8245be4",
    "3930eff8e70b4b29ab8d6def43706918",
    "cd5492eae6ed543e9c5206bc18ff8b68",
    "86ddd2a20f02201f4b1363abbabf7106",
]

TIMEOUT = 60

HIGH_DPI = bool(os.getenv("HIGH_DPI", True))

FIRST_BYTE = int(os.getenv("FIRST_BYTE", 1))
