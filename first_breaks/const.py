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

TIMEOUT = 60

HIGH_DPI = os.getenv("HIGH_DPI", True)
