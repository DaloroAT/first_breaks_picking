import os
from os import environ
from pathlib import Path
from sys import platform


SUPPORTED_TORCH = '2.0'


def is_windows() -> bool:
    return "win" in platform


def is_linux() -> bool:
    return "linux" in platform


def is_macos() -> bool:
    return "darwin" in platform


def get_cache_folder() -> Path:
    if is_linux():
        return Path(environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "first_breaks_picking"
    elif is_macos():
        return Path.home() / "Library" / "Caches" / "first_breaks_picking"
    elif is_windows():
        return Path.home() / ".cache" / "first_breaks_picking"
    else:
        raise ValueError(f"Unexpected platform {platform}.")


def is_torch_available() -> bool:
    try:
        import torch
        return True
    except (ModuleNotFoundError, ImportError):
        return False


def is_cuda_available() -> bool:
    if is_torch_available():
        import torch
        return torch.cuda.is_available()
    else:
        return False


def raise_if_no_torch() -> None:
    if not is_torch_available():
        raise ModuleNotFoundError("To use the features of 'torch', you need to install it. "
                                  'Follow this official link: https://pytorch.org/get-started/locally/')
    else:
        import torch
        if not torch.__version__.startswith(SUPPORTED_TORCH):
            raise ImportError(f"Now the library works only with version '{SUPPORTED_TORCH}' of 'torch'. "
                              'We may loosen the requirements in the future. '
                              'Follow this official link to install supported version: '
                              'https://pytorch.org/get-started/locally/')


PROJECT_ROOT = Path(__file__).parent.parent
CACHE_FOLDER = get_cache_folder()

DEMO_SGY_PATH = CACHE_FOLDER / "real_gather.sgy"
DEMO_SGY_URL = "https://raw.githubusercontent.com/DaloroAT/first_breaks_picking/main/data/real_gather.sgy"
DEMO_SGY_HASH = "92fe2992b57d69c6f572c672f63960cf"


MODEL_ONNX_PATH = CACHE_FOLDER / "fb.onnx"
MODEL_ONNX_URL = "https://oml.daloroserver.com/download/seis/fb.onnx"
MODEL_ONNX_HASH = "7e39e017b01325180e36885eccaeb17a"


MODEL_TORCH_PATH = CACHE_FOLDER / "fb_torch_unet3plus_11483cfe0c10f32a4bedc8b1351054eb.pth"
MODEL_TORCH_URL = "https://oml.daloroserver.com/download/seis/fb_torch_unet3plus_11483cfe0c10f32a4bedc8b1351054eb.pth"
MODEL_TORCH_HASH = "11483cfe0c10f32a4bedc8b1351054eb"

TIMEOUT = 60

HIGH_DPI = os.getenv("HIGH_DPI", True)
