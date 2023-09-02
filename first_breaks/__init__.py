import os
from pathlib import Path
from sys import platform


VERSION = "0.4.0b"


def is_windows() -> bool:
    return "win" in platform


def is_linux() -> bool:
    return "linux" in platform


def is_macos() -> bool:
    return "darwin" in platform


def patch_pathenv_with_default_gpu_paths() -> None:
    if is_windows():
        extra_potential_paths = []
        extra_potential_paths.extend(list(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA").glob("*")))
        extra_potential_paths.extend(list(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit").glob("*")))
        extra_potential_paths.extend(list(Path(r"C:\Program Files\NVIDIA\CUDNN").glob("*")))
        extra_potential_paths = [str(p) for p in extra_potential_paths]
        separator = ";"
    elif is_linux():
        extra_potential_paths = ["/usr/local/cuda", "/usr/include", "/lib/x86_64-linux-gnu"]
        separator = ":"
    else:
        extra_potential_paths = []
        separator = ":"

    os.environ["PATH"] = separator.join([os.environ["PATH"]] + extra_potential_paths)


patch_pathenv_with_default_gpu_paths()
