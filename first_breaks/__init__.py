import os
import sys
from pathlib import Path
from sys import platform

try:
    # try to load torch first to get access to CUDA and CuDNN
    import torch
except Exception:
    pass


def is_windows() -> bool:
    return "win" in platform


def is_linux() -> bool:
    return "linux" in platform


def is_macos() -> bool:
    return "darwin" in platform


if is_windows():
    extra_potential_paths = []
    extra_potential_paths.extend(list(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA").glob("*")))
    extra_potential_paths.extend(list(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit").glob("*")))
    extra_potential_paths.extend(list(Path(r"C:\Program Files\NVIDIA\CUDNN").glob("*")))
    extra_potential_paths = [str(p) for p in extra_potential_paths]
    separator = ";"

    os.environ["PATH"] = separator.join([os.environ["PATH"]] + extra_potential_paths)


if is_windows():
    base = os.path.dirname(sys.executable)
    app_pkgs = os.path.join(base, "app_packages")

    # Common locations for ORT binaries in wheels
    cand = [
        app_pkgs,
        os.path.join(app_pkgs, "onnxruntime"),
        os.path.join(app_pkgs, "onnxruntime", "capi"),
    ]
    for p in cand:
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)  # type: ignore
            except Exception:
                pass

    import onnxruntime as ort


import onnxruntime

onnxruntime.preload_dlls()
