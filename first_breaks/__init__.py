import os
import sys
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
