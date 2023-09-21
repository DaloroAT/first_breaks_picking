import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from first_breaks import is_windows

ONNX_DEVICE2PROVIDER = {"cuda": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider"}


def is_onnx_gpu_version_installed() -> bool:
    try:
        return ONNX_DEVICE2PROVIDER["cuda"] in ort.get_available_providers()
    except Exception:
        return False


def _colorize(txt: str) -> str:
    _blue_code = "\033[94m"
    _reset = "\033[0m"

    return f"{_blue_code}{txt}{_reset}"


GPU_USSAGE_MESSAGE = """
Before using GPU acceleration, check https://developer.nvidia.com/cuda-gpus that your GPU is CUDA compatible.
"""


CUDA_INSTALLATION_MESSAGE = f"""
Install CUDA Toolkit in one of the following ways:
\t1) Using the official website https://developer.nvidia.com/cuda-toolkit-archive. Select one \
of 11.6, 11.7, 11.8 versions. Version 12 also may work, but versions >=11.6 are recommended.
\t2) Using Conda {_colorize('conda install -c anaconda "cudatoolkit>=11.6,<12"')}.
"""


_CUDNN_LINUX_URL = "https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux"
_CUDNN_WINDOWS_URL = "https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows"

CUDNN_INSTALLATION_MESSAGE = f"""
Install cuDNN toolkit in one of the following ways:
\t1) Using official website {_CUDNN_WINDOWS_URL if is_windows() else _CUDNN_LINUX_URL}.
\t2) Using Conda {_colorize('conda install -c anaconda "cudnn=8.2"')}.
"""


_ZLIB_LIBUX_URL = "https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-linux"
_ZLIB_WINDOWS_URL = "https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-windows"

ZLIB_INSTALLATION_MESSAGE = f"""
Install ZLib {_ZLIB_WINDOWS_URL if is_windows() else _ZLIB_LIBUX_URL}.
"""


FULL_INSTALLATION_MESSAGE = "".join(
    [GPU_USSAGE_MESSAGE, CUDA_INSTALLATION_MESSAGE, CUDNN_INSTALLATION_MESSAGE, ZLIB_INSTALLATION_MESSAGE]
)


def raise_onnx_cuda_init(add_installation_instruction: bool = True) -> None:
    try:
        ort.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.float32), "cuda", 0)
    except Exception as exc:
        if add_installation_instruction:
            err_message = f"{str(exc)}\n" f"{'_' * 20}\n" f"Recommendations:\n" f"{FULL_INSTALLATION_MESSAGE}"
            raise type(exc)(err_message).with_traceback(exc.__traceback__)
        else:
            raise exc


def is_onnx_cuda_initializable() -> bool:
    try:
        raise_onnx_cuda_init()
        return True
    except Exception:
        return False


def is_zlib_installed() -> bool:
    if is_windows():
        for path in os.environ['PATH'].split(";"):
            if (Path(path) / "zlibwapi.dll"):
                pass


def is_onnx_cuda_available() -> bool:
    return is_onnx_gpu_version_installed() and is_onnx_cuda_initializable()


ONNX_CUDA_AVAILABLE = is_onnx_cuda_available()
