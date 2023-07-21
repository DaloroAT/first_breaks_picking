import json
from os import environ
from pathlib import Path
from pprint import pprint
from typing import Union, Optional, List

import onnxruntime as ort

from first_breaks.const import is_windows, is_macos, is_linux

ONNX_DEVICE2PROVIDER = {"cuda": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider"}


def is_onnx_gpu_installed() -> bool:
    try:
        return ONNX_DEVICE2PROVIDER["cuda"] in ort.get_available_providers()
    except Exception:
        return False


NECESSARY_CUDA = ["11.6", "11.7", "11.8"]
NECESSARY_CUDA_STR = ', '.join(NECESSARY_CUDA)


class _GPUSetupBaseException(Exception):
    pass


class GPUNotFound(_GPUSetupBaseException):
    pass


class CUDANotConfigured(_GPUSetupBaseException):
    pass


class CUDNNNotConfigured(_GPUSetupBaseException):
    pass


class ZLibNotConfigured(_GPUSetupBaseException):
    pass


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
of {NECESSARY_CUDA_STR} versions.
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


def _check_if_path_potentially_with_cuda_toolkit(path: Union[str, Path]) -> bool:
    path = Path(path)

    if path.is_file():
        return False

    if not (path / "version.json").exists():
        return False

    try:
        with open(path / "version.json") as fin:
            meta = json.load(fin)
        assert "cuda" in meta
        return True
    except Exception:
        return False


def _check_cuda_version_file(path: Union[str, Path]) -> Optional[CUDANotConfigured]:
    path = Path(path)

    try:
        with open(path) as fin:
            meta = json.load(fin)

        for key, version in [("cuda", "11"),
                             ("cuda_cudart", "11"),
                             ("libcufft", "10"),
                             ("libcurand", "10"),
                             ("libcublas", "11"),
                             ]:
            if not meta[key]["version"].startswith(version):
                return CUDANotConfigured(f"CUDA version in {path} is not valid. Install one"
                                         f" of {NECESSARY_CUDA_STR} versions. ")
    except Exception:
        return CUDANotConfigured(f"File {path} is invalid. ")


def validate_cuda(paths_list: List[Path]) -> None:
    paths_list = paths_list.copy()
    paths_list += [path.parent for path in paths_list]  # PATH can contain CUDA folder or /bin folder of CUDA
    paths_with_cuda = sorted(set(path for path in paths_list if _check_if_path_potentially_with_cuda_toolkit(path)))

    if not paths_with_cuda:
        raise CUDANotConfigured(f"CUDA folder not found in PATH variable. \n"
                                f"{GPU_USSAGE_MESSAGE}"
                                f"{CUDA_INSTALLATION_MESSAGE}")

    cuda_found = False
    errors = []
    for path in paths_with_cuda:
        res = _check_cuda_version_file(path / "version.json")
        if res is None:
            cuda_found = True
            break
        else:
            errors.append(res)

    if not cuda_found:
        raise errors[0]


def validate_cudnn(paths_list: List[Path]) -> None:
    paths_list = paths_list.copy()
    paths_list += [path.parent for path in paths_list]  # PATH can contain CUDA folder or /bin folder of CUDA

    bin_extension = "dll" if is_windows() else "so"

    cudnn_found = False
    for path in paths_list:
        cudnn_found = any((path / "include").glob("cudnn*.h")) and \
                      any((path / "lib").rglob("cudnn*.lib")) and \
                      any((path / "bin").glob(f"cudnn*.{bin_extension}"))
        if cudnn_found:
            break

    if not cudnn_found:
        raise CUDNNNotConfigured(f"Folder with cuDNN components not found in PATH.\n"
                                 f"{GPU_USSAGE_MESSAGE}"
                                 f"{CUDNN_INSTALLATION_MESSAGE}")


def validate_zlib(paths_list: List[Path]) -> None:
    if is_windows():
        if not any((path / "zlibwapi.dll").exists() for path in paths_list):
            raise ZLibNotConfigured(f"Folder with Zlib not found.\n"
                                    f"{ZLIB_INSTALLATION_MESSAGE}")
    else:
        raise NotImplementedError('Do on linux')


def raise_if_gpu_not_configured() -> None:
    if is_macos():
        raise GPUNotFound("There is no CUDA on Darwin system")
    elif is_windows():
        path_sep = ";"
        all_paths = [Path(path) for path in environ["PATH"].split(path_sep)
                     if "windows\\system32" not in str(path).lower()]
    else:
        path_sep = ":"
        all_paths = [Path(path) for path in environ["PATH"].split(path_sep)]

    validate_cuda(all_paths)
    validate_cudnn(all_paths)
    validate_zlib(all_paths)


def is_onnx_cuda_available() -> bool:
    if is_onnx_gpu_installed():
        try:
            raise_if_gpu_not_configured()
            return True
        except _GPUSetupBaseException:
            return False
    else:
        return False


ONNX_CUDA_AVAILABLE = is_onnx_cuda_available()



