from enum import Enum

import numpy as np
import onnxruntime as ort


class Engine(Enum):
    CUDA = "cuda"
    OPENVINO = "openvino"
    CPU = "cpu"


ONNX_DEVICE2PROVIDER = {
    Engine.CUDA.value: "CUDAExecutionProvider",
    Engine.OPENVINO.value: "OpenVINOExecutionProvider",
    Engine.CPU.value: "CPUExecutionProvider",
}


FULL_INSTALLATION_MESSAGE = """
\nCheck requirements for installed onnxruntime:\n"
1) https://onnxruntime.ai/docs/install/
2) https://onnxruntime.ai/docs/execution-providers/
"""


def raise_onnx_device_init(device: str) -> None:
    try:
        ort.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.float32), device, 0)
    except Exception as exc:
        err_message = f"{str(exc)}\n" f"{'_' * 20}\n" f"Recommendations:\n" f"{FULL_INSTALLATION_MESSAGE}"
        raise type(exc)(err_message).with_traceback(exc.__traceback__)


def is_onnx_device_available(device: str) -> bool:
    try:
        raise_onnx_device_init(device=device)
        return True
    except Exception:
        return False


def is_onnx_cuda_available() -> bool:
    return is_onnx_device_available(Engine.CUDA.value)


def is_onnx_openvino_available() -> bool:
    return is_onnx_device_available(Engine.OPENVINO.value)


def get_recommended_device() -> str:
    if is_onnx_cuda_available():
        return Engine.CUDA.value
    elif is_onnx_openvino_available():
        return Engine.OPENVINO.value
    else:
        return Engine.CPU.value
