from pathlib import Path

import pytest

from first_breaks.utils.utils import download_demo_sgy, download_model_onnx


@pytest.fixture(scope="session")
def demo_sgy() -> Path:
    return download_demo_sgy()


@pytest.fixture(scope="session")
def model_onnx() -> Path:
    return download_model_onnx()



