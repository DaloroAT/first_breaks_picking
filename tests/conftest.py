import shutil
from pathlib import Path

import pytest

from first_breaks.const import PROJECT_ROOT
from first_breaks.utils.utils import download_demo_sgy, download_model_onnx


@pytest.fixture(scope="session")
def demo_sgy() -> Path:
    return download_demo_sgy()


@pytest.fixture(scope="session")
def model_onnx() -> Path:
    return download_model_onnx()


@pytest.fixture(scope="session")
def test_logs_dir() -> Path:
    logs_dir = Path('tests_logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    yield logs_dir
    shutil.rmtree(logs_dir, ignore_errors=True)



