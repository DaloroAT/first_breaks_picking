import hashlib
import io
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Union, Optional

import numpy as np
import requests

from first_breaks.const import DEMO_SGY_URL, DEMO_SGY_PATH, TIMEOUT, DEMO_SGY_HASH, MODEL_ONNX_URL, MODEL_ONNX_HASH, \
    MODEL_ONNX_PATH


TScalar = Union[int, float, np.number]
TTimeType = Union[TScalar, List[TScalar], Tuple[TScalar, ...], np.ndarray]


class InvalidHash(Exception):
    pass


def chunk_iterable(it: Iterable[Any], size: int) -> List[Tuple[Any, ...]]:
    it = iter(it)
    return list(iter(lambda: tuple(islice(it, size)), ()))


def get_io(source: Union[Path, str, bytes], mode: str = 'r') -> Union[io.BytesIO, io.FileIO]:
    if isinstance(source, (Path, str)):
        source = Path(source).resolve()
        if 'r' in mode:
            if not source.exists():
                raise FileNotFoundError("There is no file: ", str(source))
        descriptor = io.FileIO(str(source), mode=mode)
    elif isinstance(source, bytes):
        descriptor = io.BytesIO(source)
    else:
        raise TypeError('Not supported type')
    return descriptor


def calc_hash(source: Union[Path, str, bytes, io.BytesIO, io.FileIO]) -> str:
    hash_md5 = hashlib.md5()
    if not isinstance(source, (io.BytesIO, io.FileIO)):
        source = get_io(source, mode='rb')
    for chunk in iter(lambda: source.read(4096), b""):
        hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_by_url(url: str, fname: Optional[Union[str, Path]], timeout: float = TIMEOUT) -> Optional[bytes]:
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        response.raise_for_status()
    else:
        if fname:
            Path(fname).parent.mkdir(exist_ok=True, parents=True)
            with open(fname, 'wb+') as f:
                f.write(response.content)
        return response.content


def download_and_validate_file(fname: Union[str, Path],
                               url: str,
                               md5: str,
                               timeout: float = TIMEOUT) -> Union[str, Path]:
    if not (Path(fname).exists() and calc_hash(fname) == md5):
        download_by_url(url=url, fname=fname, timeout=timeout)
    md5_last = calc_hash(fname)
    if md5_last != md5:
        raise InvalidHash(f'Hash for file {Path(fname).resolve()} in invalid. Got {md5_last}, expected {md5}')
    return fname


def download_demo_sgy(fname: Union[str, Path] = DEMO_SGY_PATH,
                      url: str = DEMO_SGY_URL,
                      md5: str = DEMO_SGY_HASH) -> Path:
    return download_and_validate_file(fname=fname, url=url, md5=md5)


def download_model_onnx(fname: Union[str, Path] = MODEL_ONNX_PATH,
                        url: str = MODEL_ONNX_URL,
                        md5: str = MODEL_ONNX_HASH) -> Path:
    return download_and_validate_file(fname=fname, url=url, md5=md5)


def sample2ms(sample: TTimeType, dt_ms: float) -> TTimeType:
    if isinstance(sample, (int, float, (np.number, np.ndarray))):
        return sample * dt_ms
    elif isinstance(sample, list):
        return list(sample2ms(val, dt_ms) for val in sample)
    elif isinstance(sample, tuple):
        return tuple(sample2ms(val, dt_ms) for val in sample)
    else:
        raise TypeError('Invalid type for samples')
