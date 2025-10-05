import colorsys
import hashlib
import inspect
import io
import random
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

from first_breaks.const import (
    DEMO_SGY_HASH,
    DEMO_SGY_PATH,
    DEMO_SGY_URL,
    MODEL_ONNX_HASH,
    MODEL_ONNX_PATH,
    MODEL_ONNX_URL,
    TIMEOUT,
)

TScalar = Union[int, float, np.number]
TTimeType = Union[TScalar, List[TScalar], Tuple[TScalar, ...], np.ndarray]


class InvalidHash(Exception):
    pass


def chunk_iterable(it: Iterable[Any], size: int) -> List[Tuple[Any, ...]]:
    it = iter(it)
    return list(iter(lambda: tuple(islice(it, size)), ()))


def get_io(source: Union[Path, str, bytes, io.BytesIO, io.FileIO], mode: str = "r") -> Union[io.BytesIO, io.FileIO]:
    if isinstance(source, (io.BytesIO, io.FileIO)):
        return source
    elif isinstance(source, (Path, str)):
        source = Path(source).resolve()
        if "r" in mode:
            if not source.exists():
                raise FileNotFoundError("There is no file: ", str(source))
        descriptor = io.FileIO(str(source), mode=mode)
    elif isinstance(source, bytes):
        descriptor = io.BytesIO(source)  # type: ignore
    else:
        raise TypeError("Not supported type")
    return descriptor


def calc_hash(source: Union[Path, str, bytes, io.BytesIO, io.FileIO]) -> str:
    hash_md5 = hashlib.md5()
    source = get_io(source, mode="rb")
    source.seek(0)
    for chunk in iter(lambda: source.read(4096), b""):  # type: ignore
        hash_md5.update(chunk)
    source.close()
    return hash_md5.hexdigest()


def download_by_url(url: str, fname: Optional[Union[str, Path]], timeout: float = TIMEOUT) -> bytes:
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    buffer = io.BytesIO()

    with tqdm(
        desc=f"Downloading {url}",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            buffer.write(data)
            bar.update(len(data))

        buffer.seek(0)
        content = buffer.getvalue()

        if fname:
            Path(fname).parent.mkdir(exist_ok=True, parents=True)
            with open(fname, "wb+") as f:
                f.write(content)
            bar.set_description(f"File {url} saved to '{Path(fname).resolve()}'")

    return content


def download_and_validate_file(
    fname: Union[str, Path], url: str, md5: str, timeout: float = TIMEOUT
) -> Union[str, Path]:
    if not (Path(fname).exists() and calc_hash(fname) == md5):
        download_by_url(url=url, fname=fname, timeout=timeout)
    md5_last = calc_hash(fname)
    if md5_last != md5:
        raise InvalidHash(f"Hash for file {Path(fname).resolve()} in invalid. Got {md5_last}, expected {md5}")
    return fname


def download_demo_sgy(
    fname: Union[str, Path] = DEMO_SGY_PATH,
    url: str = DEMO_SGY_URL,
    md5: str = DEMO_SGY_HASH,
) -> Union[str, Path]:
    return download_and_validate_file(fname=fname, url=url, md5=md5)


def download_model_onnx(
    fname: Union[str, Path] = MODEL_ONNX_PATH,
    url: str = MODEL_ONNX_URL,
    md5: str = MODEL_ONNX_HASH,
) -> Union[str, Path]:
    return download_and_validate_file(fname=fname, url=url, md5=md5)


def multiply_iterable_by(sample: TTimeType, multiplier: float, cast_to: Optional[Any] = None) -> TTimeType:
    if isinstance(sample, (int, float, str)):
        result = sample * multiplier  # type: ignore
        return cast_to(result) if cast_to is not None else result
    elif isinstance(sample, (np.number, np.ndarray)):
        result = sample * multiplier
        return result.astype(cast_to) if cast_to is not None else result
    elif isinstance(sample, list):
        return list(multiply_iterable_by(val, multiplier, cast_to) for val in sample)
    elif isinstance(sample, tuple):
        return tuple(multiply_iterable_by(val, multiplier, cast_to) for val in sample)
    else:
        raise TypeError("Invalid type for samples")


class UnitsConverter:
    def __init__(
        self,
        *args: Any,
        sgy_mcs: Optional[Union[int, float]] = None,
        sgy_ms: Optional[Union[int, float]] = None,
    ):
        if args:
            raise ValueError("Specify explicitly either `sgy_mcs`or `sgy_ms` as keyword argument")
        if (sgy_mcs is None and sgy_ms is None) or (sgy_mcs is not None and sgy_ms is not None):
            raise RuntimeError("One and only one of `sgy_mcs` or `sgy_ms` must be specified")
        elif sgy_mcs is not None:
            self.sgy_mcs = sgy_mcs
            self.sgy_ms = self.mcs2ms(sgy_mcs)  # type: ignore
        elif sgy_ms is not None:
            self.sgy_mcs = self.ms2mcs(sgy_ms)  # type: ignore
            self.sgy_ms = sgy_ms
        else:
            raise RuntimeError("Init error")

    @staticmethod
    def ms2mcs(sample: TTimeType, cast_to: Any = int) -> TTimeType:
        return multiply_iterable_by(sample, 1000, cast_to)

    @staticmethod
    def mcs2ms(sample: TTimeType, cast_to: Any = float) -> TTimeType:
        return multiply_iterable_by(sample, 0.001, cast_to)

    def ms2index(self, sample: TTimeType, cast_to: Any = int) -> TTimeType:
        return multiply_iterable_by(sample, 1 / self.sgy_ms, cast_to)  # type: ignore

    def mcs2index(self, sample: TTimeType, cast_to: Any = int) -> TTimeType:
        return multiply_iterable_by(sample, 1 / self.sgy_mcs, cast_to)

    def index2ms(self, sample: TTimeType, cast_to: Any = float) -> TTimeType:
        return multiply_iterable_by(sample, self.sgy_ms, cast_to)  # type: ignore

    def index2mcs(self, sample: TTimeType, cast_to: Any = int) -> TTimeType:
        return multiply_iterable_by(sample, self.sgy_mcs, cast_to)


def remove_unused_kwargs(kwargs: Dict[str, Any], constructor: Any) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in inspect.signature(constructor).parameters}


def _color_generator() -> Generator[Tuple[int, ...], None, None]:
    golden_ratio = 0.618033988749895
    hue = random.random()  # start from a random position
    while True:
        hue += golden_ratio
        hue %= 1
        yield tuple(int(255 * v) for v in colorsys.hsv_to_rgb(hue, 0.5, 0.95))


cgen = _color_generator()


def generate_color() -> Tuple[int, ...]:
    return next(cgen)


def resolve_postime2xy(vsp_view: bool, position: Any, time: Any) -> Tuple[Any, Any]:
    if vsp_view:
        return time, position
    else:
        return position, time


def resolve_xy2postime(vsp_view: bool, x: Any, y: Any) -> Tuple[Any, Any]:
    if vsp_view:
        return y, x
    else:
        return x, y


def as_list(sequence: Iterable[Any]) -> List[Any]:
    if isinstance(sequence, (np.ndarray, pd.Series)):
        return sequence.tolist()
    elif isinstance(sequence, (list, tuple)):
        return list(sequence)
    elif isinstance(sequence, (np.number, float, int)):
        return [sequence]
    else:
        raise TypeError("Unsupported 'sequence' type")
