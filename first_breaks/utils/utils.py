import hashlib
import io
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Union


def chunk_iterable(it: Iterable[Any], size: int) -> List[Tuple[Any, ...]]:
    it = iter(it)
    return list(iter(lambda: tuple(islice(it, size)), ()))


def get_io(source: Union[Path, str, bytes], mode: str = 'r') -> Union[io.BytesIO, io.FileIO]:
    if isinstance(source, (Path, str)):
        source = Path(source).resolve()
        if 'r' in mode:
            if not source.exists():
                raise FileNotFoundError("There is no file: ", str(source))
        desciptor = io.FileIO(str(source), mode='r')
    elif isinstance(source, bytes):
        desciptor = io.BytesIO(source)
    else:
        raise TypeError('Not supported type')
    return desciptor


def calc_hash(source: Union[Path, str, bytes, io.RawIOBase]) -> str:
    hash_md5 = hashlib.md5()
    if not isinstance(source, io.RawIOBase):
        source = get_io(source, mode='rb')
    for chunk in iter(lambda: source.read(4096), b""):
        hash_md5.update(chunk)
    return hash_md5.hexdigest()
