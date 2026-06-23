from typing import Mapping

import numpy as np

from first_breaks.sgy import Endianness, get_num_bytes
from first_breaks.sgy.headers import FORMAT_TO_NUMPY_DTYPE, HeaderInfo, InvalidHeaders


def dtype_for_format(fmt: str, endianness: Endianness) -> np.dtype:
    if fmt.endswith("s"):
        return np.dtype(f"S{get_num_bytes(fmt)}")
    if fmt not in FORMAT_TO_NUMPY_DTYPE:
        raise InvalidHeaders(f"Format is not interpretable by NumPy: {fmt!r}")
    return np.dtype(Endianness(endianness).value + FORMAT_TO_NUMPY_DTYPE[fmt])


def block_dtype(infos: Mapping[str, HeaderInfo], endianness: Endianness, block_size: int) -> np.dtype:
    return np.dtype(
        {
            "names": list(infos.keys()),
            "formats": [dtype_for_format(info.format, endianness) for info in infos.values()],
            "offsets": [info.offset for info in infos.values()],
            "itemsize": block_size,
        }
    )
