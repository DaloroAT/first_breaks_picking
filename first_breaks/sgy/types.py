from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np


SizeHW = Tuple[int, int]
SourceInput = Union[str, Path, bytes, np.ndarray]

FILE_HEADER_SIZE = 3600
TRACE_HEADER_SIZE = 240
SGY_REVISION_OFFSET = 3500
MCS_TO_MS_FACTOR = 1e-3
MS_TO_HZ_FACTOR = 1000


class DataFormat(IntEnum):
    IBM_FLOAT = 1
    INT32 = 2
    INT16 = 3
    FIXED_POINT = 4
    IEEE_FLOAT = 5
    IEEE_DOUBLE = 6

    @classmethod
    def is_valid(cls, value: Union["DataFormat", int]) -> bool:
        try:
            cls(value)
        except ValueError:
            return False
        return True

    @classmethod
    def is_supported_for_reading(cls, value: Union["DataFormat", int]) -> bool:
        return cls(value) in (cls.IBM_FLOAT, cls.INT32, cls.INT16, cls.IEEE_FLOAT, cls.IEEE_DOUBLE)

    @classmethod
    def is_supported_for_writing(cls, value: Union["DataFormat", int]) -> bool:
        return cls(value) in (cls.IBM_FLOAT, cls.INT32, cls.INT16, cls.IEEE_FLOAT, cls.IEEE_DOUBLE)


class Endianness(str, Enum):
    BIG = ">"
    LITTLE = "<"

    @classmethod
    def is_valid(cls, value: Union["Endianness", str]) -> bool:
        try:
            cls(value)
        except ValueError:
            return False
        return True


# Backward-compatible spelling used by the old public API.
Endianess = Endianness

DEFAULT_ENDIANESS = Endianness.BIG
DEFAULT_DATA_FORMAT = DataFormat.IEEE_FLOAT


class SGYRevision(IntEnum):
    REV_0 = 0x0000
    REV_1_0 = 0x0100
    REV_2_0 = 0x0200
    REV_2_1 = 0x0201

    @classmethod
    def from_header_value(cls, value: Union["SGYRevision", int]) -> "SGYRevision":
        try:
            return cls(value)
        except ValueError as exc:
            raise InvalidSGY(f"Unknown SEG-Y revision header value: {value!r}") from exc

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        endianness: Union[Endianness, str] = Endianness.BIG,
    ) -> "SGYRevision":
        if len(raw) != 2:
            raise InvalidSGY(f"SEG-Y revision header requires 2 bytes, got {len(raw)}")
        byte_order = "big" if Endianness(endianness) == Endianness.BIG else "little"
        return cls.from_header_value(int.from_bytes(raw, byteorder=byte_order, signed=False))

    def is_supported(self) -> bool:
        return self == SUPPORTED_SGY_REVISION


SUPPORTED_SGY_REVISION = SGYRevision.REV_0


FORMAT_TO_BYTES_PER_SAMPLE: Dict[DataFormat, int] = {
    DataFormat.IBM_FLOAT: 4,
    DataFormat.INT32: 4,
    DataFormat.INT16: 2,
    DataFormat.FIXED_POINT: 4,
    DataFormat.IEEE_FLOAT: 4,
    DataFormat.IEEE_DOUBLE: 8,
}

FORMAT_TO_DTYPE: Dict[DataFormat, str] = {
    DataFormat.INT32: "i4",
    DataFormat.INT16: "i2",
    DataFormat.IEEE_FLOAT: "f4",
    DataFormat.IEEE_DOUBLE: "f8",
}


class NotImplementedReader(Exception):
    pass


class InvalidSGY(Exception):
    pass


class SGYInitParamsError(Exception):
    pass


class InvalidSamplesSlice(Exception):
    pass


class UnsupportedSGYRevision(Exception):
    pass


def ensure_supported_revision(revision: Union[SGYRevision, int]) -> None:
    parsed_revision = SGYRevision.from_header_value(revision)
    if not parsed_revision.is_supported():
        raise UnsupportedSGYRevision(
            f"Only SEG-Y revision {SUPPORTED_SGY_REVISION.name} is supported, got {parsed_revision.name}"
        )


class SourceKind(Enum):
    FILE = "file"
    BYTES = "bytes"
    ARRAY = "array"


@dataclass(frozen=True)
class SGYSource:
    kind: SourceKind
    value: Optional[SourceInput] = None


# Backward-compatible alias for the previous stub name.
SourceRef = SGYSource


@dataclass(frozen=True)
class SGYLayout:
    dt_mcs: int
    num_samples: int
    num_traces: int
    data_format: DataFormat = DEFAULT_DATA_FORMAT
    endianness: Endianness = DEFAULT_ENDIANESS

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_format", DataFormat(self.data_format))
        object.__setattr__(self, "endianness", Endianness(self.endianness))

    @property
    def dt(self) -> int:
        return self.dt_mcs

    @property
    def ns(self) -> int:
        return self.num_samples

    @property
    def ntr(self) -> int:
        return self.num_traces

    @property
    def shape(self) -> SizeHW:
        return self.num_samples, self.num_traces

    @property
    def dt_ms(self) -> float:
        return self.dt_mcs * MCS_TO_MS_FACTOR

    @property
    def fs(self) -> float:
        return MS_TO_HZ_FACTOR / self.dt_ms

    @property
    def max_time_ms(self) -> float:
        return self.num_samples * self.dt_ms

    @property
    def endianess(self) -> Endianness:
        return self.endianness

    @property
    def bytes_per_sample(self) -> int:
        return FORMAT_TO_BYTES_PER_SAMPLE[self.data_format]

    @property
    def trace_data_size(self) -> int:
        return self.num_samples * self.bytes_per_sample

    @property
    def trace_block_size(self) -> int:
        return TRACE_HEADER_SIZE + self.trace_data_size
