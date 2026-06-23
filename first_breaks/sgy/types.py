from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

SizeHW = Tuple[int, int]
SourceInput = Union[str, Path, bytes, np.ndarray]

REV0_FILE_HEADER_SIZE = 3600
REV0_TRACE_HEADER_SIZE = 240
REV0_DT_OFFSET = 3216
REV0_NS_OFFSET = 3220
REV0_DATA_FORMAT_OFFSET = 3224
REV0_REVISION_OFFSET = 3500
REV0_FIXED_LENGTH_TRACE_FLAG_OFFSET = 3502
REV0_NUMBER_OF_TEXTUAL_HEADERS_OFFSET = 3504
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


DEFAULT_ENDIANNESS = Endianness.BIG
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
        endianness: Union[Endianness, str],
    ) -> "SGYRevision":
        if len(raw) != 2:
            raise InvalidSGY(f"SEG-Y revision header requires 2 bytes, got {len(raw)}")
        byte_order = "big" if Endianness(endianness) == Endianness.BIG else "little"
        return cls.from_header_value(int.from_bytes(raw, byteorder=byte_order, signed=False))


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


class SourceKind(Enum):
    FILE = "file"
    BYTES = "bytes"
    ARRAY = "array"


@dataclass(frozen=True)
class SGYSource:
    kind: SourceKind
    value: Optional[SourceInput] = None


@dataclass(frozen=True)
class SGYLayout:
    dt_mcs: int
    num_samples: int
    num_traces: int
    data_format: DataFormat
    endianness: Endianness
    revision: SGYRevision
    file_header_size: int
    trace_header_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_format", DataFormat(self.data_format))
        object.__setattr__(self, "endianness", Endianness(self.endianness))
        revision = SGYRevision.from_header_value(self.revision)
        object.__setattr__(self, "revision", revision)
        if self.dt_mcs <= 0:
            raise InvalidSGY(f"Sample interval must be positive, got {self.dt_mcs}")
        if self.num_samples <= 0:
            raise InvalidSGY(f"Number of samples must be positive, got {self.num_samples}")
        if self.num_traces < 0:
            raise InvalidSGY(f"Number of traces must be non-negative, got {self.num_traces}")

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
    def bytes_per_sample(self) -> int:
        return FORMAT_TO_BYTES_PER_SAMPLE[self.data_format]

    @property
    def trace_data_size(self) -> int:
        return self.num_samples * self.bytes_per_sample

    @property
    def trace_block_size(self) -> int:
        return self.trace_header_size + self.trace_data_size

    @classmethod
    def from_array(
        cls,
        traces: np.ndarray,
        *,
        dt_mcs: Union[int, float],
        data_format: Union[DataFormat, int],
        endianness: Union[Endianness, str],
    ) -> "SGYLayout":
        if traces.ndim not in (1, 2):
            raise SGYInitParamsError("Only 1D and 2D arrays can be used as SGY traces")
        return cls(
            dt_mcs=int(dt_mcs),
            num_samples=int(traces.shape[0]),
            num_traces=1 if traces.ndim == 1 else int(traces.shape[1]),
            data_format=DataFormat(data_format),
            endianness=Endianness(endianness),
            revision=SUPPORTED_SGY_REVISION,
            file_header_size=REV0_FILE_HEADER_SIZE,
            trace_header_size=REV0_TRACE_HEADER_SIZE,
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> "SGYLayout":
        return cls.from_header_bytes(payload[:REV0_FILE_HEADER_SIZE], file_size=len(payload))

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SGYLayout":
        source_path = Path(path)
        with source_path.open("rb") as descriptor:
            header = descriptor.read(REV0_FILE_HEADER_SIZE)
        return cls.from_header_bytes(header, file_size=source_path.stat().st_size)

    @classmethod
    def from_header_bytes(cls, header: bytes, *, file_size: int) -> "SGYLayout":
        if len(header) < REV0_FILE_HEADER_SIZE:
            raise InvalidSGY(
                f"SEG-Y source is too small to contain file headers: expected at least {REV0_FILE_HEADER_SIZE} bytes"
            )
        endianness = _detect_endianness(header)
        revision = SGYRevision.from_bytes(
            header[REV0_REVISION_OFFSET : REV0_REVISION_OFFSET + 2],
            endianness=endianness,
        )
        _ensure_fixed_size_revision(header, revision, endianness)
        data_format = DataFormat(_unpack_unsigned_short(header, REV0_DATA_FORMAT_OFFSET, endianness))
        if not DataFormat.is_supported_for_reading(data_format):
            raise NotImplementedReader(f"Data format {data_format.name} is not supported for reading")
        dt_mcs = _unpack_unsigned_short(header, REV0_DT_OFFSET, endianness)
        num_samples = _unpack_unsigned_short(header, REV0_NS_OFFSET, endianness)
        bytes_per_sample = FORMAT_TO_BYTES_PER_SAMPLE[data_format]
        trace_data_size = num_samples * bytes_per_sample
        trace_block_size = REV0_TRACE_HEADER_SIZE + trace_data_size
        traces_payload_size = file_size - REV0_FILE_HEADER_SIZE
        if traces_payload_size < 0:
            raise InvalidSGY(
                f"SEG-Y source is too small: expected at least {REV0_FILE_HEADER_SIZE} bytes, got {file_size}"
            )
        if trace_block_size <= 0 or traces_payload_size % trace_block_size != 0:
            raise InvalidSGY(
                "SEG-Y file size is inconsistent with binary header layout: "
                f"file_size={file_size}, trace_block_size={trace_block_size}"
            )
        return cls(
            dt_mcs=dt_mcs,
            num_samples=num_samples,
            num_traces=traces_payload_size // trace_block_size,
            data_format=data_format,
            endianness=endianness,
            revision=revision,
            file_header_size=REV0_FILE_HEADER_SIZE,
            trace_header_size=REV0_TRACE_HEADER_SIZE,
        )


def _unpack_unsigned_short(header: bytes, offset: int, endianness: Endianness) -> int:
    return int.from_bytes(
        header[offset : offset + 2],
        byteorder="big" if endianness == Endianness.BIG else "little",
        signed=False,
    )


def _ensure_fixed_size_revision(header: bytes, revision: SGYRevision, endianness: Endianness) -> None:
    if revision == SGYRevision.REV_0:
        return

    fixed_length_trace_flag = _unpack_unsigned_short(header, REV0_FIXED_LENGTH_TRACE_FLAG_OFFSET, endianness)
    number_of_textual_headers = _unpack_unsigned_short(header, REV0_NUMBER_OF_TEXTUAL_HEADERS_OFFSET, endianness)
    if number_of_textual_headers != 0:
        raise UnsupportedSGYRevision(
            "SEG-Y files with extended textual headers are not supported: "
            f"revision={revision.name}, number_of_textual_headers={number_of_textual_headers}"
        )
    if fixed_length_trace_flag == 0:
        raise UnsupportedSGYRevision(
            f"SEG-Y files with variable-length traces are not supported: revision={revision.name}"
        )


def _detect_endianness(header: bytes) -> Endianness:
    big_value = _unpack_unsigned_short(header, REV0_DATA_FORMAT_OFFSET, Endianness.BIG)
    little_value = _unpack_unsigned_short(header, REV0_DATA_FORMAT_OFFSET, Endianness.LITTLE)
    big_valid = DataFormat.is_valid(big_value)
    little_valid = DataFormat.is_valid(little_value)
    if big_valid and not little_valid:
        return Endianness.BIG
    if little_valid and not big_valid:
        return Endianness.LITTLE
    raise InvalidSGY(
        "Cannot determine SEG-Y endianness from binary header data sample format: "
        f"big={big_value}, little={little_value}"
    )
