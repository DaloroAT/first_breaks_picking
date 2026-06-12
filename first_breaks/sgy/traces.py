from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, IO, List, Optional, Sequence, Union

import numpy as np

from first_breaks.sgy.types import (
    FORMAT_TO_DTYPE,
    DataFormat,
    InvalidSamplesSlice,
    NotImplementedReader,
    SGYLayout, SourceInput,
)


IBM_EXPONENT_BIAS = 64
IBM_MANTISSA_BITS = 24
IBM_SIGN_BIT = 31
IBM_EXPONENT_MASK = 0x7F
IBM_MANTISSA_MASK = 0x00FFFFFF
IBM_BASE = 16
IBM_BASE_LOG2 = 4


def decode_blocks(raw: List[bytes] | List[bytearray], layout: SGYLayout) -> np.ndarray:
    if not raw:
        raise ValueError("At least one trace byte block is required")

    block_size = len(raw[0])
    if block_size == 0:
        raise ValueError("Trace byte blocks must not be empty")
    if any(len(block) != block_size for block in raw):
        raise ValueError("Trace byte blocks must have equal size")
    if block_size % layout.bytes_per_sample != 0:
        raise ValueError(
            "Trace byte block size must be divisible by bytes per sample: "
            f"block_size={block_size}, bytes_per_sample={layout.bytes_per_sample}"
        )

    num_samples = block_size // layout.bytes_per_sample
    if num_samples > layout.num_samples:
        raise ValueError(
            f"Trace byte blocks contain {num_samples} samples, but layout contains {layout.num_samples}"
        )

    buffer = b"".join(bytes(block) for block in raw)
    shape = (num_samples, len(raw))
    if layout.data_format == DataFormat.IBM_FLOAT:
        return __decode_ibm_float(buffer, shape, layout)
    if layout.data_format == DataFormat.FIXED_POINT:
        raise NotImplementedReader("Not implemented 32-bit fixed point with gain values reader")

    dtype = __trace_dtype(layout)
    return np.ndarray(shape, dtype=dtype, buffer=buffer, order="F")


def read_traces(
    pointer: IO[bytes],
    trace_ids: Sequence[int],
    layout: SGYLayout,
    min_sample: Optional[int] = None,
    max_sample: Optional[int] = None,
) -> np.ndarray:
    start_sample, stop_sample = __normalize_sample_slice(layout, min_sample, max_sample)
    normalized_trace_ids = __normalize_trace_ids(layout, trace_ids)
    num_samples = stop_sample - start_sample

    if not normalized_trace_ids:
        return np.empty((num_samples, 0), dtype=__output_dtype(layout))

    num_bytes = num_samples * layout.bytes_per_sample
    blocks: List[Union[bytes, bytearray]] = []
    for trace_id in normalized_trace_ids:
        pointer.seek(__trace_data_offset(layout, trace_id, start_sample))
        block = pointer.read(num_bytes)
        if len(block) != num_bytes:
            raise EOFError(f"Cannot read {num_bytes} bytes for trace {trace_id}")
        blocks.append(block)

    return decode_blocks(blocks, layout)


def get_chunked_reader(
    pointer: IO[bytes],
    chunk_size: int,
    layout: SGYLayout,
    min_sample: Optional[int] = None,
    max_sample: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    if chunk_size <= 0:
        raise ValueError("Argument 'chunk_size' must be positive")
    for start in range(0, layout.num_traces, chunk_size):
        stop = min(start + chunk_size, layout.num_traces)
        yield read_traces(
            pointer=pointer,
            trace_ids=list(range(start, stop)),
            min_sample=min_sample,
            max_sample=max_sample,
            layout=layout,
        )


def encode_blocks(traces: np.ndarray, layout: SGYLayout) -> List[bytearray | bytes]:
    normalized_traces = __normalize_traces(traces, layout)
    if layout.data_format == DataFormat.IBM_FLOAT:
        return __encode_ibm_float(normalized_traces, layout)
    if layout.data_format == DataFormat.FIXED_POINT:
        raise NotImplementedReader("Not implemented 32-bit fixed point with gain values writer")

    dtype = __trace_dtype(layout)
    return [normalized_traces[:, trace_id].astype(dtype).tobytes() for trace_id in range(normalized_traces.shape[1])]


def write_traces(
    pointer: IO[bytes],
    trace_ids: Sequence[int],
    traces: np.ndarray,
    layout: SGYLayout,
    start_sample: Optional[int] = None,
) -> None:
    sample_offset = 0 if start_sample is None else start_sample
    if not isinstance(sample_offset, int):
        raise InvalidSamplesSlice("Argument 'start_sample' must be integer")
    if sample_offset < 0 or sample_offset > layout.num_samples:
        raise InvalidSamplesSlice(f"Argument 'start_sample' must be in [0, {layout.num_samples}]")

    normalized_traces = __normalize_traces(traces, layout)
    if sample_offset + normalized_traces.shape[0] > layout.num_samples:
        raise InvalidSamplesSlice(
            "Trace write exceeds layout sample count: "
            f"start_sample={sample_offset}, samples={normalized_traces.shape[0]}, "
            f"num_samples={layout.num_samples}"
        )

    normalized_trace_ids = __normalize_trace_ids(layout, trace_ids)
    if len(normalized_trace_ids) != normalized_traces.shape[1]:
        raise ValueError(
            "Number of trace ids must match number of trace columns: "
            f"trace_ids={len(normalized_trace_ids)}, trace_columns={normalized_traces.shape[1]}"
        )

    for trace_id, block in zip(normalized_trace_ids, encode_blocks(normalized_traces, layout)):
        pointer.seek(__trace_data_offset(layout, trace_id, sample_offset))
        pointer.write(block)


def __normalize_sample_slice(
    layout: SGYLayout,
    min_sample: Optional[int],
    max_sample: Optional[int],
) -> tuple[int, int]:
    start = 0 if min_sample is None else min_sample
    stop = layout.num_samples if max_sample is None else max_sample

    if not isinstance(start, int) or not isinstance(stop, int):
        raise InvalidSamplesSlice("Arguments 'min_sample' and 'max_sample' must be integers")
    if start < 0 or start > layout.num_samples:
        raise InvalidSamplesSlice(f"Argument 'min_sample' must be in [0, {layout.num_samples}]")
    if stop < 0 or stop > layout.num_samples:
        raise InvalidSamplesSlice(f"Argument 'max_sample' must be in [0, {layout.num_samples}]")
    if start >= stop:
        raise InvalidSamplesSlice("Argument 'min_sample' must be less than 'max_sample'")

    return start, stop


def __normalize_trace_ids(layout: SGYLayout, trace_ids: Sequence[int]) -> list[int]:
    normalized_trace_ids = list(trace_ids)
    for trace_id in normalized_trace_ids:
        if not isinstance(trace_id, int):
            raise ValueError("Trace ids must be integers")
        if trace_id < 0 or trace_id >= layout.num_traces:
            raise ValueError(f"Trace id must be in [0, {layout.num_traces}), got {trace_id}")
    return normalized_trace_ids


def __normalize_traces(traces: np.ndarray, layout: SGYLayout) -> np.ndarray:
    normalized_traces = np.asarray(traces)
    if normalized_traces.ndim == 1:
        normalized_traces = normalized_traces.reshape((-1, 1))
    if normalized_traces.ndim != 2:
        raise ValueError("Only 1D and 2D arrays can be used as traces")
    if normalized_traces.shape[0] == 0:
        raise ValueError("Trace arrays must contain at least one sample")
    if normalized_traces.shape[0] > layout.num_samples:
        raise ValueError(
            f"Trace arrays contain {normalized_traces.shape[0]} samples, but layout contains {layout.num_samples}"
        )
    return normalized_traces


def __trace_data_offset(layout: SGYLayout, trace_id: int, sample_id: int = 0) -> int:
    return (
        layout.file_header_size
        + trace_id * layout.trace_block_size
        + layout.trace_header_size
        + sample_id * layout.bytes_per_sample
    )


def __trace_dtype(layout: SGYLayout) -> np.dtype:
    try:
        return np.dtype(f"{layout.endianness.value}{FORMAT_TO_DTYPE[layout.data_format]}")
    except KeyError as exc:
        raise NotImplementedReader(f"Data format {layout.data_format.name} is not supported") from exc


def __output_dtype(layout: SGYLayout) -> np.dtype:
    if layout.data_format == DataFormat.IBM_FLOAT:
        return np.dtype(np.float32)
    return __trace_dtype(layout)


def __decode_ibm_float(buffer: bytes, shape: tuple[int, int], layout: SGYLayout) -> np.ndarray:
    array = np.ndarray(shape, dtype=f"{layout.endianness.value}u4", buffer=buffer, order="F")

    zero_mask = array == 0
    sign = (array >> IBM_SIGN_BIT) & 0x01
    exp = (array >> IBM_MANTISSA_BITS) & IBM_EXPONENT_MASK
    frac = array & IBM_MANTISSA_MASK

    mantissa = frac.astype(np.float64) / float(1 << IBM_MANTISSA_BITS)
    sign_mult = 1.0 - 2.0 * sign.astype(np.float64)
    shift = (exp.astype(np.int32) - IBM_EXPONENT_BIAS) * IBM_BASE_LOG2
    result = np.ldexp(sign_mult * mantissa, shift).astype(np.float32)
    result[zero_mask] = 0.0
    return result


def __encode_ibm_float(traces: np.ndarray, layout: SGYLayout) -> List[Union[bytearray, bytes]]:
    return [__encode_ibm_float_trace(traces[:, trace_id], layout) for trace_id in range(traces.shape[1])]


def __encode_ibm_float_trace(trace: np.ndarray, layout: SGYLayout) -> bytes:
    result = np.zeros(len(trace), dtype=f"{layout.endianness.value}u4")
    non_zero_mask = trace != 0

    if np.any(non_zero_mask):
        values = trace[non_zero_mask].astype(np.float64)
        sign = np.where(values < 0, 1, 0).astype(np.uint32)
        abs_values = np.abs(values)

        _, ieee_exp = np.frexp(abs_values)
        ibm_exp = np.ceil(ieee_exp / float(IBM_BASE_LOG2)).astype(np.int32)
        mantissa = np.ldexp(abs_values, -IBM_BASE_LOG2 * ibm_exp)

        while np.any(mantissa >= 1.0):
            too_large = mantissa >= 1.0
            mantissa[too_large] /= IBM_BASE
            ibm_exp[too_large] += 1

        while np.any((mantissa > 0) & (mantissa < 1.0 / IBM_BASE)):
            too_small = (mantissa > 0) & (mantissa < 1.0 / IBM_BASE)
            mantissa[too_small] *= IBM_BASE
            ibm_exp[too_small] -= 1

        ibm_exp_biased = (ibm_exp + IBM_EXPONENT_BIAS).astype(np.uint32)
        mantissa_int = (mantissa * (1 << IBM_MANTISSA_BITS)).astype(np.uint32) & IBM_MANTISSA_MASK
        result[non_zero_mask] = (sign << IBM_SIGN_BIT) | (ibm_exp_biased << IBM_MANTISSA_BITS) | mantissa_int

    return result.tobytes()


class ITracesStore(ABC):
    @abstractmethod
    @property
    def layout(self) -> SGYLayout:
        raise NotImplementedError

    @abstractmethod
    def write_to_sgy_pointer(self, pointer: IO[bytes]) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self, min_sample: Optional[int] = None, max_sample: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def read_traces_by_ids(
        self,
        ids: Sequence[int],
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_chunked_reader(
        self,
        chunk_size: int,
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        raise NotImplementedError


class TracesStoreArray(ITracesStore):
    def __init__(self, array: np.ndarray, layout: SGYLayout):
        self.__array = array
        self.__layout = layout
        # validate against layout


class TracesStoreBytes(ITracesStore):
    def __init__(self, raw: bytes, layout: SGYLayout):
        self.__raw = raw
        self.__layout = layout
        self.__cached_array: np.ndarray | None = None  # in case all traces were read


class TracesStoreFile(ITracesStore):
    def __init__(self, path: Path | str, layout: SGYLayout):
        self.__path = path
        self.__layout = layout
        self.__cached_array: np.ndarray | None = None  # in case all traces were read


def get_traces(source: SourceInput, layout: SGYLayout) -> ITracesStore:
    if isinstance(source, np.ndarray):
        return TracesStoreArray(source, layout)
    elif isinstance(source, (str, Path)):
        return TracesStoreFile(source, layout)
    elif isinstance(source, bytes):
        return TracesStoreBytes(source, layout)
    else:
        raise TypeError("Unsupported source type")
