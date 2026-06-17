from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import ContextManager, Generator, IO, List, Optional, Sequence, Union

import numpy as np

from first_breaks.sgy.types import (
    FORMAT_TO_DTYPE,
    DataFormat,
    InvalidSamplesSlice,
    NotImplementedReader,
    SGYLayout,
    SourceInput,
)


IBM_EXPONENT_BIAS = 64
IBM_MANTISSA_BITS = 24
IBM_SIGN_BIT = 31
IBM_EXPONENT_MASK = 0x7F
IBM_MANTISSA_MASK = 0x00FFFFFF
IBM_BASE = 16
IBM_BASE_LOG2 = 4


def validate_raw_blocks(raw: List[bytes] | List[bytearray], layout: SGYLayout) -> None:
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


def decode_blocks(raw: List[bytes] | List[bytearray], layout: SGYLayout) -> np.ndarray:
    validate_raw_blocks(raw=raw, layout=layout)
    block_size = len(raw[0])
    num_samples = block_size // layout.bytes_per_sample
    buffer = b"".join(bytes(block) for block in raw)
    shape = (num_samples, len(raw))
    if layout.data_format == DataFormat.IBM_FLOAT:
        return _decode_ibm_float(buffer, shape, layout)
    if layout.data_format == DataFormat.FIXED_POINT:
        raise NotImplementedReader("Not implemented 32-bit fixed point with gain values reader")

    dtype = _trace_dtype(layout)
    return np.ndarray(shape, dtype=dtype, buffer=buffer, order="F")


def read_traces(
    pointer: IO[bytes],
    trace_ids: Sequence[int],
    layout: SGYLayout,
    min_sample: Optional[int] = None,
    max_sample: Optional[int] = None,
) -> np.ndarray:
    start_sample, stop_sample = _normalize_sample_slice(layout, min_sample, max_sample)
    normalized_trace_ids = _normalize_trace_ids(layout, trace_ids)
    num_samples = stop_sample - start_sample

    if not normalized_trace_ids:
        return np.empty((num_samples, 0), dtype=_output_dtype(layout))

    num_bytes = num_samples * layout.bytes_per_sample
    blocks: List[Union[bytes, bytearray]] = []
    for trace_id in normalized_trace_ids:
        pointer.seek(_trace_data_offset(layout, trace_id, start_sample))
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
    normalized_traces = _normalize_traces(traces, layout)
    if layout.data_format == DataFormat.IBM_FLOAT:
        return _encode_ibm_float(normalized_traces, layout)
    if layout.data_format == DataFormat.FIXED_POINT:
        raise NotImplementedReader("Not implemented 32-bit fixed point with gain values writer")

    dtype = _trace_dtype(layout)
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

    normalized_traces = _normalize_traces(traces, layout)
    if sample_offset + normalized_traces.shape[0] > layout.num_samples:
        raise InvalidSamplesSlice(
            "Trace write exceeds layout sample count: "
            f"start_sample={sample_offset}, samples={normalized_traces.shape[0]}, "
            f"num_samples={layout.num_samples}"
        )

    normalized_trace_ids = _normalize_trace_ids(layout, trace_ids)
    if len(normalized_trace_ids) != normalized_traces.shape[1]:
        raise ValueError(
            "Number of trace ids must match number of trace columns: "
            f"trace_ids={len(normalized_trace_ids)}, trace_columns={normalized_traces.shape[1]}"
        )

    for trace_id, block in zip(normalized_trace_ids, encode_blocks(normalized_traces, layout)):
        pointer.seek(_trace_data_offset(layout, trace_id, sample_offset))
        pointer.write(block)


def _normalize_sample_slice(
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


def _normalize_trace_ids(layout: SGYLayout, trace_ids: Sequence[int]) -> list[int]:
    normalized_trace_ids = list(trace_ids)
    for trace_id in normalized_trace_ids:
        if not isinstance(trace_id, int):
            raise ValueError("Trace ids must be integers")
        if trace_id < 0 or trace_id >= layout.num_traces:
            raise ValueError(f"Trace id must be in [0, {layout.num_traces}), got {trace_id}")
    return normalized_trace_ids


def _normalize_traces(traces: np.ndarray, layout: SGYLayout) -> np.ndarray:
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


def _trace_data_offset(layout: SGYLayout, trace_id: int, sample_id: int = 0) -> int:
    return (
        layout.file_header_size
        + trace_id * layout.trace_block_size
        + layout.trace_header_size
        + sample_id * layout.bytes_per_sample
    )


def _trace_dtype(layout: SGYLayout) -> np.dtype:
    try:
        return np.dtype(f"{layout.endianness.value}{FORMAT_TO_DTYPE[layout.data_format]}")
    except KeyError as exc:
        raise NotImplementedReader(f"Data format {layout.data_format.name} is not supported") from exc


def _output_dtype(layout: SGYLayout) -> np.dtype:
    if layout.data_format == DataFormat.IBM_FLOAT:
        return np.dtype(np.float32)
    return _trace_dtype(layout)


def _decode_ibm_float(buffer: bytes, shape: tuple[int, int], layout: SGYLayout) -> np.ndarray:
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


def _encode_ibm_float(traces: np.ndarray, layout: SGYLayout) -> List[Union[bytearray, bytes]]:
    return [_encode_ibm_float_trace(traces[:, trace_id], layout) for trace_id in range(traces.shape[1])]


def _encode_ibm_float_trace(trace: np.ndarray, layout: SGYLayout) -> bytes:
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


def is_all_data(
    ids: Sequence[int],
    layout: SGYLayout,
    min_sample: Optional[int],
    max_sample: Optional[int],
) -> bool:
    normalized_ids = list(ids)
    is_all_traces = normalized_ids == list(range(layout.num_traces))
    is_from_start = min_sample is None or min_sample == 0
    is_until_end = max_sample is None or max_sample == layout.num_samples
    return is_all_traces and is_from_start and is_until_end


def can_copy_raw_trace_data(source_layout: SGYLayout, output_layout: SGYLayout) -> bool:
    return (
        source_layout.shape == output_layout.shape
        and source_layout.data_format == output_layout.data_format
        and source_layout.endianness == output_layout.endianness
    )


def copy_raw_trace_data(
    read_pointer: IO[bytes],
    write_pointer: IO[bytes],
    source_layout: SGYLayout,
    output_layout: SGYLayout,
) -> None:
    if not can_copy_raw_trace_data(source_layout, output_layout):
        raise ValueError("Raw trace data can only be copied between matching trace-data layouts")

    for trace_id in range(source_layout.num_traces):
        read_pointer.seek(_trace_data_offset(source_layout, trace_id))
        block = read_pointer.read(source_layout.trace_data_size)
        if len(block) != source_layout.trace_data_size:
            raise EOFError(f"Cannot read {source_layout.trace_data_size} bytes for trace {trace_id}")
        write_pointer.seek(_trace_data_offset(output_layout, trace_id))
        write_pointer.write(block)


class TracesBackend(ABC):
    @property
    @abstractmethod
    def layout(self) -> SGYLayout:
        raise NotImplementedError

    @abstractmethod
    def write_to_sgy_pointer(self, pointer: IO[bytes], output_layout: Optional[SGYLayout]) -> None:
        raise NotImplementedError

    def read(self, min_sample: Optional[int] = None, max_sample: Optional[int] = None) -> np.ndarray:
        return self.read_traces_by_ids(
            ids=range(self.layout.num_traces),
            min_sample=min_sample,
            max_sample=max_sample,
        )

    @abstractmethod
    def read_traces_by_ids(
        self,
        ids: Sequence[int],
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def get_chunked_reader(
        self,
        chunk_size: int,
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        if chunk_size <= 0:
            raise ValueError("Argument 'chunk_size' must be positive")
        for start in range(0, self.layout.num_traces, chunk_size):
            stop = min(start + chunk_size, self.layout.num_traces)
            yield self.read_traces_by_ids(
                ids=range(start, stop),
                min_sample=min_sample,
                max_sample=max_sample,
            )


class TracesBackendArray(TracesBackend):
    def __init__(self, array: np.ndarray, layout: SGYLayout) -> None:
        self.__array = _normalize_traces(array, layout)
        if self.__array.shape != layout.shape:
            raise ValueError(f"Trace array shape must be {layout.shape}, got {self.__array.shape}")
        self.__layout = layout

    @classmethod
    def empty(cls, layout: SGYLayout) -> "TracesBackendArray":
        return TracesBackendArray(array=np.zeros(layout.shape), layout=layout)

    @property
    def layout(self) -> SGYLayout:
        return self.__layout

    def write_to_sgy_pointer(self, pointer: IO[bytes], output_layout: Optional[SGYLayout]) -> None:
        if output_layout is not None:
            if output_layout.shape != self.layout.shape:
                raise ValueError(f"Trace layout shape must be {self.layout.shape}, got {output_layout.shape}")
            layout = output_layout
        else:
            layout = self.__layout
        write_traces(
            pointer=pointer,
            trace_ids=range(layout.num_traces),
            traces=self.__array,
            layout=layout,
        )

    def read_traces_by_ids(
        self,
        ids: Sequence[int],
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> np.ndarray:
        start_sample, stop_sample = _normalize_sample_slice(self.layout, min_sample, max_sample)
        trace_ids = _normalize_trace_ids(self.layout, ids)
        return self.__array[start_sample:stop_sample, trace_ids].copy()


class TracesBackendRawSource(TracesBackend, ABC):
    def __init__(self, layout: SGYLayout) -> None:
        self.__layout = layout
        self.__cache: TracesBackendArray | None = None

    @property
    def layout(self) -> SGYLayout:
        return self.__layout

    @abstractmethod
    def _open_pointer(self) -> ContextManager[IO[bytes]]:
        raise NotImplementedError

    def __create_cache_if_needed_and_possible(
        self,
        ids: Sequence[int],
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> None:
        if self.__cache is None:
            if is_all_data(layout=self.__layout, ids=ids, min_sample=min_sample, max_sample=max_sample):
                full_array = self.__read_from_source(
                    ids=range(self.layout.num_traces),
                    min_sample=None,
                    max_sample=None,
                )
                self.__cache = TracesBackendArray(array=full_array, layout=self.layout)

    def read_traces_by_ids(
        self,
        ids: Sequence[int],
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> np.ndarray:
        self.__create_cache_if_needed_and_possible(ids=ids, min_sample=min_sample, max_sample=max_sample)
        if self.__cache:
            return self.__cache.read_traces_by_ids(ids=ids, min_sample=min_sample, max_sample=max_sample)
        return self.__read_from_source(ids=ids, min_sample=min_sample, max_sample=max_sample)

    def write_to_sgy_pointer(self, pointer: IO[bytes], output_layout: Optional[SGYLayout]) -> None:
        layout = self.__layout if output_layout is None else output_layout
        if can_copy_raw_trace_data(self.__layout, layout):
            with self._open_pointer() as read_pointer:
                copy_raw_trace_data(
                    read_pointer=read_pointer,
                    write_pointer=pointer,
                    source_layout=self.__layout,
                    output_layout=layout,
                )
            return
        if layout.shape != self.layout.shape:
            raise ValueError(f"Trace layout shape must be {self.layout.shape}, got {layout.shape}")

        for start in range(0, self.layout.num_traces, 1024):
            stop = min(start + 1024, self.layout.num_traces)
            traces = self.read_traces_by_ids(range(start, stop))
            write_traces(pointer=pointer, trace_ids=range(start, stop), traces=traces, layout=layout)

    def __read_from_source(
        self,
        ids: Sequence[int],
        min_sample: Optional[int],
        max_sample: Optional[int],
    ) -> np.ndarray:
        with self._open_pointer() as pointer:
            return read_traces(pointer, trace_ids=ids, layout=self.layout, min_sample=min_sample, max_sample=max_sample)


class TracesBackendBytes(TracesBackendRawSource):
    def __init__(self, raw: bytes | bytearray, layout: SGYLayout) -> None:
        super().__init__(layout)
        self.__raw = raw

    def _open_pointer(self) -> ContextManager[IO[bytes]]:
        return BytesIO(self.__raw)


class TracesBackendFile(TracesBackendRawSource):
    def __init__(self, path: Path | str, layout: SGYLayout) -> None:
        super().__init__(layout)
        self.__path = Path(path)

    def _open_pointer(self) -> ContextManager[IO[bytes]]:
        return self.__path.open("rb")


def get_traces_backend(source: SourceInput, layout: SGYLayout) -> TracesBackend:
    if isinstance(source, np.ndarray):
        return TracesBackendArray(source, layout)
    elif isinstance(source, (str, Path)):
        return TracesBackendFile(source, layout)
    elif isinstance(source, bytes):
        return TracesBackendBytes(source, layout)
    else:
        raise TypeError("Unsupported source type")
