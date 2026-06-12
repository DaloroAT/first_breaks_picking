from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from first_breaks.sgy.headers import FileHeaders, TraceHeaderField, TraceHeaders
from first_breaks.sgy.traces import get_chunked_reader, read_traces, write_traces
from first_breaks.sgy.types import (
    DEFAULT_DATA_FORMAT,
    DEFAULT_ENDIANESS,
    FORMAT_TO_BYTES_PER_SAMPLE,
    DataFormat,
    Endianness,
    InvalidSGY,
    InvalidSamplesSlice,
    NotImplementedReader,
    SGYInitParamsError,
    SGYLayout,
    SGYRevision,
    SGYSource,
    SizeHW,
    SourceInput,
    SourceKind,
)
from first_breaks.utils.utils import UnitsConverter, calc_hash


class SGY:
    fmt2bps = FORMAT_TO_BYTES_PER_SAMPLE

    def __init__(
        self,
        source: SourceInput,
        dt_mcs: Optional[Union[int, float]] = None,
        file_headers: Optional[Dict[str, Any]] = None,
        traces_headers: Optional[pd.DataFrame] = None,
    ) -> None:
        self.__source: SourceInput
        self.__source_kind: SourceKind
        self.__array: Optional[np.ndarray] = None
        self.__layout: SGYLayout
        self.__file_headers: FileHeaders
        self.__trace_headers: TraceHeaders

        self.__build_components(
            source=source,
            dt_mcs=dt_mcs,
            file_headers=file_headers,
            traces_headers=traces_headers,
        )
        self.__units_converter = UnitsConverter(sgy_mcs=self.__layout.dt_mcs)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
    ) -> "SGY":
        return cls(path)

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
    ) -> "SGY":
        return cls(payload)

    @classmethod
    def from_array(
        cls,
        traces: np.ndarray,
        *,
        dt_mcs: Union[int, float],
        file_headers: Optional[Dict[str, Any]] = None,
        traces_headers: Optional[pd.DataFrame] = None,
    ) -> "SGY":
        return cls(
            traces,
            dt_mcs=dt_mcs,
            file_headers=file_headers,
            traces_headers=traces_headers,
        )

    @property
    def source(self) -> SourceInput:
        return self.__source

    @property
    def source_ref(self) -> SGYSource:
        return SGYSource(kind=self.__source_kind, value=self.__source)

    @property
    def layout(self) -> SGYLayout:
        return self.__layout

    @property
    def revision(self) -> SGYRevision:
        return self.__layout.revision

    @property
    def file_headers(self) -> FileHeaders:
        return self.__file_headers

    @property
    def trace_headers(self) -> TraceHeaders:
        return self.__trace_headers

    @property
    def dt(self) -> int:
        return self.__layout.dt

    @property
    def dt_mcs(self) -> int:
        return self.__layout.dt_mcs

    @property
    def dt_ms(self) -> float:
        return self.__layout.dt_ms

    @property
    def fs(self) -> float:
        return self.__layout.fs

    @property
    def ns(self) -> int:
        return self.__layout.ns

    @property
    def ntr(self) -> int:
        return self.__layout.ntr

    @property
    def num_samples(self) -> int:
        return self.__layout.num_samples

    @property
    def num_traces(self) -> int:
        return self.__layout.num_traces

    @property
    def shape(self) -> SizeHW:
        return self.__layout.shape

    @property
    def max_time_ms(self) -> float:
        return self.__layout.max_time_ms

    @property
    def endianess(self) -> str:
        return self.__layout.endianness.value

    @property
    def endianness(self) -> Endianness:
        return self.__layout.endianness

    @property
    def data_format(self) -> int:
        return int(self.__layout.data_format)

    @property
    def sample_format(self) -> DataFormat:
        return self.__layout.data_format

    @property
    def is_source_ndarray(self) -> bool:
        return self.__source_kind == SourceKind.ARRAY

    @property
    def general_headers(self) -> Dict[str, Any]:
        return self.__file_headers.headers()

    @property
    def traces_headers(self) -> pd.DataFrame:
        return self.__trace_headers.scaled()

    @property
    def traces_headers_raw(self) -> pd.DataFrame:
        return self.__trace_headers.raw()

    @property
    def traces_headers_schema(self) -> TraceHeaders:
        return self.__trace_headers

    def ms2index(self, ms_value: float) -> int:
        return self.__units_converter.ms2index(ms_value)  # type: ignore[return-value]

    def get_hash(self) -> Optional[str]:
        if self.__source_kind == SourceKind.ARRAY:
            return None
        return calc_hash(self.__source)  # type: ignore[arg-type]

    def read(self, min_sample: Optional[int] = None, max_sample: Optional[int] = None) -> np.ndarray:
        return self.read_traces_by_ids(
            ids=range(self.num_traces),
            min_sample=min_sample,
            max_sample=max_sample,
        )

    def read_traces_by_ids(
        self,
        ids: Sequence[int],
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> np.ndarray:
        if self.__source_kind == SourceKind.ARRAY:
            return self.__read_array_traces(ids=ids, min_sample=min_sample, max_sample=max_sample)
        if self.__source_kind == SourceKind.BYTES:
            pointer = BytesIO(self.__source)  # type: ignore[arg-type]
            return read_traces(pointer, trace_ids=ids, layout=self.__layout, min_sample=min_sample, max_sample=max_sample)
        with Path(self.__source).open("rb") as pointer:  # type: ignore[arg-type]
            return read_traces(pointer, trace_ids=ids, layout=self.__layout, min_sample=min_sample, max_sample=max_sample)

    def get_chunked_reader(
        self,
        chunk_size: int,
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        if chunk_size <= 0:
            raise ValueError("Argument 'chunk_size' must be positive")
        if self.__source_kind == SourceKind.ARRAY:
            for start in range(0, self.num_traces, chunk_size):
                stop = min(start + chunk_size, self.num_traces)
                yield self.__read_array_traces(
                    ids=range(start, stop),
                    min_sample=min_sample,
                    max_sample=max_sample,
                )
            return
        if self.__source_kind == SourceKind.BYTES:
            pointer = BytesIO(self.__source)  # type: ignore[arg-type]
            yield from get_chunked_reader(pointer, chunk_size, self.__layout, min_sample=min_sample, max_sample=max_sample)
            return
        with Path(self.__source).open("rb") as pointer:  # type: ignore[arg-type]
            yield from get_chunked_reader(pointer, chunk_size, self.__layout, min_sample=min_sample, max_sample=max_sample)

    def read_custom_trace_header(self, byte_position: int, encoding: str) -> tuple[Any, ...]:
        for field in TraceHeaderField:
            if field.value.offset == byte_position and field.value.format == encoding:
                return tuple(self.__trace_headers[field].tolist())
        raise NotImplementedError("Reading arbitrary trace header byte positions is not implemented")

    def write(
        self,
        output_path: Union[str, Path],
        data_format: Optional[Union[DataFormat, int]] = None,
        endianess: Optional[Union[Endianness, str]] = None,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_layout = self.__make_write_layout(data_format=data_format, endianess=endianess)
        file_headers = self.__file_headers_for_layout(write_layout)
        trace_headers = TraceHeaders.from_values(self.__trace_headers.raw(), write_layout)

        with output_path.open("wb+") as pointer:
            file_headers.write_to_sgy_pointer(pointer)
            trace_headers.write_to_sgy_pointer(pointer)
            for start in range(0, self.num_traces, 1024):
                stop = min(start + 1024, self.num_traces)
                traces = self.read_traces_by_ids(range(start, stop))
                write_traces(
                    pointer=pointer,
                    trace_ids=range(start, stop),
                    traces=traces,
                    layout=write_layout,
                )

    def export_sgy_with_picks(
        self,
        output_fname: Union[str, Path],
        picks_in_mcs: List[float],
        byte_position: int = 236,
        encoding: Optional[str] = None,
        picks_unit: Optional[str] = "mcs",
    ) -> None:
        field = TraceHeaderField.FB_PICK
        if byte_position != field.value.offset:
            raise NotImplementedError("Only FB_PICK trace header export is implemented")
        if encoding is not None and encoding != field.value.format:
            raise NotImplementedError("Only FB_PICK trace header export with its standard encoding is implemented")
        if len(picks_in_mcs) != self.num_traces:
            raise ValueError(f"Number of picks ({len(picks_in_mcs)}) must match number of traces ({self.num_traces})")
        if picks_unit not in ("ms", "mcs", "sample"):
            raise ValueError("Argument 'picks_unit' must be one of 'ms', 'mcs', or 'sample'")

        cast_to = int
        if picks_unit == "ms":
            picks = self.__units_converter.mcs2ms(picks_in_mcs, cast_to=cast_to)
        elif picks_unit == "sample":
            picks = self.__units_converter.mcs2index(picks_in_mcs, cast_to=cast_to)
        else:
            picks = np.asarray(picks_in_mcs).astype(cast_to)

        trace_headers = self.__trace_headers.raw()
        trace_headers[field.name] = picks

        exported = SGY(
            self.read(),
            dt_mcs=self.dt_mcs,
            file_headers=self.general_headers,
            traces_headers=trace_headers,
        )
        exported.write(output_fname, data_format=self.sample_format, endianess=self.endianness)

    def __build_components(
        self,
        *,
        source: SourceInput,
        dt_mcs: Optional[Union[int, float]],
        file_headers: Optional[Dict[str, Any]],
        traces_headers: Optional[pd.DataFrame],
    ) -> None:
        if isinstance(source, np.ndarray):
            self.__build_array_components(source, dt_mcs, file_headers, traces_headers)
            return

        if dt_mcs is not None:
            raise SGYInitParamsError("Argument 'dt_mcs' must be empty if SGY is created from external source")
        if file_headers is not None:
            raise SGYInitParamsError("Argument 'file_headers' must be empty if SGY is created from external source")
        if traces_headers is not None:
            raise SGYInitParamsError("Argument 'traces_headers' must be empty if SGY is created from external source")

        if isinstance(source, bytes):
            self.__build_bytes_components(source)
            return
        if isinstance(source, (str, Path)):
            self.__build_file_components(Path(source))
            return
        raise SGYInitParamsError("Only `str`, `Path`, `bytes`, and `np.ndarray` sources are supported")

    def __build_array_components(
        self,
        source: np.ndarray,
        dt_mcs: Optional[Union[int, float]],
        file_headers: Optional[Dict[str, Any]],
        traces_headers: Optional[pd.DataFrame],
    ) -> None:
        if dt_mcs is None:
            raise SGYInitParamsError("Argument 'dt_mcs' is required if np.ndarray is used as input")
        layout = SGYLayout.from_array(
            source,
            dt_mcs=dt_mcs,
            data_format=DEFAULT_DATA_FORMAT,
            endianness=DEFAULT_ENDIANESS,
        )
        self.__source = source
        self.__source_kind = SourceKind.ARRAY
        self.__array = self.__normalize_array(source)
        self.__layout = layout
        self.__file_headers = FileHeaders.from_layout(layout, overrides=file_headers)
        self.__trace_headers = (
            TraceHeaders.from_values(traces_headers, layout) if traces_headers is not None else TraceHeaders.empty(layout)
        )

    def __build_bytes_components(self, source: bytes) -> None:
        layout = SGYLayout.from_bytes(source)
        pointer = BytesIO(source)
        self.__source = source
        self.__source_kind = SourceKind.BYTES
        self.__array = None
        self.__layout = layout
        self.__file_headers = FileHeaders.from_sgy_pointer(pointer, layout)
        self.__trace_headers = TraceHeaders.from_sgy_pointer(pointer, layout)

    def __build_file_components(self, source: Path) -> None:
        layout = SGYLayout.from_file(source)
        with source.open("rb") as pointer:
            file_headers = FileHeaders.from_sgy_pointer(pointer, layout)
            trace_headers = TraceHeaders.from_sgy_pointer(pointer, layout)
        self.__source = source
        self.__source_kind = SourceKind.FILE
        self.__array = None
        self.__layout = layout
        self.__file_headers = file_headers
        self.__trace_headers = trace_headers

    def __normalize_array(self, source: np.ndarray) -> np.ndarray:
        traces = np.asarray(source)
        if traces.ndim == 1:
            traces = traces.reshape((-1, 1))
        return traces

    def __read_array_traces(
        self,
        *,
        ids: Sequence[int],
        min_sample: Optional[int],
        max_sample: Optional[int],
    ) -> np.ndarray:
        if self.__array is None:
            raise RuntimeError("Array source is not available")
        start, stop = self.__normalize_sample_slice(min_sample, max_sample)
        trace_ids = self.__normalize_trace_ids(ids)
        return self.__array[start:stop, trace_ids].copy()

    def __normalize_sample_slice(
        self,
        min_sample: Optional[int],
        max_sample: Optional[int],
    ) -> tuple[int, int]:
        start = 0 if min_sample is None else min_sample
        stop = self.num_samples if max_sample is None else max_sample
        if not isinstance(start, int) or not isinstance(stop, int):
            raise InvalidSamplesSlice("Arguments 'min_sample' and 'max_sample' must be integers")
        if start < 0 or start > self.num_samples:
            raise InvalidSamplesSlice(f"Argument 'min_sample' must be in [0, {self.num_samples}]")
        if stop < 0 or stop > self.num_samples:
            raise InvalidSamplesSlice(f"Argument 'max_sample' must be in [0, {self.num_samples}]")
        if start >= stop:
            raise InvalidSamplesSlice("Argument 'min_sample' must be less than 'max_sample'")
        return start, stop

    def __normalize_trace_ids(self, ids: Sequence[int]) -> list[int]:
        trace_ids = list(ids)
        for trace_id in trace_ids:
            if not isinstance(trace_id, int):
                raise ValueError("Trace ids must be integers")
            if trace_id < 0 or trace_id >= self.num_traces:
                raise ValueError(f"Trace id must be in [0, {self.num_traces}), got {trace_id}")
        return trace_ids

    def __make_write_layout(
        self,
        *,
        data_format: Optional[Union[DataFormat, int]],
        endianess: Optional[Union[Endianness, str]],
    ) -> SGYLayout:
        return SGYLayout(
            dt_mcs=self.__layout.dt_mcs,
            num_samples=self.__layout.num_samples,
            num_traces=self.__layout.num_traces,
            data_format=self.__layout.data_format if data_format is None else DataFormat(data_format),
            endianness=self.__layout.endianness if endianess is None else Endianness(endianess),
            revision=self.__layout.revision,
            file_header_size=self.__layout.file_header_size,
            trace_header_size=self.__layout.trace_header_size,
        )

    def __file_headers_for_layout(self, layout: SGYLayout) -> FileHeaders:
        values = self.__file_headers.headers()
        values[FileHeaders.dt_name] = layout.dt_mcs
        values[FileHeaders.dt_name_orig] = layout.dt_mcs
        values[FileHeaders.ns_name] = layout.num_samples
        values[FileHeaders.ns_name_orig] = layout.num_samples
        values[FileHeaders.data_sample_format_name] = int(layout.data_format)
        return FileHeaders.from_values(values, layout)


__all__ = [
    "DataFormat",
    "Endianness",
    "InvalidSGY",
    "InvalidSamplesSlice",
    "NotImplementedReader",
    "SGY",
    "SGYRevision",
    "SGYInitParamsError",
]
