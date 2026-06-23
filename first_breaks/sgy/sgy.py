from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from first_breaks.sgy.headers import (
    FileHeaderField,
    FileHeaders,
    HeaderInfo,
    TraceHeaderField,
    TraceHeaders,
    read_custom_traces_header,
    write_custom_traces_header,
)
from first_breaks.sgy.traces import TracesBackend, get_traces_backend
from first_breaks.sgy.types import (
    DEFAULT_DATA_FORMAT,
    DEFAULT_ENDIANNESS,
    FORMAT_TO_BYTES_PER_SAMPLE,
    DataFormat,
    Endianness,
    InvalidSGY,
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
    def __init__(
        self,
        source: SourceInput,
        dt_mcs: Optional[Union[int, float]] = None,
        file_headers: Optional[Dict[str, Any]] = None,
        traces_headers: Optional[pd.DataFrame] = None,
    ) -> None:
        self.__source: SourceInput
        self.__source_kind: SourceKind
        self.__layout: SGYLayout
        self.__file_headers: FileHeaders
        self.__trace_headers: TraceHeaders
        self.__traces: TracesBackend

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
    def endianness(self) -> Endianness:
        return self.__layout.endianness

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

    def ms2index(self, ms_value: float) -> int:
        return self.__units_converter.ms2index(ms_value)  # type: ignore[return-value]

    def get_hash(self) -> Optional[str]:
        if self.__source_kind == SourceKind.ARRAY:
            return None
        return calc_hash(self.__source)  # type: ignore[arg-type]

    def read(self, min_sample: Optional[int] = None, max_sample: Optional[int] = None) -> np.ndarray:
        return self.__traces.read(min_sample=min_sample, max_sample=max_sample)

    def read_traces_by_ids(
        self,
        ids: Sequence[int],
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> np.ndarray:
        return self.__traces.read_traces_by_ids(ids=ids, min_sample=min_sample, max_sample=max_sample)

    def get_chunked_reader(
        self,
        chunk_size: int,
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        yield from self.__traces.get_chunked_reader(chunk_size, min_sample=min_sample, max_sample=max_sample)

    def read_custom_trace_header(self, byte_position: int, encoding: str) -> tuple[Any, ...]:
        info = HeaderInfo(byte_position, encoding)
        if self.__source_kind == SourceKind.BYTES:
            return read_custom_traces_header(BytesIO(self.__source), info, self.__layout)  # type: ignore[arg-type]
        if self.__source_kind == SourceKind.FILE:
            with Path(self.__source).open("rb") as pointer:  # type: ignore[arg-type]
                return read_custom_traces_header(pointer, info, self.__layout)
        raise NotImplementedReader("Custom trace header reading is available only for file and bytes sources")

    def write(
        self,
        output_path: Union[str, Path],
        data_format: Optional[Union[DataFormat, int]] = None,
        endianness: Optional[Union[Endianness, str]] = None,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_layout = self.__make_write_layout(data_format=data_format, endianness=endianness)
        file_headers = self.__file_headers_for_layout(write_layout)
        trace_headers = TraceHeaders.from_values(self.__trace_headers.raw(), write_layout)

        with output_path.open("wb+") as pointer:
            file_headers.write_to_sgy_pointer(pointer)
            trace_headers.write_to_sgy_pointer(pointer)
            self.__traces.write_to_sgy_pointer(pointer, output_layout=write_layout)

    def export_sgy_with_picks(
        self,
        output_fname: Union[str, Path],
        picks_in_mcs: List[float],
        byte_position: int = 236,
        encoding: Optional[str] = None,
        picks_unit: Optional[str] = "mcs",
    ) -> None:
        output_path = Path(output_fname)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        effective_encoding = TraceHeaderField.FB_PICK.value.format if encoding is None else encoding
        info = HeaderInfo(byte_position, effective_encoding)
        if len(picks_in_mcs) != self.num_traces:
            raise ValueError(f"Number of picks ({len(picks_in_mcs)}) must match number of traces ({self.num_traces})")
        if picks_unit not in ("ms", "mcs", "sample"):
            raise ValueError("Argument 'picks_unit' must be one of 'ms', 'mcs', or 'sample'")

        cast_to = float if effective_encoding in ("f", "d") else int
        if picks_unit == "ms":
            picks = self.__units_converter.mcs2ms(picks_in_mcs, cast_to=cast_to)
        elif picks_unit == "sample":
            picks = self.__units_converter.mcs2index(picks_in_mcs, cast_to=cast_to)
        else:
            picks = np.asarray(picks_in_mcs).astype(cast_to)

        write_layout = self.__make_write_layout(data_format=self.sample_format, endianness=self.endianness)
        file_headers = self.__file_headers_for_layout(write_layout)
        trace_headers = TraceHeaders.from_values(self.__trace_headers.raw(), write_layout)

        with output_path.open("wb+") as pointer:
            file_headers.write_to_sgy_pointer(pointer)
            trace_headers.write_to_sgy_pointer(pointer)
            self.__traces.write_to_sgy_pointer(pointer, output_layout=write_layout)
            write_custom_traces_header(pointer, np.asarray(picks), info, write_layout)

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
            endianness=DEFAULT_ENDIANNESS,
        )
        self.__source = source
        self.__source_kind = SourceKind.ARRAY
        self.__layout = layout
        self.__file_headers = FileHeaders.from_layout(layout, overrides=file_headers)
        self.__trace_headers = (
            TraceHeaders.from_values(traces_headers, layout)
            if traces_headers is not None
            else TraceHeaders.empty(layout)
        )
        self.__traces = get_traces_backend(source, layout)

    def __build_bytes_components(self, source: bytes) -> None:
        layout = SGYLayout.from_bytes(source)
        from io import BytesIO

        pointer = BytesIO(source)
        self.__source = source
        self.__source_kind = SourceKind.BYTES
        self.__layout = layout
        self.__file_headers = FileHeaders.from_sgy_pointer(pointer, layout)
        self.__trace_headers = TraceHeaders.from_sgy_pointer(pointer, layout)
        self.__traces = get_traces_backend(source, layout)

    def __build_file_components(self, source: Path) -> None:
        layout = SGYLayout.from_file(source)
        with source.open("rb") as pointer:
            file_headers = FileHeaders.from_sgy_pointer(pointer, layout)
            trace_headers = TraceHeaders.from_sgy_pointer(pointer, layout)
        self.__source = source
        self.__source_kind = SourceKind.FILE
        self.__layout = layout
        self.__file_headers = file_headers
        self.__trace_headers = trace_headers
        self.__traces = get_traces_backend(source, layout)

    def __make_write_layout(
        self,
        *,
        data_format: Optional[Union[DataFormat, int]],
        endianness: Optional[Union[Endianness, str]],
    ) -> SGYLayout:
        return SGYLayout(
            dt_mcs=self.__layout.dt_mcs,
            num_samples=self.__layout.num_samples,
            num_traces=self.__layout.num_traces,
            data_format=self.__layout.data_format if data_format is None else DataFormat(data_format),
            endianness=self.__layout.endianness if endianness is None else Endianness(endianness),
            revision=self.__layout.revision,
            file_header_size=self.__layout.file_header_size,
            trace_header_size=self.__layout.trace_header_size,
        )

    def __file_headers_for_layout(self, layout: SGYLayout) -> FileHeaders:
        values = self.__file_headers.values()
        values[FileHeaderField.DT] = layout.dt_mcs
        values[FileHeaderField.DT_ORIG] = layout.dt_mcs
        values[FileHeaderField.NS] = layout.num_samples
        values[FileHeaderField.NS_ORIG] = layout.num_samples
        values[FileHeaderField.DATA_SAMPLE_FORMAT] = int(layout.data_format)
        return FileHeaders.from_values(values, layout)


__all__ = [
    "DataFormat",
    "Endianness",
    "InvalidSGY",
    "NotImplementedReader",
    "SGY",
    "SGYRevision",
    "SGYInitParamsError",
]
