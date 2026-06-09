from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from first_breaks.sgy.headers import FileHeaders, TraceHeaders
from first_breaks.sgy.traces import Traces
from first_breaks.sgy.types import (
    DEFAULT_DATA_FORMAT,
    DEFAULT_ENDIANESS,
    FORMAT_TO_BYTES_PER_SAMPLE,
    DataFormat,
    Endianness,
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
from first_breaks.utils.utils import UnitsConverter


class SGY:
    fmt2bps = FORMAT_TO_BYTES_PER_SAMPLE

    def __init__(
        self,
        source: SourceInput,
        dt_mcs: Optional[Union[int, float]] = None,
        file_headers: Optional[Dict[str, Any]] = None,
        traces_headers: Optional[pd.DataFrame] = None,
    ) -> None:
        (
            self.__source,
            self.__layout,
            self.__file_headers,
            self.__trace_headers,
            self.__traces,
        ) = self.__build_components(
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
    def source(self) -> Optional[SourceInput]:
        return self.__source.value

    @property
    def source_ref(self) -> SGYSource:
        return self.__source

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
    def traces(self) -> Traces:
        return self.__traces

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
        return self.__source.kind == SourceKind.ARRAY

    @property
    def general_headers(self) -> Dict[str, Any]:
        return self.__file_headers.headers()

    @property
    def traces_headers(self) -> pd.DataFrame:
        return self.__trace_headers.scaled()

    @property
    def traces_headers_raw(self) -> pd.DataFrame:
        return self.__trace_headers.raw()

    def ms2index(self, ms_value: float) -> int:
        return self.__units_converter.ms2index(ms_value)

    def get_hash(self) -> Optional[str]:
        raise NotImplementedError

    def read(self, min_sample: Optional[int] = None, max_sample: Optional[int] = None) -> np.ndarray:
        return self.__traces.read(min_sample=min_sample, max_sample=max_sample)

    def read_traces_by_ids(
        self,
        ids: Sequence[int],
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> np.ndarray:
        return self.__traces.read_by_ids(ids, min_sample=min_sample, max_sample=max_sample)

    def get_chunked_reader(
        self,
        chunk_size: int,
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        if chunk_size <= 0:
            raise ValueError("Argument 'chunk_size' must be positive")
        for start in range(0, self.num_traces, chunk_size):
            stop = min(start + chunk_size, self.num_traces)
            yield self.read_traces_by_ids(range(start, stop), min_sample=min_sample, max_sample=max_sample)

    def replace_traces(self, traces: np.ndarray) -> None:
        self.__traces.replace_array(traces)

    def read_custom_trace_header(self, byte_position: int, encoding: str) -> Tuple[Any, ...]:
        raise NotImplementedError

    def write(
        self,
        output_path: Union[str, Path],
        data_format: Optional[Union[DataFormat, int]] = None,
        endianess: Optional[Union[Endianness, str]] = None,
    ) -> None:
        raise NotImplementedError

    def export_sgy_with_picks(
        self,
        output_fname: Union[str, Path],
        picks_in_mcs: List[float],
        byte_position: int = 236,
        encoding: Optional[str] = None,
        picks_unit: Optional[str] = "mcs",
    ) -> None:
        raise NotImplementedError

    def __build_components(
        self,
        *,
        source: SourceInput,
        dt_mcs: Optional[Union[int, float]],
        file_headers: Optional[Dict[str, Any]],
        traces_headers: Optional[pd.DataFrame],
    ) -> tuple[SGYSource, SGYLayout, FileHeaders, TraceHeaders, Traces]:
        if isinstance(source, np.ndarray):
            return self.__build_array_components(source, dt_mcs, file_headers, traces_headers)

        if dt_mcs is not None:
            raise SGYInitParamsError("Argument 'dt_mcs' must be empty if SGY is created from external source")
        if isinstance(source, bytes):
            SGYLayout.from_bytes(source)
            raise NotImplementedError("SGY parsing from bytes is not implemented yet")
        if isinstance(source, (str, Path)):
            SGYLayout.from_file(source)
            raise NotImplementedError("SGY parsing from files is not implemented yet")
        raise SGYInitParamsError("Only `str`, `Path`, `bytes`, and `np.ndarray` sources are supported")

    def __build_array_components(
        self,
        source: np.ndarray,
        dt_mcs: Optional[Union[int, float]],
        file_headers: Optional[Dict[str, Any]],
        traces_headers: Optional[pd.DataFrame],
    ) -> tuple[SGYSource, SGYLayout, FileHeaders, TraceHeaders, Traces]:
        if dt_mcs is None:
            raise SGYInitParamsError("Argument 'dt_mcs' is required if np.ndarray is used as input")
        layout = SGYLayout.from_array(
            source,
            dt_mcs=dt_mcs,
            data_format=DEFAULT_DATA_FORMAT,
            endianness=DEFAULT_ENDIANESS,
        )
        source_ref = SGYSource(kind=SourceKind.ARRAY, value=source)
        file_header_component = FileHeaders.from_layout(layout, overrides=file_headers)
        trace_header_component = (
            TraceHeaders.from_values(traces_headers, layout) if traces_headers is not None else TraceHeaders.empty(layout)
        )
        if len(trace_header_component.scaled()) not in (0, layout.num_traces):
            raise SGYInitParamsError(
                f"Trace headers contain {len(trace_header_component.scaled())} rows, expected {layout.num_traces}"
            )
        traces_component = Traces.from_array(source, layout, copy=False)
        return source_ref, layout, file_header_component, trace_header_component, traces_component


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
