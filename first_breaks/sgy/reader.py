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
    SGY_REVISION_OFFSET,
    SUPPORTED_SGY_REVISION,
    DataFormat,
    Endianess,
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
    ensure_supported_revision,
)
from first_breaks.utils.utils import UnitsConverter


class SGY:
    fmt2bps = FORMAT_TO_BYTES_PER_SAMPLE

    def __init__(
        self,
        source: SourceInput,
        dt_mcs: Optional[Union[int, float]] = None,
        general_headers_schema: Optional[FileHeaders] = None,
        traces_headers_schema: Optional[TraceHeaders] = None,
        file_headers: Optional[Dict[str, Any]] = None,
        traces_headers: Optional[pd.DataFrame] = None,
    ) -> None:
        self._file_headers_template = general_headers_schema or FileHeaders()
        self._trace_headers_template = traces_headers_schema or TraceHeaders()
        (
            self._source,
            self._revision,
            self._layout,
            self._file_headers,
            self._trace_headers,
            self._traces,
        ) = self._build_components(
            source=source,
            dt_mcs=dt_mcs,
            file_headers=file_headers,
            traces_headers=traces_headers,
        )
        self._units_converter = UnitsConverter(sgy_mcs=self._layout.dt_mcs)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        general_headers_schema: Optional[FileHeaders] = None,
        traces_headers_schema: Optional[TraceHeaders] = None,
    ) -> "SGY":
        return cls(path, general_headers_schema=general_headers_schema, traces_headers_schema=traces_headers_schema)

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        general_headers_schema: Optional[FileHeaders] = None,
        traces_headers_schema: Optional[TraceHeaders] = None,
    ) -> "SGY":
        return cls(payload, general_headers_schema=general_headers_schema, traces_headers_schema=traces_headers_schema)

    @classmethod
    def from_array(
        cls,
        traces: np.ndarray,
        *,
        dt_mcs: Union[int, float],
        file_headers: Optional[Dict[str, Any]] = None,
        traces_headers: Optional[pd.DataFrame] = None,
        general_headers_schema: Optional[FileHeaders] = None,
        traces_headers_schema: Optional[TraceHeaders] = None,
    ) -> "SGY":
        return cls(
            traces,
            dt_mcs=dt_mcs,
            general_headers_schema=general_headers_schema,
            traces_headers_schema=traces_headers_schema,
            file_headers=file_headers,
            traces_headers=traces_headers,
        )

    @property
    def source(self) -> Optional[SourceInput]:
        return self._source.value

    @property
    def source_ref(self) -> SGYSource:
        return self._source

    @property
    def layout(self) -> SGYLayout:
        return self._layout

    @property
    def revision(self) -> SGYRevision:
        return self._revision

    @property
    def file_headers(self) -> FileHeaders:
        return self._file_headers

    @property
    def trace_headers(self) -> TraceHeaders:
        return self._trace_headers

    @property
    def traces(self) -> Traces:
        return self._traces

    @property
    def dt(self) -> int:
        return self._layout.dt

    @property
    def dt_mcs(self) -> int:
        return self._layout.dt_mcs

    @property
    def dt_ms(self) -> float:
        return self._layout.dt_ms

    @property
    def fs(self) -> float:
        return self._layout.fs

    @property
    def ns(self) -> int:
        return self._layout.ns

    @property
    def ntr(self) -> int:
        return self._layout.ntr

    @property
    def num_samples(self) -> int:
        return self._layout.num_samples

    @property
    def num_traces(self) -> int:
        return self._layout.num_traces

    @property
    def shape(self) -> SizeHW:
        return self._layout.shape

    @property
    def max_time_ms(self) -> float:
        return self._layout.max_time_ms

    @property
    def endianess(self) -> str:
        return self._layout.endianness.value

    @property
    def endianness(self) -> Endianness:
        return self._layout.endianness

    @property
    def data_format(self) -> int:
        return int(self._layout.data_format)

    @property
    def sample_format(self) -> DataFormat:
        return self._layout.data_format

    @property
    def is_source_ndarray(self) -> bool:
        return self._source.kind == SourceKind.ARRAY

    @property
    def general_headers(self) -> Dict[str, Any]:
        return self._file_headers.to_dict()

    @property
    def traces_headers(self) -> pd.DataFrame:
        return self._trace_headers.to_dataframe()

    @property
    def traces_headers_raw(self) -> pd.DataFrame:
        return self._trace_headers.raw_dataframe()

    @property
    def general_headers_schema(self) -> FileHeaders:
        return self._file_headers

    @property
    def traces_headers_schema(self) -> TraceHeaders:
        return self._trace_headers

    def ms2index(self, ms_value: float) -> int:
        return self._units_converter.ms2index(ms_value)

    def get_hash(self) -> Optional[str]:
        raise NotImplementedError

    def read(self, min_sample: Optional[int] = None, max_sample: Optional[int] = None) -> np.ndarray:
        return self._traces.read(min_sample=min_sample, max_sample=max_sample)

    def read_traces_by_ids(
        self,
        ids: Sequence[int],
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> np.ndarray:
        return self._traces.read_by_ids(ids, min_sample=min_sample, max_sample=max_sample)

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
        self._traces.replace_array(traces)

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

    def _build_components(
        self,
        *,
        source: SourceInput,
        dt_mcs: Optional[Union[int, float]],
        file_headers: Optional[Dict[str, Any]],
        traces_headers: Optional[pd.DataFrame],
    ) -> tuple[SGYSource, SGYRevision, SGYLayout, FileHeaders, TraceHeaders, Traces]:
        if isinstance(source, np.ndarray):
            return self._build_array_components(source, dt_mcs, file_headers, traces_headers)

        if dt_mcs is not None:
            raise SGYInitParamsError("Argument 'dt_mcs' must be empty if SGY is created from external source")
        if isinstance(source, bytes):
            revision = self._detect_revision_from_bytes(source)
            ensure_supported_revision(revision)
            raise NotImplementedError("SGY parsing from bytes is not implemented yet")
        if isinstance(source, (str, Path)):
            revision = self._detect_revision_from_file(source)
            ensure_supported_revision(revision)
            raise NotImplementedError("SGY parsing from files is not implemented yet")
        raise SGYInitParamsError("Only `str`, `Path`, `bytes`, and `np.ndarray` sources are supported")

    def _build_array_components(
        self,
        source: np.ndarray,
        dt_mcs: Optional[Union[int, float]],
        file_headers: Optional[Dict[str, Any]],
        traces_headers: Optional[pd.DataFrame],
    ) -> tuple[SGYSource, SGYRevision, SGYLayout, FileHeaders, TraceHeaders, Traces]:
        if dt_mcs is None:
            raise SGYInitParamsError("Argument 'dt_mcs' is required if np.ndarray is used as input")
        if source.ndim not in (1, 2):
            raise SGYInitParamsError("Only 1D and 2D arrays can be used as SGY traces")

        num_samples = int(source.shape[0])
        num_traces = 1 if source.ndim == 1 else int(source.shape[1])
        layout = SGYLayout(
            dt_mcs=int(dt_mcs),
            num_samples=num_samples,
            num_traces=num_traces,
            data_format=DEFAULT_DATA_FORMAT,
            endianness=DEFAULT_ENDIANESS,
        )
        source_ref = SGYSource(kind=SourceKind.ARRAY, value=source)
        file_header_component = FileHeaders.from_layout(
            layout,
            values=file_headers,
            schema=self._file_headers_template.schema,
        )
        trace_header_component = TraceHeaders(
            traces_headers,
            schema=self._trace_headers_template.schema,
        ) if traces_headers is not None else TraceHeaders.empty(num_traces, schema=self._trace_headers_template.schema)
        if len(trace_header_component.to_dataframe()) not in (0, num_traces):
            raise SGYInitParamsError(
                f"Trace headers contain {len(trace_header_component.to_dataframe())} rows, expected {num_traces}"
            )
        traces_component = Traces.from_array(source, layout, copy=False)
        return source_ref, SUPPORTED_SGY_REVISION, layout, file_header_component, trace_header_component, traces_component

    def _detect_revision_from_bytes(self, payload: bytes) -> SGYRevision:
        revision_stop = SGY_REVISION_OFFSET + 2
        if len(payload) < revision_stop:
            raise InvalidSGY(
                f"SEG-Y source is too small to contain revision header: expected at least {revision_stop} bytes"
            )
        return SGYRevision.from_bytes(payload[SGY_REVISION_OFFSET:revision_stop], endianness=DEFAULT_ENDIANESS)

    def _detect_revision_from_file(self, path: Union[str, Path]) -> SGYRevision:
        with Path(path).open("rb") as f:
            f.seek(SGY_REVISION_OFFSET)
            raw = f.read(2)
        return SGYRevision.from_bytes(raw, endianness=DEFAULT_ENDIANESS)


__all__ = [
    "DataFormat",
    "Endianess",
    "Endianness",
    "InvalidSGY",
    "InvalidSamplesSlice",
    "NotImplementedReader",
    "SGY",
    "SGYRevision",
    "SGYInitParamsError",
]
