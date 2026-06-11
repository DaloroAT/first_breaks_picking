from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, IO, Mapping, NamedTuple, Optional, Type, Union

import numpy as np
import pandas as pd

from first_breaks.sgy.types import Endianness, SGYLayout


class InvalidHeaders(Exception):
    pass


class HeaderInfo(NamedTuple):
    offset: int
    format: str


class FileHeaderField(Enum):
    TEXTUAL_FILE_HEADER = HeaderInfo(0, "3200s")
    JOB = HeaderInfo(3200, "i")
    LINE = HeaderInfo(3204, "i")
    REEL = HeaderInfo(3208, "i")
    DATA_TRACE_PER_ENSEMBLE = HeaderInfo(3212, "H")
    AUXILIARY_TRACE_PER_ENSEMBLE = HeaderInfo(3214, "H")
    DT = HeaderInfo(3216, "H")
    DT_ORIG = HeaderInfo(3218, "H")
    NS = HeaderInfo(3220, "H")
    NS_ORIG = HeaderInfo(3222, "H")
    DATA_SAMPLE_FORMAT = HeaderInfo(3224, "H")
    ENSEMBLE_FOLD = HeaderInfo(3226, "H")
    TRACE_SORTING = HeaderInfo(3228, "h")
    VERTICAL_SUM_CODE = HeaderInfo(3230, "H")
    SWEEP_FREQUENCY_START = HeaderInfo(3232, "H")
    SWEEP_FREQUENCY_END = HeaderInfo(3234, "H")
    SWEEP_LENGTH = HeaderInfo(3236, "H")
    SWEEP_TYPE = HeaderInfo(3238, "h")
    SWEEP_CHANNEL = HeaderInfo(3240, "h")
    SWEEP_TAPER_LENGTH_START = HeaderInfo(3242, "H")
    SWEEP_TAPER_LENGTH_END = HeaderInfo(3244, "H")
    TAPER_TYPE = HeaderInfo(3246, "H")
    CORRELATED_DATA_TRACES = HeaderInfo(3248, "H")
    BINARY_GAIN = HeaderInfo(3250, "H")
    AMPLITUDE_RECOVERY_METHOD = HeaderInfo(3252, "H")
    MEASUREMENT_SYSTEM = HeaderInfo(3254, "H")
    IMPULSE_SIGNAL_POLARITY = HeaderInfo(3256, "H")
    VIBRATORY_POLARITY_CODE = HeaderInfo(3258, "H")
    UNASSIGNED1 = HeaderInfo(3260, "240s")
    SEGY_FORMAT_REVISION_NUMBER = HeaderInfo(3500, "H")
    FIXED_LENGTH_TRACE_FLAG = HeaderInfo(3502, "H")
    NUMBER_OF_TEXTUAL_HEADERS = HeaderInfo(3504, "H")
    UNASSIGNED2 = HeaderInfo(3506, "94s")


class TraceHeaderField(Enum):
    TRACENO = HeaderInfo(0, "i")
    TRACE_SEQUENCE_FILE = HeaderInfo(4, "i")
    FFID = HeaderInfo(8, "i")
    CHAN = HeaderInfo(12, "i")
    SOURCE = HeaderInfo(16, "i")
    CDP = HeaderInfo(20, "i")
    SEQNO = HeaderInfo(24, "i")
    TRC_TYPE = HeaderInfo(28, "h")
    STACKNT = HeaderInfo(30, "h")
    TRFOLD = HeaderInfo(32, "h")
    DATA_USE = HeaderInfo(34, "h")
    OFFSET = HeaderInfo(36, "i")
    REC_ELEV = HeaderInfo(40, "i")
    SOU_ELEV = HeaderInfo(44, "i")
    DEPTH = HeaderInfo(48, "i")
    REC_DATUM = HeaderInfo(52, "i")
    SOU_DATUM = HeaderInfo(56, "i")
    SOU_H2OD = HeaderInfo(60, "i")
    REC_H2OD = HeaderInfo(64, "i")
    ELEVATION_SCALAR = HeaderInfo(68, "h")
    SOURCE_GROUP_SCALAR = HeaderInfo(70, "h")
    SOU_X = HeaderInfo(72, "i")
    SOU_Y = HeaderInfo(76, "i")
    REC_X = HeaderInfo(80, "i")
    REC_Y = HeaderInfo(84, "i")
    COORDINATE_UNITS = HeaderInfo(88, "h")
    WEATHERING_VELOCITY = HeaderInfo(90, "h")
    SUBWEATHERING_VELOCITY = HeaderInfo(92, "h")
    UPHOLE = HeaderInfo(94, "h")
    REC_UPHOLE = HeaderInfo(96, "h")
    SOU_STAT = HeaderInfo(98, "h")
    REC_STAT = HeaderInfo(100, "h")
    TOT_STAT = HeaderInfo(102, "h")
    LAG_TIME_A = HeaderInfo(104, "h")
    LAG_TIME_B = HeaderInfo(106, "h")
    DELAY_RECORDING_TIME = HeaderInfo(108, "h")
    TLIVE_S = HeaderInfo(110, "h")
    TFULL_S = HeaderInfo(112, "h")
    NUMSMP = HeaderInfo(114, "H")
    DT = HeaderInfo(116, "H")
    IGAIN = HeaderInfo(118, "h")
    PREAMP = HeaderInfo(120, "h")
    EARLYG = HeaderInfo(122, "h")
    COR_FLAG = HeaderInfo(124, "h")
    SWEEPFREQSTART = HeaderInfo(126, "h")
    SWEEPFREQEND = HeaderInfo(128, "h")
    SWEEPLEN = HeaderInfo(130, "h")
    SWEEPTYPE = HeaderInfo(132, "h")
    SWEEPTAPSTART = HeaderInfo(134, "h")
    SWEEPTAPEND = HeaderInfo(136, "h")
    SWEEPTAPCODE = HeaderInfo(138, "h")
    AAXFILT = HeaderInfo(140, "h")
    AAXSLOP = HeaderInfo(142, "h")
    FREQXN = HeaderInfo(144, "h")
    FXNSLOP = HeaderInfo(146, "h")
    FREQXL = HeaderInfo(148, "h")
    FREQXH = HeaderInfo(150, "h")
    FXLSLOP = HeaderInfo(152, "h")
    FXHSLOP = HeaderInfo(154, "h")
    YEAR = HeaderInfo(156, "h")
    DAY = HeaderInfo(158, "h")
    HOUR = HeaderInfo(160, "h")
    MINUTE = HeaderInfo(162, "h")
    SECOND = HeaderInfo(164, "h")
    TIME_BASIC_CODE = HeaderInfo(166, "h")
    TRACE_WEIGHTING_FACTOR = HeaderInfo(168, "h")
    GEOPHONE_GROUP_NUMBER_ROLL1 = HeaderInfo(170, "h")
    GEOPHONE_GROUP_NUMBER_FIRST = HeaderInfo(172, "h")
    GEOPHONE_GROUP_NUMBER_LAST = HeaderInfo(174, "h")
    GAP_SIZE = HeaderInfo(176, "h")
    OVER_TRAVEL = HeaderInfo(178, "h")
    CDP_X = HeaderInfo(180, "i")
    CDP_Y = HeaderInfo(184, "i")
    ILINE_NO = HeaderInfo(188, "i")
    XLINE_NO = HeaderInfo(192, "i")
    SHOT_POINT = HeaderInfo(196, "i")
    SHOT_POINT_SCALAR = HeaderInfo(200, "h")
    TRACE_VALUE_MEASUREMENT = HeaderInfo(202, "h")
    TRANSDUCTION_CONSTANT_MANTISSA = HeaderInfo(204, "i")
    TRANSDUCTION_CONSTANT_POWER = HeaderInfo(208, "h")
    TRANSDUCTION_UNIT = HeaderInfo(210, "h")
    TRACE_IDENTIFIER = HeaderInfo(212, "h")
    SCALAR_TRACE_HEADER = HeaderInfo(214, "h")
    SOURCE_TYPE = HeaderInfo(216, "h")
    SOURCE_ENERGY_DIRECTION_MANTISSA = HeaderInfo(218, "i")
    SOURCE_ENERGY_DIRECTION_EXPONENT = HeaderInfo(222, "H")
    SOURCE_MEASUREMENT_MANTISSA = HeaderInfo(224, "i")
    SOURCE_MEASUREMENT_EXPONENT = HeaderInfo(228, "H")
    SOURCE_MEASUREMENT_UNIT = HeaderInfo(230, "h")
    UNASSIGNED1 = HeaderInfo(232, "i")
    FB_PICK = HeaderInfo(236, "I")


FORMAT_TO_SIZE: Dict[str, int] = {
    "c": 1,
    "b": 1,
    "B": 1,
    "?": 1,
    "h": 2,
    "H": 2,
    "i": 4,
    "I": 4,
    "l": 4,
    "L": 4,
    "q": 8,
    "Q": 8,
    "f": 4,
    "d": 8,
    "s": 1,
}

FORMAT_TO_NUMPY_DTYPE: Dict[str, str] = {
    "b": "i1",
    "B": "u1",
    "h": "i2",
    "H": "u2",
    "i": "i4",
    "I": "u4",
    "l": "i4",
    "L": "u4",
    "q": "i8",
    "Q": "u8",
    "f": "f4",
    "d": "f8",
}

RADEX_FILE_HEADER_NAMES: Mapping[FileHeaderField, str] = {field: field.name.lower() for field in FileHeaderField}

RADEX_TRACE_HEADER_NAMES: Mapping[TraceHeaderField, str] = {field: field.name for field in TraceHeaderField}
RADEX_TRACE_HEADER_NAMES = {
    **RADEX_TRACE_HEADER_NAMES,
    TraceHeaderField.TRACE_SEQUENCE_FILE: "trace_sequence_file",
    TraceHeaderField.DATA_USE: "data_use",
    TraceHeaderField.ELEVATION_SCALAR: "elevation_scalar",
    TraceHeaderField.SOURCE_GROUP_SCALAR: "source_group_scalar",
    TraceHeaderField.COORDINATE_UNITS: "coordinate_units",
    TraceHeaderField.WEATHERING_VELOCITY: "weathering_velocity",
    TraceHeaderField.SUBWEATHERING_VELOCITY: "subweathering_velocity",
    TraceHeaderField.LAG_TIME_A: "lag_time_a",
    TraceHeaderField.LAG_TIME_B: "lag_time_b",
    TraceHeaderField.DELAY_RECORDING_TIME: "delay_recording_time",
    TraceHeaderField.TIME_BASIC_CODE: "time_basic_code",
    TraceHeaderField.TRACE_WEIGHTING_FACTOR: "trace_weighting_factor",
    TraceHeaderField.GEOPHONE_GROUP_NUMBER_ROLL1: "geophone_group_number_roll1",
    TraceHeaderField.GEOPHONE_GROUP_NUMBER_FIRST: "geophone_group_number_first",
    TraceHeaderField.GEOPHONE_GROUP_NUMBER_LAST: "geophone_group_number_last",
    TraceHeaderField.GAP_SIZE: "gap_size",
    TraceHeaderField.OVER_TRAVEL: "over_travel",
    TraceHeaderField.SHOT_POINT: "shot_point",
    TraceHeaderField.SHOT_POINT_SCALAR: "shot_point_scalar",
    TraceHeaderField.TRACE_VALUE_MEASUREMENT: "trace_value_measurement",
    TraceHeaderField.TRANSDUCTION_CONSTANT_MANTISSA: "transduction_constant_mantissa",
    TraceHeaderField.TRANSDUCTION_CONSTANT_POWER: "transduction_constant_power",
    TraceHeaderField.TRANSDUCTION_UNIT: "transduction_unit",
    TraceHeaderField.TRACE_IDENTIFIER: "trace_identifier",
    TraceHeaderField.SCALAR_TRACE_HEADER: "scalar_trace_header",
    TraceHeaderField.SOURCE_TYPE: "source_type",
    TraceHeaderField.SOURCE_ENERGY_DIRECTION_MANTISSA: "source_energy_direction_mantissa",
    TraceHeaderField.SOURCE_ENERGY_DIRECTION_EXPONENT: "source_energy_direction_exponent",
    TraceHeaderField.SOURCE_MEASUREMENT_MANTISSA: "source_measurement_mantissa",
    TraceHeaderField.SOURCE_MEASUREMENT_EXPONENT: "source_measurement_exponent",
    TraceHeaderField.SOURCE_MEASUREMENT_UNIT: "source_measurement_unit",
    TraceHeaderField.UNASSIGNED1: "unassigned1",
}

FileHeaderNameMapping = Optional[Mapping[FileHeaderField, str]]
TraceHeaderNameMapping = Optional[Mapping[TraceHeaderField, str]]


def get_num_bytes(fmt: str) -> int:
    tp = fmt[-1]
    if tp not in FORMAT_TO_SIZE:
        raise InvalidHeaders("Format is not interpretable")
    num_str = fmt[:-1]
    num = int(num_str) if num_str.isdigit() else 1
    return FORMAT_TO_SIZE[tp] * num


class FileHeadersBackend(ABC):
    @property
    @abstractmethod
    def layout(self) -> SGYLayout:
        raise NotImplementedError

    @abstractmethod
    def values(self) -> Dict[FileHeaderField, Any]:
        raise NotImplementedError

    @abstractmethod
    def write_to_sgy_pointer(self, pointer: IO[bytes]) -> None:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, field: FileHeaderField) -> Any:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, field: FileHeaderField, value: Any) -> None:
        raise NotImplementedError


class TraceHeadersBackend(ABC):
    @property
    @abstractmethod
    def layout(self) -> SGYLayout:
        raise NotImplementedError

    @abstractmethod
    def raw(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def write_to_sgy_pointer(self, pointer: IO[bytes]) -> None:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, field: TraceHeaderField) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, field: TraceHeaderField, values: Any) -> None:
        raise NotImplementedError


class FileHeadersPython(FileHeadersBackend):
    def __init__(self, values: Mapping[Union[str, FileHeaderField], Any], layout: SGYLayout) -> None:
        self.__layout = layout
        self.__values = _normalize_values(values, FileHeaderField)

    @property
    def layout(self) -> SGYLayout:
        return self.__layout

    def values(self) -> Dict[FileHeaderField, Any]:
        return deepcopy(self.__values)

    def write_to_sgy_pointer(self, pointer: IO[bytes]) -> None:
        pointer.seek(0)
        pointer.write(_encode_block(self.__values, FileHeaderField, self.layout, self.layout.file_header_size))

    def __getitem__(self, field: FileHeaderField) -> Any:
        return deepcopy(self.__values[field])

    def __setitem__(self, field: FileHeaderField, value: Any) -> None:
        _validate_value(value, field.value.format, self.layout)
        self.__values[field] = deepcopy(value)


class FileHeadersBytes(FileHeadersBackend):
    def __init__(self, raw: Union[bytes, bytearray], layout: SGYLayout) -> None:
        if len(raw) != layout.file_header_size:
            raise InvalidHeaders(f"File headers require {layout.file_header_size} bytes, got {len(raw)}")
        self.__layout = layout
        self.__raw = bytearray(raw)
        self.__values_cache: Optional[Dict[FileHeaderField, Any]] = None

    @property
    def layout(self) -> SGYLayout:
        return self.__layout

    @classmethod
    def from_sgy_pointer(cls, pointer: IO[bytes], layout: SGYLayout) -> "FileHeadersBytes":
        pointer.seek(0)
        raw = pointer.read(layout.file_header_size)
        if len(raw) != layout.file_header_size:
            raise InvalidHeaders(f"Cannot read {layout.file_header_size} bytes for file headers")
        return cls(raw, layout)

    def values(self) -> Dict[FileHeaderField, Any]:
        if self.__values_cache is None:
            self.__values_cache = _decode_block(self.__raw, FileHeaderField, self.layout, self.layout.file_header_size)
        return deepcopy(self.__values_cache)

    def write_to_sgy_pointer(self, pointer: IO[bytes]) -> None:
        pointer.seek(0)
        pointer.write(self.__raw)

    def __getitem__(self, field: FileHeaderField) -> Any:
        return deepcopy(_decode_value(self.__raw[field.value.offset : field.value.offset + get_num_bytes(field.value.format)], field, self.layout))

    def __setitem__(self, field: FileHeaderField, value: Any) -> None:
        _patch_value(self.__raw, 0, field, value, self.layout, self.layout.file_header_size)
        self.__values_cache = None


class TraceHeadersPython(TraceHeadersBackend):
    def __init__(self, values: pd.DataFrame, layout: SGYLayout) -> None:
        self.__layout = layout
        self.__raw = _normalize_trace_values(values)
        if len(self.__raw) != layout.num_traces:
            raise InvalidHeaders(f"Trace headers contain {len(self.__raw)} rows, expected {layout.num_traces}")

    @property
    def layout(self) -> SGYLayout:
        return self.__layout

    def raw(self) -> pd.DataFrame:
        return self.__raw.copy()

    def write_to_sgy_pointer(self, pointer: IO[bytes]) -> None:
        _write_trace_header_blocks(
            pointer,
            _encode_blocks(self.__raw, TraceHeaderField, self.layout, self.layout.trace_header_size),
            self.layout,
        )

    def __getitem__(self, field: TraceHeaderField) -> pd.Series:
        return self.__raw[field.name].copy()

    def __setitem__(self, field: TraceHeaderField, values: Any) -> None:
        values_array = _normalize_trace_update(values, self.layout.num_traces)
        for value in values_array:
            _validate_value(value, field.value.format, self.layout)
        self.__raw[field.name] = values_array


class TraceHeadersBytes(TraceHeadersBackend):
    def __init__(self, raw: Union[bytes, bytearray], layout: SGYLayout) -> None:
        expected_size = layout.num_traces * layout.trace_header_size
        if len(raw) != expected_size:
            raise InvalidHeaders(f"Trace headers require {expected_size} bytes, got {len(raw)}")
        self.__layout = layout
        self.__raw = bytearray(raw)
        self.__raw_cache: Optional[pd.DataFrame] = None

    @property
    def layout(self) -> SGYLayout:
        return self.__layout

    @classmethod
    def from_sgy_pointer(cls, pointer: IO[bytes], layout: SGYLayout) -> "TraceHeadersBytes":
        raw = bytearray(layout.num_traces * layout.trace_header_size)
        for trace_idx in range(layout.num_traces):
            pointer.seek(_trace_header_offset(trace_idx, layout))
            start = trace_idx * layout.trace_header_size
            stop = start + layout.trace_header_size
            block = pointer.read(layout.trace_header_size)
            if len(block) != layout.trace_header_size:
                raise InvalidHeaders(f"Cannot read {layout.trace_header_size} bytes for trace header {trace_idx}")
            raw[start:stop] = block
        return cls(raw, layout)

    def raw(self) -> pd.DataFrame:
        if self.__raw_cache is None:
            self.__raw_cache = _decode_blocks(self.__raw, TraceHeaderField, self.layout, self.layout.trace_header_size)
        return self.__raw_cache.copy()

    def write_to_sgy_pointer(self, pointer: IO[bytes]) -> None:
        _write_trace_header_blocks(pointer, self.__raw, self.layout)

    def __getitem__(self, field: TraceHeaderField) -> pd.Series:
        return self.raw()[field.name]

    def __setitem__(self, field: TraceHeaderField, values: Any) -> None:
        values_array = _normalize_trace_update(values, self.layout.num_traces)
        for trace_idx, value in enumerate(values_array):
            _patch_value(self.__raw, trace_idx, field, value, self.layout, self.layout.trace_header_size)
        self.__raw_cache = None


class FileHeaders:
    dt_name = FileHeaderField.DT.name
    dt_name_orig = FileHeaderField.DT_ORIG.name
    ns_name = FileHeaderField.NS.name
    ns_name_orig = FileHeaderField.NS_ORIG.name
    data_sample_format_name = FileHeaderField.DATA_SAMPLE_FORMAT.name

    def __init__(self, backend: FileHeadersBackend) -> None:
        self.__backend = backend

    @classmethod
    def from_layout(
        cls,
        layout: SGYLayout,
        overrides: Optional[Mapping[Union[str, FileHeaderField], Any]] = None,
    ) -> "FileHeaders":
        values = {field: _default_value_for_format(field.value.format) for field in FileHeaderField}
        values[FileHeaderField.DT] = layout.dt_mcs
        values[FileHeaderField.DT_ORIG] = layout.dt_mcs
        values[FileHeaderField.NS] = layout.num_samples
        values[FileHeaderField.NS_ORIG] = layout.num_samples
        values[FileHeaderField.DATA_SAMPLE_FORMAT] = int(layout.data_format)
        values[FileHeaderField.SEGY_FORMAT_REVISION_NUMBER] = int(layout.revision)
        if overrides is not None:
            values.update({_normalize_field(key, FileHeaderField): value for key, value in overrides.items()})
        return cls.from_values(values, layout)

    @classmethod
    def from_values(cls, values: Mapping[Union[str, FileHeaderField], Any], layout: SGYLayout) -> "FileHeaders":
        return cls(FileHeadersPython(values, layout))

    @classmethod
    def from_sgy_pointer(cls, pointer: IO[bytes], layout: SGYLayout) -> "FileHeaders":
        return cls(FileHeadersBytes.from_sgy_pointer(pointer, layout))

    def headers(self, name_mapping: FileHeaderNameMapping = None) -> Dict[str, Any]:
        return {
            field.name if name_mapping is None else name_mapping.get(field, field.name): value
            for field, value in self.__backend.values().items()
        }

    @property
    def fields(self) -> tuple[FileHeaderField, ...]:
        return tuple(FileHeaderField)

    def write_to_sgy_pointer(self, pointer: IO[bytes]) -> None:
        self.__backend.write_to_sgy_pointer(pointer)

    def __getitem__(self, key: Union[str, FileHeaderField]) -> Any:
        return self.__backend[_normalize_field(key, FileHeaderField)]

    def __setitem__(self, key: Union[str, FileHeaderField], value: Any) -> None:
        self.__backend[_normalize_field(key, FileHeaderField)] = value


class TraceHeaders:
    fb_pick_default = TraceHeaderField.FB_PICK.name
    scalar_from2apply: Dict[TraceHeaderField, tuple[TraceHeaderField, ...]] = {
        TraceHeaderField.ELEVATION_SCALAR: (
            TraceHeaderField.REC_ELEV,
            TraceHeaderField.SOU_ELEV,
            TraceHeaderField.DEPTH,
            TraceHeaderField.REC_DATUM,
            TraceHeaderField.SOU_DATUM,
            TraceHeaderField.SOU_H2OD,
            TraceHeaderField.REC_H2OD,
        ),
        TraceHeaderField.SOURCE_GROUP_SCALAR: (
            TraceHeaderField.SOU_X,
            TraceHeaderField.SOU_Y,
            TraceHeaderField.REC_X,
            TraceHeaderField.REC_Y,
        ),
        TraceHeaderField.SHOT_POINT_SCALAR: (TraceHeaderField.SHOT_POINT,),
        TraceHeaderField.SCALAR_TRACE_HEADER: (
            TraceHeaderField.UPHOLE,
            TraceHeaderField.REC_UPHOLE,
            TraceHeaderField.SOU_STAT,
            TraceHeaderField.REC_STAT,
            TraceHeaderField.TOT_STAT,
            TraceHeaderField.LAG_TIME_A,
            TraceHeaderField.LAG_TIME_B,
            TraceHeaderField.DELAY_RECORDING_TIME,
            TraceHeaderField.TLIVE_S,
            TraceHeaderField.TFULL_S,
        ),
    }

    def __init__(self, backend: TraceHeadersBackend) -> None:
        self.__backend = backend

    @classmethod
    def empty(cls, layout: SGYLayout) -> "TraceHeaders":
        values = {field.name: np.zeros(layout.num_traces, dtype=np.int64) for field in TraceHeaderField}
        return cls.from_values(pd.DataFrame(values), layout)

    @classmethod
    def from_values(cls, values: pd.DataFrame, layout: SGYLayout) -> "TraceHeaders":
        return cls(TraceHeadersPython(values, layout))

    @classmethod
    def from_sgy_pointer(cls, pointer: IO[bytes], layout: SGYLayout) -> "TraceHeaders":
        return cls(TraceHeadersBytes.from_sgy_pointer(pointer, layout))

    def raw(self, name_mapping: TraceHeaderNameMapping = None) -> pd.DataFrame:
        return _rename_trace_columns(self.__backend.raw(), name_mapping)

    def scaled(self, name_mapping: TraceHeaderNameMapping = None) -> pd.DataFrame:
        scaled = self.__backend.raw()
        for scalar_field, value_fields in self.scalar_from2apply.items():
            columns = [field.name for field in value_fields]
            scaled[columns] = _scale_values(scaled[columns], scaled[scalar_field.name])
        return _rename_trace_columns(scaled, name_mapping)

    @property
    def fields(self) -> tuple[TraceHeaderField, ...]:
        return tuple(TraceHeaderField)

    def write_to_sgy_pointer(self, pointer: IO[bytes]) -> None:
        self.__backend.write_to_sgy_pointer(pointer)

    def __getitem__(self, key: Union[str, TraceHeaderField]) -> pd.Series:
        return self.__backend[_normalize_field(key, TraceHeaderField)]

    def __setitem__(self, key: Union[str, TraceHeaderField], values: Any) -> None:
        self.__backend[_normalize_field(key, TraceHeaderField)] = values


def read_file_header(pointer: IO[bytes], field: FileHeaderField, layout: SGYLayout) -> Any:
    pointer.seek(field.value.offset)
    raw = pointer.read(get_num_bytes(field.value.format))
    if len(raw) != get_num_bytes(field.value.format):
        raise InvalidHeaders(f"Cannot read {get_num_bytes(field.value.format)} bytes for header format {field.value.format!r}")
    return _decode_value(raw, field, layout)


def write_file_header(pointer: IO[bytes], field: FileHeaderField, value: Any, layout: SGYLayout) -> None:
    pointer.seek(field.value.offset)
    pointer.write(_encode_value(value, field, layout))


def read_trace_header(pointer: IO[bytes], field: TraceHeaderField, trace_idx: int, layout: SGYLayout) -> Any:
    pointer.seek(_trace_header_offset(trace_idx, layout) + field.value.offset)
    raw = pointer.read(get_num_bytes(field.value.format))
    if len(raw) != get_num_bytes(field.value.format):
        raise InvalidHeaders(f"Cannot read {get_num_bytes(field.value.format)} bytes for header format {field.value.format!r}")
    return _decode_value(raw, field, layout)


def write_trace_header(pointer: IO[bytes], field: TraceHeaderField, value: Any, trace_idx: int, layout: SGYLayout) -> None:
    pointer.seek(_trace_header_offset(trace_idx, layout) + field.value.offset)
    pointer.write(_encode_value(value, field, layout))


def _normalize_field(key: Union[str, Enum], enum_cls: Type[Enum]) -> Any:
    if isinstance(key, enum_cls):
        return key
    if isinstance(key, str):
        try:
            return enum_cls[key]
        except KeyError as exc:
            raise InvalidHeaders(f"Unknown header name: {key}") from exc
    raise InvalidHeaders(f"Unsupported header key type: {type(key)!r}")


def _validate_complete_fields(fields: Any, enum_cls: Type[Enum]) -> None:
    actual = set(fields)
    required = set(enum_cls)
    redundant = actual - required
    missed = required - actual
    if redundant:
        raise InvalidHeaders(f"Redundant header keys: {sorted(field.name for field in redundant)}")
    if missed:
        raise InvalidHeaders(f"Missed header keys: {sorted(field.name for field in missed)}")


def _normalize_values(values: Mapping[Union[str, Enum], Any], enum_cls: Type[Enum]) -> Dict[Any, Any]:
    normalized = {_normalize_field(key, enum_cls): deepcopy(value) for key, value in values.items()}
    _validate_complete_fields(normalized.keys(), enum_cls)
    return normalized


def _normalize_trace_values(values: pd.DataFrame) -> pd.DataFrame:
    fields = [_normalize_field(column, TraceHeaderField) for column in values.columns]
    _validate_complete_fields(fields, TraceHeaderField)
    normalized = values.copy()
    normalized.columns = [field.name for field in fields]
    return normalized


def _record_dtype(enum_cls: Type[Enum], layout: SGYLayout, block_size: int) -> np.dtype:
    return np.dtype(
        {
            "names": [field.name for field in enum_cls],
            "formats": [_dtype_for_format(field.value.format, layout) for field in enum_cls],
            "offsets": [field.value.offset for field in enum_cls],
            "itemsize": block_size,
        }
    )


def _decode_block(raw: Union[bytes, bytearray], enum_cls: Type[Enum], layout: SGYLayout, block_size: int) -> Dict[Any, Any]:
    if len(raw) != block_size:
        raise InvalidHeaders(f"Header block requires {block_size} bytes, got {len(raw)}")
    record = np.frombuffer(raw, dtype=_record_dtype(enum_cls, layout, block_size), count=1)[0]
    return {field: _to_python_value(record[field.name], field.value.format) for field in enum_cls}


def _decode_blocks(raw: Union[bytes, bytearray], enum_cls: Type[Enum], layout: SGYLayout, block_size: int) -> pd.DataFrame:
    if len(raw) % block_size != 0:
        raise InvalidHeaders(f"Header blocks length {len(raw)} is not divisible by block size {block_size}")
    records = np.frombuffer(raw, dtype=_record_dtype(enum_cls, layout, block_size), count=len(raw) // block_size)
    return pd.DataFrame({field.name: records[field.name].copy() for field in enum_cls})


def _encode_block(values: Mapping[Enum, Any], enum_cls: Type[Enum], layout: SGYLayout, block_size: int) -> bytearray:
    _validate_complete_fields(values.keys(), enum_cls)
    raw = bytearray(block_size)
    record = np.frombuffer(raw, dtype=_record_dtype(enum_cls, layout, block_size), count=1)
    for field, value in values.items():
        _validate_value(value, field.value.format, layout)
        record[field.name][0] = _to_numpy_value(value, field.value.format)
    return raw


def _encode_blocks(values: pd.DataFrame, enum_cls: Type[Enum], layout: SGYLayout, block_size: int) -> bytearray:
    fields = [_normalize_field(column, enum_cls) for column in values.columns]
    _validate_complete_fields(fields, enum_cls)
    normalized = values.copy()
    normalized.columns = [field.name for field in fields]
    raw = bytearray(len(normalized) * block_size)
    records = np.frombuffer(raw, dtype=_record_dtype(enum_cls, layout, block_size), count=len(normalized))
    for field in enum_cls:
        values_array = _normalize_trace_update(normalized[field.name].to_numpy(), len(normalized))
        for value in values_array:
            _validate_value(value, field.value.format, layout)
        records[field.name] = [_to_numpy_value(value, field.value.format) for value in values_array]
    return raw


def _decode_value(raw: Union[bytes, bytearray], field: Enum, layout: SGYLayout) -> Any:
    if len(raw) != get_num_bytes(field.value.format):
        raise InvalidHeaders(f"Header value requires {get_num_bytes(field.value.format)} bytes, got {len(raw)}")
    value = np.frombuffer(raw, dtype=_dtype_for_format(field.value.format, layout), count=1)[0]
    return _to_python_value(value, field.value.format)


def _encode_value(value: Any, field: Enum, layout: SGYLayout) -> bytes:
    _validate_value(value, field.value.format, layout)
    return np.asarray([_to_numpy_value(value, field.value.format)], dtype=_dtype_for_format(field.value.format, layout)).tobytes()


def _patch_value(raw: bytearray, record_idx: int, field: Enum, value: Any, layout: SGYLayout, block_size: int) -> None:
    if len(raw) % block_size != 0:
        raise InvalidHeaders(f"Header blocks length {len(raw)} is not divisible by block size {block_size}")
    _validate_value(value, field.value.format, layout)
    records = np.frombuffer(raw, dtype=_record_dtype(type(field), layout, block_size), count=len(raw) // block_size)
    records[field.name][record_idx] = _to_numpy_value(value, field.value.format)


def _scale_values(values: pd.DataFrame, scalar: pd.Series) -> pd.DataFrame:
    scalar_values = scalar.astype(np.float64).replace(0, 1)
    scalar_values = np.where(scalar_values < 0, 1 / np.abs(scalar_values), scalar_values)
    return values.multiply(scalar_values, axis=0)


def _dtype_for_format(fmt: str, layout: SGYLayout) -> np.dtype:
    if fmt.endswith("s"):
        return np.dtype(f"S{get_num_bytes(fmt)}")
    if fmt not in FORMAT_TO_NUMPY_DTYPE:
        raise InvalidHeaders(f"Format is not interpretable by NumPy: {fmt!r}")
    return np.dtype(Endianness(layout.endianness).value + FORMAT_TO_NUMPY_DTYPE[fmt])


def _validate_value(value: Any, fmt: str, layout: SGYLayout) -> None:
    if fmt.endswith("s"):
        value_bytes = value.encode("ascii", errors="replace") if isinstance(value, str) else bytes(value)
        if len(value_bytes) > get_num_bytes(fmt):
            raise InvalidHeaders(f"Value for header format {fmt!r} requires at most {get_num_bytes(fmt)} bytes")
        return
    dtype = _dtype_for_format(fmt, layout)
    value = value.item() if isinstance(value, np.generic) else value
    if dtype.kind in ("i", "u"):
        info = np.iinfo(dtype)
        if not info.min <= int(value) <= info.max:
            raise InvalidHeaders(f"Value {value!r} is outside range [{info.min}, {info.max}] for header format {fmt!r}")
    elif dtype.kind == "f":
        try:
            float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidHeaders(f"Value {value!r} cannot be encoded as header format {fmt!r}") from exc


def _to_numpy_value(value: Any, fmt: str) -> Any:
    if fmt.endswith("s"):
        value_bytes = value.encode("ascii", errors="replace") if isinstance(value, str) else bytes(value)
        return value_bytes.ljust(get_num_bytes(fmt), b"\x00")
    return value.item() if isinstance(value, np.generic) else value


def _to_python_value(value: Any, fmt: str) -> Any:
    if fmt.endswith("s"):
        return bytes(value).rstrip(b"\x00")
    return value.item() if isinstance(value, np.generic) else value


def _default_value_for_format(fmt: str) -> Any:
    return b"" if fmt.endswith("s") else 0


def _normalize_trace_update(values: Any, num_traces: int) -> np.ndarray:
    values_array = np.asarray(values)
    if values_array.ndim == 0:
        values_array = np.full(num_traces, values_array.item())
    if len(values_array) != num_traces:
        raise InvalidHeaders(f"Trace header update contains {len(values_array)} values, expected {num_traces}")
    return values_array


def _trace_header_offset(trace_idx: int, layout: SGYLayout) -> int:
    return layout.file_header_size + trace_idx * layout.trace_block_size


def _write_trace_header_blocks(pointer: IO[bytes], raw: Union[bytes, bytearray], layout: SGYLayout) -> None:
    for trace_idx in range(layout.num_traces):
        pointer.seek(_trace_header_offset(trace_idx, layout))
        start = trace_idx * layout.trace_header_size
        stop = start + layout.trace_header_size
        pointer.write(raw[start:stop])


def _rename_trace_columns(values: pd.DataFrame, name_mapping: TraceHeaderNameMapping) -> pd.DataFrame:
    renamed = values.copy()
    if name_mapping is not None:
        renamed = renamed.rename(columns={field.name: name_mapping.get(field, field.name) for field in TraceHeaderField})
    return renamed
