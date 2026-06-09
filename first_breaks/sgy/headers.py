from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any, Dict, IO, Mapping, NamedTuple, Optional, Type, Union

import numpy as np
import pandas as pd

from first_breaks.sgy.types import SGYLayout, Endianness


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

RADEX_FILE_HEADER_NAMES: Mapping[FileHeaderField, str] = {
    field: field.name.lower() for field in FileHeaderField
}

RADEX_TRACE_HEADER_NAMES: Mapping[TraceHeaderField, str] = {
    field: field.name for field in TraceHeaderField
}
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


def normalize_file_header_field(key: Union[str, FileHeaderField]) -> FileHeaderField:
    return _normalize_header_field(key, FileHeaderField)


def normalize_trace_header_field(key: Union[str, TraceHeaderField]) -> TraceHeaderField:
    return _normalize_header_field(key, TraceHeaderField)


def _normalize_file_header_dict(headers: Mapping[Union[str, FileHeaderField], Any]) -> Dict[FileHeaderField, Any]:
    return _normalize_header_dict(headers, FileHeaderField)


def _validate_file_header_fields(fields: Any) -> None:
    _validate_header_fields(fields, FileHeaderField)


def _validate_trace_header_fields(fields: Any) -> None:
    _validate_header_fields(fields, TraceHeaderField)


def _normalize_header_field(key: Union[str, Enum], enum_cls: Type[Enum]) -> Any:
    if isinstance(key, enum_cls):
        return key
    if isinstance(key, str):
        try:
            return enum_cls[key]
        except KeyError as exc:
            raise InvalidHeaders(f"Unknown header name: {key}") from exc
    raise InvalidHeaders(f"Unsupported header key type: {type(key)!r}")


def _normalize_header_dict(headers: Mapping[Union[str, Enum], Any], enum_cls: Type[Enum]) -> Dict[Any, Any]:
    normalized = {_normalize_header_field(key, enum_cls): value for key, value in headers.items()}
    _validate_header_fields(normalized.keys(), enum_cls)
    return normalized


def _validate_header_fields(fields: Any, enum_cls: Type[Enum]) -> None:
    actual = set(fields)
    required = set(enum_cls)
    redundant = actual - required
    missed = required - actual
    if redundant:
        raise InvalidHeaders(f"Redundant header keys: {sorted(field.name for field in redundant)}")
    if missed:
        raise InvalidHeaders(f"Missed header keys: {sorted(field.name for field in missed)}")


def _file_header_name(field: FileHeaderField, name_mapping: FileHeaderNameMapping = None) -> str:
    if name_mapping is None:
        return field.name
    return name_mapping.get(field, field.name)


def _trace_header_name(field: TraceHeaderField, name_mapping: TraceHeaderNameMapping = None) -> str:
    if name_mapping is None:
        return field.name
    return name_mapping.get(field, field.name)


def _endian_prefix(endianness: Union[Endianness, str]) -> str:
    return Endianness(endianness).value


def _default_value_for_format(fmt: str) -> Any:
    if fmt.endswith("s"):
        return b""
    return 0


def _scale_values(values: pd.DataFrame, scalar: pd.Series) -> pd.DataFrame:
    scalar_values = scalar.astype(np.float64).replace(0, 1)
    scalar_values = np.where(scalar_values < 0, 1 / np.abs(scalar_values), scalar_values)
    return values.multiply(scalar_values, axis=0)


def _unscale_values(values: pd.DataFrame, scalar: pd.Series) -> pd.DataFrame:
    scalar_values = scalar.astype(np.float64).replace(0, 1)
    scalar_values = np.where(scalar_values < 0, np.abs(scalar_values), 1 / scalar_values)
    return values.multiply(scalar_values, axis=0)


class FileHeaders:
    dt_name = FileHeaderField.DT.name
    dt_name_orig = FileHeaderField.DT_ORIG.name
    ns_name = FileHeaderField.NS.name
    ns_name_orig = FileHeaderField.NS_ORIG.name
    data_sample_format_name = FileHeaderField.DATA_SAMPLE_FORMAT.name

    def __init__(self, raw: Union[bytes, bytearray], layout: SGYLayout) -> None:
        if len(raw) != layout.file_header_size:
            raise InvalidHeaders(f"File headers require {layout.file_header_size} bytes, got {len(raw)}")
        self.__layout = layout
        self.__raw = bytearray(raw)
        self.__headers_cache: Optional[Dict[FileHeaderField, Any]] = None

    @classmethod
    def from_layout(
        cls,
        layout: SGYLayout,
        overrides: Optional[Mapping[Union[str, FileHeaderField], Any]] = None,
    ) -> "FileHeaders":
        headers = {field: _default_value_for_format(field.value.format) for field in FileHeaderField}
        headers[FileHeaderField.DT] = layout.dt_mcs
        headers[FileHeaderField.DT_ORIG] = layout.dt_mcs
        headers[FileHeaderField.NS] = layout.num_samples
        headers[FileHeaderField.NS_ORIG] = layout.num_samples
        headers[FileHeaderField.DATA_SAMPLE_FORMAT] = int(layout.data_format)
        headers[FileHeaderField.SEGY_FORMAT_REVISION_NUMBER] = int(layout.revision)
        if overrides is not None:
            for key, value in overrides.items():
                headers[normalize_file_header_field(key)] = value
        return cls.from_values(headers, layout)

    @classmethod
    def from_values(cls, values: Mapping[Union[str, FileHeaderField], Any], layout: SGYLayout) -> "FileHeaders":
        headers = _normalize_file_header_dict(values)
        raw = bytearray(layout.file_header_size)
        _patch_record(raw, 0, headers, FileHeaderField, layout, layout.file_header_size)
        return cls(raw, layout)

    @classmethod
    def from_sgy_pointer(cls, pointer: IO[bytes], layout: SGYLayout) -> "FileHeaders":
        pointer.seek(0)
        raw = pointer.read(layout.file_header_size)
        if len(raw) != layout.file_header_size:
            raise InvalidHeaders(f"Cannot read {layout.file_header_size} bytes for file headers")
        return cls(raw, layout)

    # Backward-compatible alias while this module is settling.
    read_from_sgy_pointer = from_sgy_pointer

    def write_with_sgy_pointer(self, pointer: IO[bytes]) -> None:
        pointer.seek(0)
        pointer.write(self.to_bytes())

    def headers(self, name_mapping: FileHeaderNameMapping = None) -> Dict[str, Any]:
        headers = self.__materialize_headers()
        return {_file_header_name(field, name_mapping): deepcopy(value) for field, value in headers.items()}

    @property
    def fields(self) -> tuple[FileHeaderField, ...]:
        return tuple(FileHeaderField)

    def to_bytes(self) -> bytes:
        return bytes(self.__raw)

    def set(self, key: Union[str, FileHeaderField], value: Any) -> None:
        field = normalize_file_header_field(key)
        _patch_record(self.__raw, 0, {field: value}, FileHeaderField, self.__layout, self.__layout.file_header_size)
        self.__headers_cache = None

    def __getitem__(self, key: Union[str, FileHeaderField]) -> Any:
        field = normalize_file_header_field(key)
        return deepcopy(_read_field_from_record(self.__raw, 0, field, self.__layout))

    def __materialize_headers(self) -> Dict[FileHeaderField, Any]:
        if self.__headers_cache is None:
            self.__headers_cache = _materialize_record(self.__raw, 0, FileHeaderField, self.__layout)
        return self.__headers_cache


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

    def __init__(self, raw: Union[bytes, bytearray], layout: SGYLayout) -> None:
        expected_size = layout.num_traces * layout.trace_header_size
        if len(raw) != expected_size:
            raise InvalidHeaders(f"Trace headers require {expected_size} bytes, got {len(raw)}")
        self.__layout = layout
        self.__raw = bytearray(raw)
        self.__raw_dataframe_cache: Optional[pd.DataFrame] = None
        self.__scaled_dataframe_cache: Optional[pd.DataFrame] = None

    @classmethod
    def empty(cls, layout: SGYLayout) -> "TraceHeaders":
        headers = {
            field.name: np.zeros(layout.num_traces, dtype=np.int64)
            for field in TraceHeaderField
        }
        return cls.from_values(pd.DataFrame(headers), layout)

    @classmethod
    def from_values(cls, values: pd.DataFrame, layout: SGYLayout) -> "TraceHeaders":
        normalized = _normalize_trace_dataframe(values)
        if len(normalized) != layout.num_traces:
            raise InvalidHeaders(f"Trace headers contain {len(normalized)} rows, expected {layout.num_traces}")
        raw = bytearray(layout.num_traces * layout.trace_header_size)
        for trace_idx, (_, row) in enumerate(normalized.iterrows()):
            values_by_field = {field: row[field.name] for field in TraceHeaderField}
            _patch_record(raw, trace_idx, values_by_field, TraceHeaderField, layout, layout.trace_header_size)
        return cls(raw, layout)

    @classmethod
    def from_sgy_pointer(cls, pointer: IO[bytes], layout: SGYLayout) -> "TraceHeaders":
        raw = bytearray(layout.num_traces * layout.trace_header_size)
        for trace_idx in range(layout.num_traces):
            pointer.seek(layout.file_header_size + trace_idx * layout.trace_block_size)
            start = trace_idx * layout.trace_header_size
            stop = start + layout.trace_header_size
            block = pointer.read(layout.trace_header_size)
            if len(block) != layout.trace_header_size:
                raise InvalidHeaders(f"Cannot read {layout.trace_header_size} bytes for trace header {trace_idx}")
            raw[start:stop] = block
        return cls(raw, layout)

    # Backward-compatible alias while this module is settling.
    read_from_sgy_pointer = from_sgy_pointer

    def write_with_sgy_pointer(self, pointer: IO[bytes]) -> None:
        for trace_idx in range(self.__layout.num_traces):
            pointer.seek(self.__layout.file_header_size + trace_idx * self.__layout.trace_block_size)
            start = trace_idx * self.__layout.trace_header_size
            stop = start + self.__layout.trace_header_size
            pointer.write(self.__raw[start:stop])

    def raw(self, name_mapping: TraceHeaderNameMapping = None) -> pd.DataFrame:
        return self.__rename_dataframe(self.__materialize_raw(), name_mapping)

    def scaled(self, name_mapping: TraceHeaderNameMapping = None) -> pd.DataFrame:
        if self.__scaled_dataframe_cache is None:
            scaled = self.__materialize_raw().copy()
            for scalar_field, value_fields in self.scalar_from2apply.items():
                columns = [field.name for field in value_fields]
                scaled[columns] = _scale_values(scaled[columns], self.__materialize_raw()[scalar_field.name])
            self.__scaled_dataframe_cache = scaled
        return self.__rename_dataframe(self.__scaled_dataframe_cache, name_mapping)

    @property
    def fields(self) -> tuple[TraceHeaderField, ...]:
        return tuple(TraceHeaderField)

    def __getitem__(self, key: Union[str, TraceHeaderField]) -> pd.Series:
        field = normalize_trace_header_field(key)
        return self.scaled()[field.name]

    def to_bytes(self) -> bytes:
        return bytes(self.__raw)

    def set(self, field: Union[str, TraceHeaderField], values: Union[Any, pd.Series, np.ndarray]) -> None:
        parsed_field = normalize_trace_header_field(field)
        values_array = np.asarray(values)
        if values_array.ndim == 0:
            values_array = np.full(self.__layout.num_traces, values_array.item())
        if len(values_array) != self.__layout.num_traces:
            raise InvalidHeaders(f"Trace header update contains {len(values_array)} values, expected {self.__layout.num_traces}")
        for trace_idx, value in enumerate(values_array):
            _patch_record(self.__raw, trace_idx, {parsed_field: value}, TraceHeaderField, self.__layout, self.__layout.trace_header_size)
        self.__invalidate_cache()

    def __materialize_raw(self) -> pd.DataFrame:
        if self.__raw_dataframe_cache is None:
            records = _records_from_raw(self.__raw, TraceHeaderField, self.__layout, self.__layout.trace_header_size)
            self.__raw_dataframe_cache = pd.DataFrame({field.name: records[field.name] for field in TraceHeaderField})
        return self.__raw_dataframe_cache

    def __rename_dataframe(self, headers: pd.DataFrame, name_mapping: TraceHeaderNameMapping = None) -> pd.DataFrame:
        renamed = headers.copy()
        if name_mapping is not None:
            renamed = renamed.rename(columns={field.name: _trace_header_name(field, name_mapping) for field in TraceHeaderField})
        return renamed

    def __invalidate_cache(self) -> None:
        self.__raw_dataframe_cache = None
        self.__scaled_dataframe_cache = None


def read_file_header(
    pointer: IO[bytes],
    field: FileHeaderField,
    layout: SGYLayout,
) -> Any:
    pointer.seek(field.value.offset)
    raw = pointer.read(get_num_bytes(field.value.format))
    if len(raw) != get_num_bytes(field.value.format):
        raise InvalidHeaders(f"Cannot read {get_num_bytes(field.value.format)} bytes for header format {field.value.format!r}")
    return _decode_value(raw, field.value, layout)


def write_file_header(
    pointer: IO[bytes],
    field: FileHeaderField,
    value: Any,
    layout: SGYLayout,
) -> None:
    pointer.seek(field.value.offset)
    pointer.write(_encode_value(value, field.value, layout))


def read_trace_header(
    pointer: IO[bytes],
    field: TraceHeaderField,
    trace_idx: int,
    layout: SGYLayout,
) -> Any:
    pointer.seek(_trace_header_offset(field, trace_idx, layout))
    raw = pointer.read(get_num_bytes(field.value.format))
    if len(raw) != get_num_bytes(field.value.format):
        raise InvalidHeaders(f"Cannot read {get_num_bytes(field.value.format)} bytes for header format {field.value.format!r}")
    return _decode_value(raw, field.value, layout)


def write_trace_header(
    pointer: IO[bytes],
    field: TraceHeaderField,
    value: Any,
    trace_idx: int,
    layout: SGYLayout,
) -> None:
    pointer.seek(_trace_header_offset(field, trace_idx, layout))
    pointer.write(_encode_value(value, field.value, layout))


def _trace_header_offset(field: TraceHeaderField, trace_idx: int, layout: SGYLayout) -> int:
    return layout.file_header_size + trace_idx * layout.trace_block_size + field.value.offset


def _trace_header_dtype(layout: SGYLayout) -> np.dtype:
    return _record_dtype(TraceHeaderField, layout, layout.trace_header_size)


def _record_dtype(enum_cls: Type[Enum], layout: SGYLayout, itemsize: int) -> np.dtype:
    prefix = _endian_prefix(layout.endianness)
    return np.dtype(
        {
            "names": [field.name for field in enum_cls],
            "formats": [_numpy_format(field.value.format, prefix) for field in enum_cls],
            "offsets": [field.value.offset for field in enum_cls],
            "itemsize": itemsize,
        }
    )


def _numpy_format(fmt: str, endian_prefix: str) -> str:
    if fmt.endswith("s"):
        return f"S{get_num_bytes(fmt)}"
    if fmt not in FORMAT_TO_NUMPY_DTYPE:
        raise InvalidHeaders(f"Format is not interpretable by NumPy: {fmt!r}")
    return endian_prefix + FORMAT_TO_NUMPY_DTYPE[fmt]


def _records_from_raw(raw: bytearray, enum_cls: Type[Enum], layout: SGYLayout, itemsize: int) -> np.ndarray:
    dtype = _record_dtype(enum_cls, layout, itemsize)
    count = len(raw) // itemsize
    return np.frombuffer(raw, dtype=dtype, count=count)


def _materialize_record(raw: bytearray, record_idx: int, enum_cls: Type[Enum], layout: SGYLayout) -> Dict[Any, Any]:
    itemsize = layout.file_header_size if enum_cls is FileHeaderField else layout.trace_header_size
    records = _records_from_raw(raw, enum_cls, layout, itemsize)
    return {field: _to_python_value(records[field.name][record_idx], field.value.format) for field in enum_cls}


def _read_field_from_record(raw: bytearray, record_idx: int, field: Enum, layout: SGYLayout) -> Any:
    itemsize = layout.file_header_size if isinstance(field, FileHeaderField) else layout.trace_header_size
    records = _records_from_raw(raw, type(field), layout, itemsize)
    return _to_python_value(records[field.name][record_idx], field.value.format)


def _patch_record(
    raw: bytearray,
    record_idx: int,
    values: Mapping[Enum, Any],
    enum_cls: Type[Enum],
    layout: SGYLayout,
    itemsize: int,
) -> None:
    records = _records_from_raw(raw, enum_cls, layout, itemsize)
    for field, value in values.items():
        _validate_value(value, field.value.format, layout)
        records[field.name][record_idx] = _prepare_value(value, field.value.format)


def _decode_value(raw: bytes, info: HeaderInfo, layout: SGYLayout) -> Any:
    dtype = np.dtype(_numpy_format(info.format, _endian_prefix(layout.endianness)))
    value = np.frombuffer(raw, dtype=dtype, count=1)[0]
    return _to_python_value(value, info.format)


def _encode_value(value: Any, info: HeaderInfo, layout: SGYLayout) -> bytes:
    _validate_value(value, info.format, layout)
    dtype = np.dtype(_numpy_format(info.format, _endian_prefix(layout.endianness)))
    return np.asarray([_prepare_value(value, info.format)], dtype=dtype).tobytes()


def _prepare_value(value: Any, fmt: str) -> Any:
    if fmt.endswith("s"):
        value_bytes = value.encode("ascii", errors="replace") if isinstance(value, str) else bytes(value)
        return value_bytes.ljust(get_num_bytes(fmt), b"\x00")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_python_value(value: Any, fmt: str) -> Any:
    if fmt.endswith("s"):
        return bytes(value).rstrip(b"\x00")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_value(value: Any, fmt: str, layout: SGYLayout) -> None:
    if fmt.endswith("s"):
        value_bytes = value.encode("ascii", errors="replace") if isinstance(value, str) else bytes(value)
        if len(value_bytes) > get_num_bytes(fmt):
            raise InvalidHeaders(f"Value for header format {fmt!r} requires at most {get_num_bytes(fmt)} bytes")
        return
    dtype = np.dtype(_numpy_format(fmt, _endian_prefix(layout.endianness)))
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


def _normalize_trace_dataframe(headers_raw: pd.DataFrame) -> pd.DataFrame:
    field_columns = [normalize_trace_header_field(column) for column in headers_raw.columns]
    _validate_trace_header_fields(field_columns)
    normalized = headers_raw.copy()
    normalized.columns = [field.name for field in field_columns]
    return normalized
