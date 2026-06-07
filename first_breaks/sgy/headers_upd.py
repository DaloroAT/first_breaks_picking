from __future__ import annotations

import io
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, TypeVar, IO

import numpy as np
import pandas as pd

from first_breaks.sgy.types import SGYLayout


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


def get_num_bytes(fmt: str) -> int:
    tp = fmt[-1]
    if tp not in FORMAT_TO_SIZE:
        raise InvalidHeaders("Format is not interpretable")
    num_str = fmt[:-1]
    num = int(num_str) if num_str.isdigit() else 1
    return FORMAT_TO_SIZE[tp] * num


class FileHeaders:
    def __init__(self, headers: Dict[str | FileHeaderField, Any]):
        input_keys = set(headers.keys())
        required_keys = set(list(FileHeaderField))
        redundant_keys = input_keys - required_keys
        missed_keys = required_keys - input_keys
        if redundant_keys:
            raise KeyError(f"Redundant header keys: {redundant_keys}")
        if missed_keys:
            raise KeyError(f"Missed header keys: {missed_keys}")

        self.__headers = {FileHeaderField(k): v for k, v in headers.items()}

    @classmethod
    def read_from_sgy_pointer(cls, pointer: IO) -> "FileHeaders":
        raise NotImplementedError

    def write_with_sgy_pointer(self, pointer: IO) -> None:
        raise NotImplementedError

    @property
    def headers(self) -> Dict[FileHeaderField, Any]:
        return self.__headers.copy()

    def __getitem__(self, key: str | FileHeaderField) -> Any:
        key = FileHeaderField(key)
        return deepcopy(self.__headers[key])


class TracesHeaders:
    def __init__(self, headers_raw: pd.DataFrame) -> None:
        input_keys = set(headers_raw.columns)
        required_keys = set(list(TraceHeaderField))
        redundant_keys = input_keys - required_keys
        missed_keys = required_keys - input_keys
        if redundant_keys:
            raise KeyError(f"Redundant header keys: {redundant_keys}")
        if missed_keys:
            raise KeyError(f"Missed header keys: {missed_keys}")

        self.__headers_raw = headers_raw  # replace columns?

    @staticmethod
    def __scale(headers_raw: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def __unscale(headers_scaled: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def read_from_sgy_pointer(cls, pointer: IO) -> "FileHeaders":
        raise NotImplementedError

    def write_with_sgy_pointer(self, pointer: IO) -> None:
        raise NotImplementedError

    @property
    def headers_raw(self) -> pd.DataFrame:
        # give copy
        raise NotImplementedError

    @property
    def headers_scaled(self) -> pd.DataFrame:
        # give copy
        raise NotImplementedError

    def __getitem__(self, key: str | TraceHeaderField) -> Any:
        raise NotImplementedError


def read_file_header(pointer: IO, info: HeaderInfo) -> Any:
    raise NotImplementedError


def write_file_header(pointer: IO, info: HeaderInfo, value: Any) -> None:
    raise NotImplementedError


def read_trace_header(pointer: IO, info: HeaderInfo, trace_idx: int, layout: SGYLayout) -> Any:
    raise NotImplementedError


def write_trace_header(pointer: IO, info: HeaderInfo, value: Any, trace_idx: int, layout: SGYLayout) -> None:
    raise NotImplementedError

