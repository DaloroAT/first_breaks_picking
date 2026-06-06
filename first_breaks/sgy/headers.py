from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple


class InvalidHeaders(Exception):
    pass


class NamingStyle(Enum):
    """Header naming styles used by different seismic software.

    Currently supported:
    - RADEXPRO: RadexPro-style header names (default)

    Future additions may include:
    - SEISEE: SeiSee-style header names
    - SEISMICUNIX: Seismic Unix-style header names
    """

    RADEXPRO = "radexpro"


THeadersAttr = List[Tuple[int, str, str]]


class HeaderInfo(NamedTuple):
    """Header definition with byte offset and struct format."""

    offset: int
    format: str


class FileHeaderEnum(Enum):
    """Enumeration of SEG-Y binary file header fields."""

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


class TraceHeaderEnum(Enum):
    """Enumeration of SEG-Y trace header fields."""

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
    """Calculate number of bytes for a struct format string."""
    tp = fmt[-1]
    if tp not in FORMAT_TO_SIZE:
        raise InvalidHeaders("Format is not interpretable")
    num_str = fmt[:-1]
    num = int(num_str) if num_str.isdigit() else 1
    return FORMAT_TO_SIZE[tp] * num


class Headers:
    """Base class for header schemas with backward compatibility."""

    headers_schema: THeadersAttr
    format2size: Dict[str, int] = FORMAT_TO_SIZE

    def get_num_bytes(self, fmt: str) -> int:
        return get_num_bytes(fmt)

    def validate(self) -> None:
        headers_names = [header_info[1] for header_info in self.headers_schema]
        if not all(isinstance(name, str) for name in headers_names):
            raise InvalidHeaders("Header names must be strings")
        if len(set(headers_names)) != len(headers_names):
            raise InvalidHeaders("Header names must be unique")
        available_formats = self.format2size.keys()
        if not all(header_info[2][-1] in available_formats for header_info in self.headers_schema):
            raise InvalidHeaders("Some of headers have unavailable formats")

    def fill_offsets_if_empty(self) -> THeadersAttr:
        offsets_is_none = [header_info[0] is None for header_info in self.headers_schema]
        if all(offsets_is_none):
            upd_headers = []
            offset = 0
            for header in self.headers_schema:
                upd_header = (offset, header[1], header[2])
                upd_headers.append(upd_header)
                offset += self.get_num_bytes(header[2])
            return upd_headers
        elif any(offsets_is_none) and not all(offsets_is_none):
            raise InvalidHeaders("To fill offsets, they must all be either empty or filled (no modification)")
        return self.headers_schema


class FileHeaders(Headers):
    """File header schema with backward compatible interface."""

    # Name mappings for different naming styles
    _STYLE_TO_NAMES: Dict[NamingStyle, Dict[FileHeaderEnum, str]] = {
        NamingStyle.RADEXPRO: {
            FileHeaderEnum.TEXTUAL_FILE_HEADER: "textual_file_header",
            FileHeaderEnum.JOB: "job",
            FileHeaderEnum.LINE: "line",
            FileHeaderEnum.REEL: "reel",
            FileHeaderEnum.DATA_TRACE_PER_ENSEMBLE: "data_trace_per_ensemble",
            FileHeaderEnum.AUXILIARY_TRACE_PER_ENSEMBLE: "auxiliary_trace_per_ensemble",
            FileHeaderEnum.DT: "dt",
            FileHeaderEnum.DT_ORIG: "dt_orig",
            FileHeaderEnum.NS: "ns",
            FileHeaderEnum.NS_ORIG: "ns_orig",
            FileHeaderEnum.DATA_SAMPLE_FORMAT: "data_sample_format",
            FileHeaderEnum.ENSEMBLE_FOLD: "ensemble_fold",
            FileHeaderEnum.TRACE_SORTING: "trace_sorting",
            FileHeaderEnum.VERTICAL_SUM_CODE: "vertical_sum_code",
            FileHeaderEnum.SWEEP_FREQUENCY_START: "sweep_frequency_start",
            FileHeaderEnum.SWEEP_FREQUENCY_END: "sweep_frequency_end",
            FileHeaderEnum.SWEEP_LENGTH: "sweep_length",
            FileHeaderEnum.SWEEP_TYPE: "sweep_type",
            FileHeaderEnum.SWEEP_CHANNEL: "sweep_channel",
            FileHeaderEnum.SWEEP_TAPER_LENGTH_START: "sweep_taper_length_start",
            FileHeaderEnum.SWEEP_TAPER_LENGTH_END: "sweep_taper_length_end",
            FileHeaderEnum.TAPER_TYPE: "taper_type",
            FileHeaderEnum.CORRELATED_DATA_TRACES: "correlated_data_traces",
            FileHeaderEnum.BINARY_GAIN: "binary_gain",
            FileHeaderEnum.AMPLITUDE_RECOVERY_METHOD: "amplitude_recovery_method",
            FileHeaderEnum.MEASUREMENT_SYSTEM: "measurement_system",
            FileHeaderEnum.IMPULSE_SIGNAL_POLARITY: "impulse_signal_polarity",
            FileHeaderEnum.VIBRATORY_POLARITY_CODE: "vibratory_polarity_code",
            FileHeaderEnum.UNASSIGNED1: "unassigned1",
            FileHeaderEnum.SEGY_FORMAT_REVISION_NUMBER: "segy_format_revision_number",
            FileHeaderEnum.FIXED_LENGTH_TRACE_FLAG: "fixed_length_trace_flag",
            FileHeaderEnum.NUMBER_OF_TEXTUAL_HEADERS: "number_of_textual_headers",
            FileHeaderEnum.UNASSIGNED2: "unassigned2",
        },
    }

    # Default style for backward compatibility
    _ENUM_TO_NAME: Dict[FileHeaderEnum, str] = _STYLE_TO_NAMES[NamingStyle.RADEXPRO]

    def __init__(self, naming_style: NamingStyle = NamingStyle.RADEXPRO) -> None:
        self.naming_style = naming_style
        self._ENUM_TO_NAME = self._STYLE_TO_NAMES[naming_style]

        # Set attribute name accessors based on naming style
        self.dt_name = self._ENUM_TO_NAME[FileHeaderEnum.DT]
        self.dt_name_orig = self._ENUM_TO_NAME[FileHeaderEnum.DT_ORIG]
        self.ns_name = self._ENUM_TO_NAME[FileHeaderEnum.NS]
        self.ns_name_orig = self._ENUM_TO_NAME[FileHeaderEnum.NS_ORIG]
        self.data_sample_format_name = self._ENUM_TO_NAME[FileHeaderEnum.DATA_SAMPLE_FORMAT]

        self.headers_schema = [
            (h.value.offset, self._ENUM_TO_NAME[h], h.value.format) for h in FileHeaderEnum
        ]
        self.validate()


class TraceHeaders(Headers):
    """Trace header schema with backward compatible interface."""

    # Name mappings for different naming styles
    _STYLE_TO_NAMES: Dict[NamingStyle, Dict[TraceHeaderEnum, str]] = {
        NamingStyle.RADEXPRO: {
            TraceHeaderEnum.TRACENO: "TRACENO",
            TraceHeaderEnum.TRACE_SEQUENCE_FILE: "trace_sequence_file",
            TraceHeaderEnum.FFID: "FFID",
            TraceHeaderEnum.CHAN: "CHAN",
            TraceHeaderEnum.SOURCE: "SOURCE",
            TraceHeaderEnum.CDP: "CDP",
            TraceHeaderEnum.SEQNO: "SEQNO",
            TraceHeaderEnum.TRC_TYPE: "TRC_TYPE",
            TraceHeaderEnum.STACKNT: "STACKNT",
            TraceHeaderEnum.TRFOLD: "TRFOLD",
            TraceHeaderEnum.DATA_USE: "data_use",
            TraceHeaderEnum.OFFSET: "OFFSET",
            TraceHeaderEnum.REC_ELEV: "REC_ELEV",
            TraceHeaderEnum.SOU_ELEV: "SOU_ELEV",
            TraceHeaderEnum.DEPTH: "DEPTH",
            TraceHeaderEnum.REC_DATUM: "REC_DATUM",
            TraceHeaderEnum.SOU_DATUM: "SOU_DATUM",
            TraceHeaderEnum.SOU_H2OD: "SOU_H2OD",
            TraceHeaderEnum.REC_H2OD: "REC_H2OD",
            TraceHeaderEnum.ELEVATION_SCALAR: "elevation_scalar",
            TraceHeaderEnum.SOURCE_GROUP_SCALAR: "source_group_scalar",
            TraceHeaderEnum.SOU_X: "SOU_X",
            TraceHeaderEnum.SOU_Y: "SOU_Y",
            TraceHeaderEnum.REC_X: "REC_X",
            TraceHeaderEnum.REC_Y: "REC_Y",
            TraceHeaderEnum.COORDINATE_UNITS: "coordinate_units",
            TraceHeaderEnum.WEATHERING_VELOCITY: "weathering_velocity",
            TraceHeaderEnum.SUBWEATHERING_VELOCITY: "subweathering_velocity",
            TraceHeaderEnum.UPHOLE: "UPHOLE",
            TraceHeaderEnum.REC_UPHOLE: "REC_UPHOLE",
            TraceHeaderEnum.SOU_STAT: "SOU_STAT",
            TraceHeaderEnum.REC_STAT: "REC_STAT",
            TraceHeaderEnum.TOT_STAT: "TOT_STAT",
            TraceHeaderEnum.LAG_TIME_A: "lag_time_a",
            TraceHeaderEnum.LAG_TIME_B: "lag_time_b",
            TraceHeaderEnum.DELAY_RECORDING_TIME: "delay_recording_time",
            TraceHeaderEnum.TLIVE_S: "TLIVE_S",
            TraceHeaderEnum.TFULL_S: "TFULL_S",
            TraceHeaderEnum.NUMSMP: "NUMSMP",
            TraceHeaderEnum.DT: "DT",
            TraceHeaderEnum.IGAIN: "IGAIN",
            TraceHeaderEnum.PREAMP: "PREAMP",
            TraceHeaderEnum.EARLYG: "EARLYG",
            TraceHeaderEnum.COR_FLAG: "COR_FLAG",
            TraceHeaderEnum.SWEEPFREQSTART: "SWEEPFREQSTART",
            TraceHeaderEnum.SWEEPFREQEND: "SWEEPFREQEND",
            TraceHeaderEnum.SWEEPLEN: "SWEEPLEN",
            TraceHeaderEnum.SWEEPTYPE: "SWEEPTYPE",
            TraceHeaderEnum.SWEEPTAPSTART: "SWEEPTAPSTART",
            TraceHeaderEnum.SWEEPTAPEND: "SWEEPTAPEND",
            TraceHeaderEnum.SWEEPTAPCODE: "SWEEPTAPCODE",
            TraceHeaderEnum.AAXFILT: "AAXFILT",
            TraceHeaderEnum.AAXSLOP: "AAXSLOP",
            TraceHeaderEnum.FREQXN: "FREQXN",
            TraceHeaderEnum.FXNSLOP: "FXNSLOP",
            TraceHeaderEnum.FREQXL: "FREQXL",
            TraceHeaderEnum.FREQXH: "FREQXH",
            TraceHeaderEnum.FXLSLOP: "FXLSLOP",
            TraceHeaderEnum.FXHSLOP: "FXHSLOP",
            TraceHeaderEnum.YEAR: "YEAR",
            TraceHeaderEnum.DAY: "DAY",
            TraceHeaderEnum.HOUR: "HOUR",
            TraceHeaderEnum.MINUTE: "MINUTE",
            TraceHeaderEnum.SECOND: "SECOND",
            TraceHeaderEnum.TIME_BASIC_CODE: "time_basic_code",
            TraceHeaderEnum.TRACE_WEIGHTING_FACTOR: "trace_weighting_factor",
            TraceHeaderEnum.GEOPHONE_GROUP_NUMBER_ROLL1: "geophone_group_number_roll1",
            TraceHeaderEnum.GEOPHONE_GROUP_NUMBER_FIRST: "geophone_group_number_first",
            TraceHeaderEnum.GEOPHONE_GROUP_NUMBER_LAST: "geophone_group_number_last",
            TraceHeaderEnum.GAP_SIZE: "gap_size",
            TraceHeaderEnum.OVER_TRAVEL: "over_travel",
            TraceHeaderEnum.CDP_X: "CDP_X",
            TraceHeaderEnum.CDP_Y: "CDP_Y",
            TraceHeaderEnum.ILINE_NO: "ILINE_NO",
            TraceHeaderEnum.XLINE_NO: "XLINE_NO",
            TraceHeaderEnum.SHOT_POINT: "shot_point",
            TraceHeaderEnum.SHOT_POINT_SCALAR: "shot_point_scalar",
            TraceHeaderEnum.TRACE_VALUE_MEASUREMENT: "trace_value_measurement",
            TraceHeaderEnum.TRANSDUCTION_CONSTANT_MANTISSA: "transduction_constant_mantissa",
            TraceHeaderEnum.TRANSDUCTION_CONSTANT_POWER: "transduction_constant_power",
            TraceHeaderEnum.TRANSDUCTION_UNIT: "transduction_unit",
            TraceHeaderEnum.TRACE_IDENTIFIER: "trace_identifier",
            TraceHeaderEnum.SCALAR_TRACE_HEADER: "scalar_trace_header",
            TraceHeaderEnum.SOURCE_TYPE: "source_type",
            TraceHeaderEnum.SOURCE_ENERGY_DIRECTION_MANTISSA: "source_energy_direction_mantissa",
            TraceHeaderEnum.SOURCE_ENERGY_DIRECTION_EXPONENT: "source_energy_direction_exponent",
            TraceHeaderEnum.SOURCE_MEASUREMENT_MANTISSA: "source_measurement_mantissa",
            TraceHeaderEnum.SOURCE_MEASUREMENT_EXPONENT: "source_measurement_exponent",
            TraceHeaderEnum.SOURCE_MEASUREMENT_UNIT: "source_measurement_unit",
            TraceHeaderEnum.UNASSIGNED1: "unassigned1",
            TraceHeaderEnum.FB_PICK: "FB_PICK",
        },
    }

    # Default style for backward compatibility
    _ENUM_TO_NAME: Dict[TraceHeaderEnum, str] = _STYLE_TO_NAMES[NamingStyle.RADEXPRO]

    def __init__(self, naming_style: NamingStyle = NamingStyle.RADEXPRO) -> None:
        self.naming_style = naming_style
        self._ENUM_TO_NAME = self._STYLE_TO_NAMES[naming_style]

        # Set attribute name accessor based on naming style
        self.fb_pick_default = self._ENUM_TO_NAME[TraceHeaderEnum.FB_PICK]

        self.headers_schema = [
            (h.value.offset, self._ENUM_TO_NAME[h], h.value.format) for h in TraceHeaderEnum
        ]
        self.validate()

        # Build scalar application map using current naming style
        n = self._ENUM_TO_NAME  # Short alias for readability
        self.scalar_from2apply: Dict[str, List[str]] = {
            n[TraceHeaderEnum.ELEVATION_SCALAR]: [
                n[TraceHeaderEnum.REC_ELEV],
                n[TraceHeaderEnum.SOU_ELEV],
                n[TraceHeaderEnum.DEPTH],
                n[TraceHeaderEnum.REC_DATUM],
                n[TraceHeaderEnum.SOU_DATUM],
                n[TraceHeaderEnum.SOU_H2OD],
                n[TraceHeaderEnum.REC_H2OD],
            ],
            n[TraceHeaderEnum.SOURCE_GROUP_SCALAR]: [
                n[TraceHeaderEnum.SOU_X],
                n[TraceHeaderEnum.SOU_Y],
                n[TraceHeaderEnum.REC_X],
                n[TraceHeaderEnum.REC_Y],
            ],
            n[TraceHeaderEnum.SHOT_POINT_SCALAR]: [
                n[TraceHeaderEnum.SHOT_POINT],
            ],
            n[TraceHeaderEnum.SCALAR_TRACE_HEADER]: [
                n[TraceHeaderEnum.UPHOLE],
                n[TraceHeaderEnum.REC_UPHOLE],
                n[TraceHeaderEnum.SOU_STAT],
                n[TraceHeaderEnum.REC_STAT],
                n[TraceHeaderEnum.TOT_STAT],
                n[TraceHeaderEnum.LAG_TIME_A],
                n[TraceHeaderEnum.LAG_TIME_B],
                n[TraceHeaderEnum.DELAY_RECORDING_TIME],
                n[TraceHeaderEnum.TLIVE_S],
                n[TraceHeaderEnum.TFULL_S],
            ],
        }
