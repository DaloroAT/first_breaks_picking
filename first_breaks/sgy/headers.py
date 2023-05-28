from typing import Dict, List, Tuple, Any


class InvalidHeaders(Exception):
    pass


THeadersAttr = List[Tuple[int, str, str]]


class Headers:
    # headers in format [(offset_1, header_name_1, format_1), (offset_2, header_name_2, format_2), ...]
    # offset started from 0
    headers_schema: THeadersAttr
    format2size: Dict[str, int] = {
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

    def get_num_bytes(self, fmt: str) -> int:
        tp = fmt[-1]
        if tp not in self.format2size:
            raise InvalidHeaders("Format is not interpretable")
        num = fmt[:-1]
        if num.isdigit():
            num = int(num)  # type: ignore
        else:
            num = 1  # type: ignore
        return self.format2size[tp] * num  # type: ignore

    def validate(self) -> None:
        headers_names = [header_info[1] for header_info in self.headers_schema]
        if not all(isinstance(name, str) for name in headers_names):
            raise InvalidHeaders("Header names must be strings")
        if not len(set(headers_names)) == len(headers_names):
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

    def get_template(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class FileHeaders(Headers):
    dt_name = "dt"
    dt_name_orig = "dt_orig"
    ns_name = "ns"
    ns_name_orig = "ns_orig"
    data_sample_format_name = "data_sample_format"

    def __init__(self) -> None:
        self.headers_schema = [
            (0, "textual_file_header", "3200s"),
            (3200, "job", "i"),
            (3204, "line", "i"),
            (3208, "reel", "i"),
            (3212, "data_trace_per_ensemble", "H"),
            (3214, "auxiliary_trace_per_ensemble", "H"),
            (3216, self.dt_name, "H"),
            (3218, self.dt_name_orig, "H"),
            (3220, self.ns_name, "H"),
            (3222, self.ns_name_orig, "H"),
            (3224, self.data_sample_format_name, "H"),
            (3226, "ensemble_fold", "H"),
            (3228, "trace_sorting", "h"),
            (3230, "vertical_sum_code", "H"),
            (3232, "sweep_frequency_start", "H"),
            (3234, "sweep_frequency_end", "H"),
            (3236, "sweep_length", "H"),
            (3238, "sweep_type", "h"),
            (3240, "sweep_channel", "h"),
            (3242, "sweep_taper_length_start", "H"),
            (3244, "sweep_taper_length_end", "H"),
            (3246, "taper_type", "H"),
            (3248, "correlated_data_traces", "H"),
            (3250, "binary_gain", "H"),
            (3252, "amplitude_recovery_method", "H"),
            (3254, "measurement_system", "H"),
            (3256, "impulse_signal_polarity", "H"),
            (3258, "vibratory_polarity_code", "H"),
            (3260, "unassigned1", "240s"),
            (3500, "segy_format_revision_number", "H"),
            (3502, "fixed_length_trace_flag", "H"),
            (3504, "number_of_textual_headers", "H"),
            (3506, "unassigned2", "94s"),
        ]
        self.validate()


class TraceHeaders(Headers):
    def __init__(self) -> None:
        self.headers_schema = [
            (0, "TRACENO", "i"),
            (4, "trace_sequence_file", "i"),
            (8, "FFID", "i"),
            (12, "CHAN", "i"),
            (16, "SOURCE", "i"),
            (20, "CDP", "i"),
            (24, "SEQNO", "i"),
            (28, "TRC_TYPE", "h"),
            (30, "STACKNT", "h"),
            (32, "TRFOLD", "h"),
            (34, "data_use", "h"),
            (36, "OFFSET", "i"),
            (40, "REC_ELEV", "i"),
            (44, "SOU_ELEV", "i"),
            (48, "DEPTH", "i"),
            (52, "REC_DATUM", "i"),
            (56, "SOU_DATUM", "i"),
            (60, "SOU_H2OD", "i"),
            (64, "REC_H2OD", "i"),
            (68, "elevation_scalar", "h"),
            (70, "source_group_scalar", "h"),
            (72, "SOU_X", "i"),
            (76, "SOU_Y", "i"),
            (80, "REC_X", "i"),
            (84, "REC_Y", "i"),
            (88, "coordinate_units", "h"),
            (90, "weathering_velocity", "h"),
            (92, "subweathering_velocity", "h"),
            (94, "UPHOLE", "h"),
            (96, "REC_UPHOLE", "h"),
            (98, "SOU_STAT", "h"),
            (100, "REC_STAT", "h"),
            (102, "TOT_STAT", "h"),
            (104, "lag_time_a", "h"),
            (106, "lag_time_b", "h"),
            (108, "delay_recording_time", "h"),
            (110, "TLIVE_S", "h"),
            (112, "TFULL_S", "h"),
            (114, "NUMSMP", "H"),
            (116, "DT", "H"),
            (118, "IGAIN", "h"),
            (120, "PREAMP", "h"),
            (122, "EARLYG", "h"),
            (124, "COR_FLAG", "h"),
            (126, "SWEEPFREQSTART", "h"),
            (128, "SWEEPFREQEND", "h"),
            (130, "SWEEPLEN", "h"),
            (132, "SWEEPTYPE", "h"),
            (134, "SWEEPTAPSTART", "h"),
            (136, "SWEEPTAPEND", "h"),
            (138, "SWEEPTAPCODE", "h"),
            (140, "AAXFILT", "h"),
            (142, "AAXSLOP", "h"),
            (144, "FREQXN", "h"),
            (146, "FXNSLOP", "h"),
            (148, "FREQXL", "h"),
            (150, "FREQXH", "h"),
            (152, "FXLSLOP", "h"),
            (154, "FXHSLOP", "h"),
            (156, "YEAR", "h"),
            (158, "DAY", "h"),
            (160, "HOUR", "h"),
            (162, "MINUTE", "h"),
            (164, "SECOND", "h"),
            (166, "time_basic_code", "h"),
            (168, "trace_weighting_factor", "h"),
            (170, "geophone_group_number_roll1", "h"),
            (172, "geophone_group_number_first", "h"),
            (174, "geophone_group_number_last", "h"),
            (176, "gap_size", "h"),
            (178, "over_travel", "h"),
            (180, "CDP_X", "i"),
            (184, "CDP_Y", "i"),
            (188, "ILINE_NO", "i"),
            (192, "XLINE_NO", "i"),
            (196, "shot_point", "i"),
            (200, "shot_point_scalar", "h"),
            (202, "trace_value_measurement", "h"),
            (204, "transduction_constant_mantissa", "i"),
            (208, "transduction_constant_power", "h"),
            (210, "transduction_unit", "h"),
            (212, "trace_identifier", "h"),
            (214, "scalar_trace_header", "h"),
            (216, "source_type", "h"),
            (218, "source_energy_direction_mantissa", "i"),
            (222, "source_energy_direction_exponent", "H"),
            (224, "source_measurement_mantissa", "i"),
            (228, "source_measurement_exponent", "H"),
            (230, "source_measurement_unit", "h"),
            (232, "unassigned1", "i"),
            (236, "FB_PICK", "i"),
        ]
        self.validate()
