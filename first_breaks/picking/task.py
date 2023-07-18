import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import chunk_iterable, multiply_iterable_by

MINIMUM_TRACES_PER_GATHER = 2


class ProcessingParametersException(Exception):
    pass


class Task:
    def __init__(
        self,
        sgy: Union[SGY, str, Path, bytes],
        traces_per_gather: int = 24,
        maximum_time: float = 0.0,
        traces_to_inverse: Sequence[int] = (),
        gain: float = 1,
        clip: float = 1,
    ) -> None:
        self.sgy = sgy if isinstance(sgy, SGY) else SGY(sgy)

        self.traces_per_gather = traces_per_gather
        self.maximum_time = maximum_time
        self.traces_to_inverse = traces_to_inverse
        self.gain = gain
        self.clip = clip

        self.traces_per_gather_parsed = self.validate_and_parse_traces_per_gather(traces_per_gather)
        self.maximum_time_parsed = self.validate_and_parse_maximum_time(maximum_time)
        self.maximum_time_sample = self.sgy.ms2index(self.maximum_time_parsed)
        self.traces_to_inverse_parsed = self.validate_and_parse_traces_to_inverse(traces_to_inverse)
        self.gain_parsed = self.validate_and_parse_gain(gain)
        self.clip_parsed = self.validate_and_parse_clip(clip)

        self.picks_in_samples: Optional[Union[Sequence[float], np.ndarray]] = None
        self.confidence: Optional[Union[Sequence[float], np.ndarray]] = None
        self.success: Optional[bool] = None
        self.error_message: Optional[str] = None
        self.model_hash: Optional[str] = None

    @classmethod
    def validate_traces_per_gather(cls, traces_per_gather: int) -> None:
        if not isinstance(traces_per_gather, int):
            raise ProcessingParametersException("`traces_per_gather` must be integer")
        if traces_per_gather < MINIMUM_TRACES_PER_GATHER:
            raise ProcessingParametersException(
                f"`traces_per_gather` must be greater or " f"equal to {MINIMUM_TRACES_PER_GATHER}"
            )

    @classmethod
    def validate_maximum_time(cls, maximum_time: float) -> None:
        if not isinstance(maximum_time, (int, float)):
            raise ProcessingParametersException("`maximum_time` must be real")

        if maximum_time < 0.0:
            raise ProcessingParametersException("`maximum_time` must be positive or equal to 0")

    @classmethod
    def validate_traces_to_inverse(cls, traces_to_inverse: Sequence[int]) -> None:
        if not isinstance(traces_to_inverse, (tuple, list)):
            raise ProcessingParametersException("`traces_to_inverse` must be tuple or list")

        if not all(isinstance(val, int) for val in traces_to_inverse):
            raise ProcessingParametersException("Elements of `traces_to_inverse` must be int")

        if not all(val >= 1 for val in traces_to_inverse):
            raise ProcessingParametersException("Elements of `traces_to_inverse` must be greater or equal to 1")

    @classmethod
    def validate_gain(cls, gain: float) -> None:
        if not isinstance(gain, (int, float)):
            raise ProcessingParametersException("`gain` must be real")
        if gain == 0:
            raise ProcessingParametersException("`gain` must not be zero")

    @classmethod
    def validate_clip(cls, clip: float) -> None:
        if not isinstance(clip, (int, float)):
            raise ProcessingParametersException("`clip` must be real")
        if clip <= 0:
            raise ProcessingParametersException("`clip` must be positive")

    def validate_and_parse_traces_per_gather(self, traces_per_gather: int) -> int:
        self.validate_traces_per_gather(traces_per_gather)
        ntr = self.sgy.shape[1]
        return min(ntr, traces_per_gather)

    def validate_and_parse_maximum_time(self, maximum_time: float) -> float:
        self.validate_maximum_time(maximum_time)
        if maximum_time == 0.0:
            maximum_time = self.sgy.max_time_ms
        else:
            index = self.sgy.ms2index(maximum_time)
            if index == 0:
                warnings.warn(
                    "The maximum time is not zero and is less than the duration of one sample, "
                    "so the maximum time will be equal to the length of the trace."
                )
                maximum_time = self.sgy.max_time_ms
            else:
                maximum_time = min(maximum_time, self.sgy.max_time_ms)
        return maximum_time

    def validate_and_parse_traces_to_inverse(self, traces_to_inverse: Sequence[int]) -> Sequence[int]:
        self.validate_traces_to_inverse(traces_to_inverse)
        # to python indices
        traces_to_inverse = [tr - 1 for tr in traces_to_inverse if tr <= self.traces_per_gather_parsed]
        traces_to_inverse = sorted(set(traces_to_inverse))
        return traces_to_inverse

    @classmethod
    def validate_and_parse_gain(cls, gain: float) -> float:
        cls.validate_gain(gain)
        return float(gain)

    @classmethod
    def validate_and_parse_clip(cls, clip: float) -> float:
        cls.validate_clip(clip)
        return float(clip)

    def get_gathers_ids(self) -> List[Tuple[int]]:
        return list(chunk_iterable(list(range(self.sgy.shape[1])), self.traces_per_gather_parsed))  # type: ignore

    @property
    def num_gathers(self) -> int:
        return len(self.get_gathers_ids())

    @property
    def picks_in_ms(self) -> Optional[List[float]]:
        if self.picks_in_samples is not None:
            return multiply_iterable_by(self.picks_in_samples, self.sgy.dt_ms)  # type: ignore
        else:
            return None

    @property
    def picks_in_mcs(self) -> Optional[List[int]]:
        if self.picks_in_samples is not None:
            return multiply_iterable_by(self.picks_in_samples, self.sgy.dt_mcs, cast_to=int)  # type: ignore
        else:
            return None

    def _check_and_convert_picks(self) -> List[float]:  # type: ignore
        if self.picks_in_samples is None:
            raise ValueError("There are no picks. Put them manually or process the task first")
        if isinstance(self.picks_in_samples, (tuple, list)):
            picks_in_samples = self.picks_in_samples
        elif isinstance(self.picks_in_samples, np.ndarray):
            picks_in_samples = self.picks_in_samples.tolist()
        else:
            raise TypeError("Only 1D sequence can be saved")
        return picks_in_samples  # type: ignore

    def _prepare_output_for_nonbinary_export(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        picks_in_samples = self._check_and_convert_picks()
        picks_in_ms = self.sgy.units_converter.index2ms(picks_in_samples, cast_to=float)

        confidence = self.confidence

        is_source_file = isinstance(self.sgy.source, (str, Path))
        if is_source_file:
            source_filename = str(Path(self.sgy.source).name)  # type: ignore
            source_full_name = str(Path(self.sgy.source).resolve())  # type: ignore
        else:
            source_filename = None
            source_full_name = None

        meta = {
            "is_source_file": is_source_file,
            "is_source_ndarray": self.sgy.is_source_ndarray,
            "filename": source_filename,
            "full_name": source_full_name,
            "hash": self.sgy.get_hash(),
            "dt_ms": self.sgy.dt_ms,
            "is_picked_with_model": bool(self.success),
            "model_hash": self.model_hash,
            "traces_per_gather": self.traces_per_gather_parsed,
            "maximum_time": self.maximum_time_parsed,
            "traces_to_inverse": self.traces_to_inverse_parsed,
            "gain": self.gain_parsed,
            "clip": self.clip_parsed,
        }
        data = {
            "trace": list(range(1, len(picks_in_samples) + 1)),
            "picks_in_samples": picks_in_samples,
            "picks_in_ms": picks_in_ms,
            "confidence": confidence,
        }
        return meta, data

    def export_result_as_sgy(
        self,
        filename: Union[str, Path],
        byte_position: int = 236,
        encoding: str = "I",
        picks_unit: str = "mcs",
    ) -> None:
        picks_in_samples = self._check_and_convert_picks()
        self.sgy.export_sgy_with_picks(
            output_fname=filename,
            picks_in_samples=picks_in_samples,
            encoding=encoding,
            byte_position=byte_position,
            picks_unit=picks_unit,
        )

    def export_result_as_json(self, filename: Union[str, Path]) -> None:
        meta, data = self._prepare_output_for_nonbinary_export()
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as fout:
            json.dump({**meta, **data}, fout)

    def export_result_as_txt(self, filename: Union[str, Path]) -> None:
        meta, data = self._prepare_output_for_nonbinary_export()
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as fout:
            content = [f"{k}={v}" for k, v in meta.items()]
            data_str = pd.DataFrame(data).to_string(index=False, justify="right")
            content.append(data_str)
            content = "\n".join(content)
            fout.write(content)
