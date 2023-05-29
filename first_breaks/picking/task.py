from pathlib import Path
from typing import Union, Sequence, List, Tuple, Optional

from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import chunk_iterable, sample2ms

MINIMUM_TRACES_PER_GATHER = 2


class ProcessingParametersException(Exception):
    pass


class Task:
    def __init__(self,
                 sgy: Union[SGY, str, Path, bytes],
                 traces_per_gather: int = 24,
                 maximum_time: float = 0.0,
                 traces_to_inverse: Sequence[int] = (),
                 gain: float = 1,
                 clip: float = 1):
        self.sgy = sgy if isinstance(sgy, SGY) else SGY(sgy)

        self.traces_per_gather = traces_per_gather
        self.maximum_time = maximum_time
        self.traces_to_inverse = traces_to_inverse
        self.gain = gain
        self.clip = clip

        self.traces_per_gather_parsed = self.validate_and_parse_traces_per_gather(traces_per_gather)
        self.maximum_time_parsed = self.validate_and_parse_maximum_time(maximum_time)
        self.maximum_time_sample = self._convert_maximum_time_to_index()
        self.traces_to_inverse_parsed = self.validate_and_parse_traces_to_inverse(traces_to_inverse)
        self.gain_parsed = self.validate_and_parse_gain(gain)
        self.clip_parsed = self.validate_and_parse_clip(clip)

        self._picks_in_ms = None
        self.picks_in_samples = None
        self.confidence = None
        self.success = None
        self.error_message = None

    @classmethod
    def validate_traces_per_gather(cls, traces_per_gather: int):
        if not isinstance(traces_per_gather, int):
            raise ProcessingParametersException("`traces_per_gather` must be integer")
        if traces_per_gather < MINIMUM_TRACES_PER_GATHER:
            raise ProcessingParametersException(f"`traces_per_gather` must be greater or "
                                                f"equal to {MINIMUM_TRACES_PER_GATHER}")

    @classmethod
    def validate_maximum_time(cls, maximum_time: float):
        if not isinstance(maximum_time, (int, float)):
            raise ProcessingParametersException("`maximum_time` must be real")

        if maximum_time < 0.0:
            raise ProcessingParametersException("`maximum_time` must be positive or equal to 0")

    @classmethod
    def validate_traces_to_inverse(cls, traces_to_inverse: Sequence[int]):
        if not isinstance(traces_to_inverse, (tuple, list)):
            raise ProcessingParametersException("`traces_to_inverse` must be tuple or list")

        if not all(isinstance(val, int) for val in traces_to_inverse):
            raise ProcessingParametersException("Elements of `traces_to_inverse` must be int")

        if not all(val >= 1 for val in traces_to_inverse):
            raise ProcessingParametersException("Elements of `traces_to_inverse` must be greater or equal to 1")

    @classmethod
    def validate_gain(cls, gain: float):
        if not isinstance(gain, (int, float)):
            raise ProcessingParametersException("`gain` must be real")
        if gain == 0:
            raise ProcessingParametersException("`gain` must not be zero")

    @classmethod
    def validate_clip(cls, clip: float):
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
        if maximum_time > 0.0:
            maximum_time = min(maximum_time, self.sgy.shape[0] * self.sgy.dt_ms)
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

    def _convert_maximum_time_to_index(self) -> int:
        if self.maximum_time_parsed == 0.0:
            return self.sgy.shape[0]
        else:
            return int(self.maximum_time_parsed / (self.sgy.dt_ms))

    def get_gathers_ids(self) -> List[Tuple[int]]:
        return list(chunk_iterable(list(range(self.sgy.shape[1])), self.traces_per_gather_parsed))

    @property
    def num_gathers(self) -> int:
        return len(self.get_gathers_ids())

    @property
    def picks_in_ms(self) -> Optional[List[float]]:
        if self._picks_in_ms:
            return self._picks_in_ms
        elif self.picks_in_samples is not None:
            self._picks_in_ms = sample2ms(self.picks_in_samples, self.sgy.dt_ms)
            return self._picks_in_ms
        else:
            return None
