from math import ceil
from pathlib import Path
from typing import Sequence, Union, Tuple, Any

import numpy as np

from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import chunk_iterable

import onnxruntime as ort


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
        self.maximum_time_index = self._convert_maximum_time_to_index()
        self.traces_to_inverse_parsed = self.validate_and_parse_traces_to_inverse(traces_to_inverse)
        self.gain_parsed = self.validate_and_parse_gain(gain)
        self.clip_parsed = self.validate_and_parse_clip(clip)

        self.picks = None
        self.is_ms = None
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
            maximum_time = min(maximum_time, self.sgy.shape[0] * self.sgy.dt * 1e-3)
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
            return int(self.maximum_time_parsed / (self.sgy.dt * 1e-3))

    @property
    def num_batches(self) -> int:
        return ceil(self.sgy.shape[1] / self.traces_per_gather_parsed)


def preprocess_gather(data: np.ndarray, gain: float = 1.0, clip: float = 1.0) -> np.ndarray:
    data = data.copy()
    norma = np.mean(np.abs(data), axis=0)
    norma[np.abs(norma) < 1e-9 * np.max(np.abs(norma))] = 1
    data = data / norma * gain
    return np.clip(data, -clip, clip)


class PickerONNX:
    def __init__(self, onnx_path: Union[str, Path]):
        self.model = ort.InferenceSession(onnx_path)

    def callback_processing_started(self) -> Any:
        pass

    def callback_processing_finished(self) -> Any:
        pass

    def callback_step_started(self, step: int) -> Any:
        pass

    def callback_step_finished(self, step: int) -> Any:
        pass

    def pick_gather(self, gather: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert gather.ndim == 2
        assert all(dim > 0 for dim in gather.shape)
        outputs = self.model.run(None, {"input": gather[None, None, ...]})
        return outputs[0][0], outputs[1][0]

    def process_task(self, task: Task, return_picks_in_ms: bool = False) -> Task:
        ntr = task.sgy.shape[1]

        task_picks = []
        task_confidence = []

        self.callback_processing_started()

        for step, gather_ids in enumerate(chunk_iterable(list(range(ntr)), task.traces_per_gather_parsed)):
            self.callback_step_started(step)

            amplitudes = np.array([-1 if idx in task.traces_to_inverse else 1 for idx in range(len(gather_ids))],
                                  dtype=np.float32)

            gather = task.sgy.read_traces_by_ids(gather_ids)

            gather = preprocess_gather(gather, task.gain, task.clip)
            gather = amplitudes[None, :] * gather
            gather = gather[:task.maximum_time_index, :]

            picks, confidence = self.pick_gather(gather)

            if return_picks_in_ms:
                picks = picks * 1e-3 * task.sgy.dt

            task_picks.extend(picks.tolist())
            task_confidence.extend(confidence.tolist())

            self.callback_step_finished(step)

        self.callback_processing_finished()

        task.is_ms = return_picks_in_ms
        task.success = True
        task.picks = task_picks
        task.confidence = task_confidence

        return task
