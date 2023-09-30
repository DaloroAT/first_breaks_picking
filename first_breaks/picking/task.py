import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import model_validator

from first_breaks.data_models.dependent import SGYModel
from first_breaks.data_models.independent import (
    F1F2,
    F3F4,
    Clip,
    ConfidenceOptional,
    DefaultModel,
    Gain,
    MaximumTime,
    ModelHashOptional,
    Normalize,
    PicksInSamplesOptional,
    TracesPerGather,
    TracesToInverse,
)
from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import (
    chunk_iterable,
    download_demo_sgy,
    multiply_iterable_by,
)

MINIMUM_TRACES_PER_GATHER = 2


class ProcessingParametersException(Exception):
    pass


class Success(DefaultModel):
    success: Optional[bool] = None


class ErrorMessage(DefaultModel):
    error_message: Optional[str] = None


class Task(
    SGYModel,
    TracesPerGather,
    MaximumTime,
    TracesToInverse,
    F1F2,
    F3F4,
    Gain,
    Clip,
    Normalize,
    PicksInSamplesOptional,
    ConfidenceOptional,
    ErrorMessage,
    Success,
    ModelHashOptional,
):
    @property
    def maximum_time_sample(self) -> int:
        return self.sgy.ms2index(self.maximum_time)

    @model_validator(mode="after")
    def align_parameters(self) -> "Task":
        prev_assignment = self.model_config.get("validate_assignment", None)
        self.model_config["validate_assignment"] = False

        # sgy
        self.sgy = self.sgy if isinstance(self.sgy, SGY) else SGY(self.sgy)

        # traces_per_gather
        self.traces_per_gather = min(self.sgy.num_traces, self.traces_per_gather)

        # maximum_time
        maximum_time = self.maximum_time
        if maximum_time == 0.0:
            self.maximum_time = self.sgy.max_time_ms
        else:
            index = self.sgy.ms2index(maximum_time)
            if index == 0:
                warnings.warn(
                    "The maximum time is not zero and is less than the duration of one sample, "
                    "so the maximum time will be equal to the length of the trace."
                )
                self.maximum_time = self.sgy.max_time_ms
            else:
                self.maximum_time = min(maximum_time, self.sgy.max_time_ms)

        # traces_to_inverse
        # to python indices
        traces_to_inverse = [tr - 1 for tr in self.traces_to_inverse if tr <= self.traces_per_gather]
        self.traces_to_inverse = sorted(set(traces_to_inverse))

        self.model_config["validate_assignment"] = prev_assignment
        return self

    def get_gathers_ids(self) -> List[Tuple[int]]:
        return list(chunk_iterable(list(range(self.sgy.shape[1])), self.traces_per_gather))  # type: ignore

    @property
    def num_gathers(self) -> int:
        return len(self.get_gathers_ids())

    @property
    def picks_in_ms(self) -> Optional[List[float]]:
        if self.picks_in_samples is not None:
            picks_in_samples_list = self._check_and_convert_picks()
            return multiply_iterable_by(picks_in_samples_list, self.sgy.dt_ms)  # type: ignore
        else:
            return None

    @property
    def picks_in_mcs(self) -> Optional[List[int]]:
        if self.picks_in_samples is not None:
            picks_in_samples_list = self._check_and_convert_picks()
            return multiply_iterable_by(picks_in_samples_list, self.sgy.dt_mcs, cast_to=int)  # type: ignore
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
            "traces_per_gather": self.traces_per_gather,
            "maximum_time": self.maximum_time,
            "traces_to_inverse": self.traces_to_inverse,
            "gain": self.gain,
            "clip": self.clip,
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
