import warnings
from typing import List, Optional, Tuple

from pydantic import Field, model_validator

from first_breaks.data_models.dependent import SGYModel
from first_breaks.data_models.independent import (
    DefaultModel,
    ExceptionOptional,
    ModelHashOptional,
    PicksID,
)
from first_breaks.picking.picks import PickingParameters, Picks
from first_breaks.utils.utils import chunk_iterable

MINIMUM_TRACES_PER_GATHER = 2


class ProcessingParametersException(Exception):
    pass


class Success(DefaultModel):
    success: Optional[bool] = None


class ErrorMessage(DefaultModel):
    error_message: Optional[str] = None


class Task(
    SGYModel,
    PickingParameters,
    ErrorMessage,
    Success,
    ModelHashOptional,
    ExceptionOptional,
    PicksID,
):
    picks: Optional[Picks] = Field(None, description="Result of picking process")

    @property
    def picking_parameters(self) -> PickingParameters:
        return PickingParameters(**self.model_dump())

    @property
    def maximum_time_sample(self) -> int:
        return self.sgy.ms2index(self.maximum_time)

    @model_validator(mode="after")  # type: ignore
    def align_parameters(self) -> "Task":
        prev_assignment = self.model_config.get("validate_assignment", None)
        self.model_config["validate_assignment"] = False

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
        return self  # type: ignore

    def get_gathers_ids(self) -> List[Tuple[int]]:
        return list(chunk_iterable(list(range(self.sgy.shape[1])), self.traces_per_gather))  # type: ignore

    @property
    def num_gathers(self) -> int:
        return len(self.get_gathers_ids())

    def get_result(self) -> Picks:
        if self.picks is not None:
            return self.picks
        else:
            raise ValueError("Picks are not calculated")
