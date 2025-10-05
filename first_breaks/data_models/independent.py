import traceback
from typing import Literal, Optional, Sequence, Tuple, Union
from uuid import uuid4

import numpy as np
from pydantic import UUID4, BaseModel, Field, field_validator

TColor = Union[Tuple[int, int, int], Tuple[int, int, int, int], Tuple[int, ...]]
TNormalize = Union[Literal["trace", "gather"], float, int, np.ndarray, None]


class DefaultModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        protected_namespaces = ()


class TracesPerGather(DefaultModel):
    traces_per_gather: int = Field(12, ge=1, description="How we split the sequence of traces into individual gathers")


class MaximumTime(DefaultModel):
    maximum_time: Union[int, float] = Field(
        0, ge=0.0, description="Limits the window for processing along the time axis"
    )


class TracesToInverse(DefaultModel):
    traces_to_inverse: Sequence[int] = Field((), description="Inverse traces amplitudes on the gathers level")

    @field_validator("traces_to_inverse")
    def validate_traces_to_inverse(cls, traces_to_inverse: Sequence[int]) -> Sequence[int]:
        if not all(val >= 0 for val in traces_to_inverse):
            raise ValueError("Elements of `traces_to_inverse` must be greater or equal to 0")
        return traces_to_inverse


class Normalize(DefaultModel):
    normalize: TNormalize = Field(
        "trace",
        description="How to normalize gather",
    )


class Gain(DefaultModel):
    gain: float = Field(1.0, description="Gain value")


class Clip(DefaultModel):
    clip: float = Field(
        0.9,
        gt=0.0,
        description="Clip amplitudes to this value if their absolute value is greater than this number",
    )


class F1F2(DefaultModel):
    f1_f2: Optional[Tuple[float, float]] = Field(None, description="Frequency pair for growing part band filter")

    @field_validator("f1_f2")
    def validate_encoding(cls, v: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if v is not None:
            if v[0] >= v[1]:
                raise ValueError("f2 should be greater than f1")
        return v


class F3F4(DefaultModel):
    f3_f4: Optional[Tuple[float, float]] = Field(None, description="Frequency pair for decreasing part band filter")

    @field_validator("f3_f4")
    def validate_encoding(cls, v: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if v is not None:
            if v[0] >= v[1]:
                raise ValueError("f4 should be greater than f3")
        return v


class FillBlack(DefaultModel):
    fill_black: Optional[Literal["left", "right"]] = Field(
        "left",
        description="Where and how to fill wiggles with black",
    )


class RegionPolyColor(DefaultModel):
    region_poly_color: TColor = Field((100, 100, 100, 50), description="Color of region below maximum time")


class RegionContourColor(DefaultModel):
    region_contour_color: TColor = Field(
        (100, 100, 100),
        description="Color of dashed line which shows maximum time and how file is split into gathers",
    )


class RegionContourWidth(DefaultModel):
    region_contour_width: float = Field(
        0.2,
        description="Width of dashed line which shows maximum time and how file is split into gathers",
    )


class TraceBytePosition(DefaultModel):
    byte_position: int = Field(0, ge=0, lt=240, description="Byte index in trace headers")


class ModelHashOptional(DefaultModel):
    model_hash: Optional[str] = Field(None, description="Hash of model checkpoint")


class VSPView(DefaultModel):
    vsp_view: bool = Field(
        False, description="Set the view when the vertical axis is the trace number and the horizontal axis is time"
    )


class InvertX(DefaultModel):
    invert_x: bool = Field(False, description="If True, the X-axis values will increase from right to the left")


class InvertY(DefaultModel):
    invert_y: bool = Field(True, description="If True, the Y-axis values will increase from top to bottom")


class BatchSize(DefaultModel):
    batch_size: int = Field(1, ge=1, description="Batch size")


class ExceptionOptional(DefaultModel):
    exception: Optional[Exception] = None

    def get_formatted_traceback(self) -> str:
        formatted_traceback = traceback.format_exception(
            type(self.exception), self.exception, self.exception.__traceback__
        )
        return "".join(formatted_traceback)


class PicksID(DefaultModel):
    picks_id: Optional[UUID4] = Field(None, description="ID for picks")

    def assign_new_picks_id(self) -> None:
        self.picks_id = uuid4()
