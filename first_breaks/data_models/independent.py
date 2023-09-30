from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

TColor = Union[Tuple[int, int, int, int], Tuple[int, int, int]]
TNormalize = Union[Literal["trace", "gather"], float, int, np.ndarray, None]


class DefaultModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


class TracesPerGather(DefaultModel):
    traces_per_gather: int = Field(12, ge=1, description="How we split the sequence of traces into individual gathers")


class MaximumTime(DefaultModel):
    maximum_time: Union[int, float] = Field(
        0, ge=0.0, description="Limits the window for processing along the time axis"
    )


class TracesToInverse(DefaultModel):
    traces_to_inverse: Sequence[int] = Field((), description="Inverse traces amplitudes on the gathers level")

    @field_validator("traces_to_inverse")
    def validate(cls, traces_to_inverse: Sequence[int]) -> Sequence[int]:
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
                raise ValueError(f"f2 should be greater than f1")
        return v


class F3F4(DefaultModel):
    f3_f4: Optional[Tuple[float, float]] = Field(None, description="Frequency pair for decreasing part band filter")

    @field_validator("f3_f4")
    def validate_encoding(cls, v: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if v is not None:
            if v[0] >= v[1]:
                raise ValueError(f"f4 should be greater than f3")
        return v


class FillBlack(DefaultModel):
    fill_black: Optional[Literal["left", "right"]] = Field(
        "left",
        description="Where and how to fill wiggles with black",
    )


class PicksColor(DefaultModel):
    picks_color: TColor = Field((255, 0, 0), description="Color for picks")


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


class PicksUnit(DefaultModel):
    picks_unit: Literal["ms", "mcs", "sample"] = Field("mcs", description="First breaks / picks unit")


class PicksInSamplesOptional(DefaultModel):
    picks_in_samples: Optional[Union[np.ndarray, Sequence[Union[int]]]] = Field(
        None, description="First breaks presented as samples"
    )


class ConfidenceOptional(DefaultModel):
    confidence: Optional[Union[np.ndarray, Sequence[Union[int, float]]]] = Field(
        None, description="Confidence of first breaks"
    )


class ModelHashOptional(DefaultModel):
    model_hash: Optional[str] = Field(None, description="Hash of model checkpoint")


class PicksValue(DefaultModel):
    picks_value: Union[np.ndarray, List[Union[int, float]], Tuple[Union[int, float], ...]] = Field(
        ...,
        description="Values of first breaks",
    )


class VSPView(DefaultModel):
    vsp_view: bool = Field(
        False, description="Set the view when the vertical axis is the trace number and the horizontal axis is time"
    )


class InvertX(DefaultModel):
    invert_x: bool = Field(False, description="If True, the X-axis values will increase from right to the left")


class InvertY(DefaultModel):
    invert_y: bool = Field(True, description="If True, the Y-axis values will increase from top to bottom")
