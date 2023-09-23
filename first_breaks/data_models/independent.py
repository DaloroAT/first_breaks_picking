from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

TColor = Union[Tuple[int, int, int, int], Tuple[int, int, int]]
TNormalize = Union[Literal["trace", "gather"], float, int, np.ndarray, None]


class DefaultModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


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


class F3F4(DefaultModel):
    f3_f4: Optional[Tuple[float, float]] = Field(None, description="Frequency pair for decreasing part band filter")


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


class PicksValue(DefaultModel):
    picks_value: Union[np.ndarray, List[Union[int, float]], Tuple[Union[int, float], ...]] = Field(
        ...,
        description="Values of first breaks",
    )


class VSPView(DefaultModel):
    vsp_view: bool = Field(
        False,
        description="Set the view when the vertical axis is the trace number and the horizontal axis is time"
    )
