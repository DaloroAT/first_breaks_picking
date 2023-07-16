from typing import Optional, Literal, Union, Tuple

from pydantic import BaseModel, Field


TColor = Union[Tuple[int, int, int, int], Tuple[int, int, int]]


class DefaultModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class Normalize(BaseModel):
    normalize: Union[Literal["trace", "gather"], float, int, None] = Field(
        "trace",
        description="How to normalize gather",
    )


class Gain(BaseModel):
    gain: float = Field(1.0, description="Gain value")


class Clip(BaseModel):
    clip: float = Field(
        0.9,
        gt=0.0,
        description="Clip amplitudes to this value if their absolute value is greater than this number",
    )


class FillBlackLeft(BaseModel):
    fill_black_left: bool = Field(
        True,
        description="If True fill wiggles with black on the left side, otherwise on the right side",
    )


class PicksColor(BaseModel):
    picks_color: TColor = Field((255, 0, 0), description="Color for picks")


class RegionPolyColor(BaseModel):
    region_poly_color: TColor = Field((100, 100, 100, 50), description="Color of region below maximum time")


class RegionContourColor(BaseModel):
    region_contour_color: TColor = Field(
        (100, 100, 100),
        description="Color of dashed line which shows maximum time and how file is split into gathers",
    )


class RegionContourWidth(BaseModel):
    region_contour_width: float = Field(
        0.2,
        description="Width of dashed line which shows maximum time and how file is split into gathers",
    )




