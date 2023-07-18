from first_breaks.data_models.dependent import XAxis
from first_breaks.data_models.independent import (
    Clip,
    FillBlack,
    Gain,
    Normalize,
    PicksColor,
    RegionContourColor,
    RegionContourWidth,
    RegionPolyColor,
)


class Defaults(
    Normalize,
    Gain,
    Clip,
    FillBlack,
    PicksColor,
    RegionPolyColor,
    RegionContourColor,
    RegionContourWidth,
    XAxis,
):
    pass


DEFAULTS = Defaults()
