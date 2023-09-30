from first_breaks.data_models.dependent import XAxis, Device
from first_breaks.data_models.independent import (
    Clip,
    FillBlack,
    Gain,
    Normalize,
    PicksColor,
    RegionContourColor,
    RegionContourWidth,
    RegionPolyColor, F1F2, F3F4, VSPView, InvertY, InvertX, TracesPerGather, BatchSize, MaximumTime,
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
    F1F2,
    F3F4,
    VSPView,
    InvertX,
    InvertY,
    TracesPerGather,
    Device,
    BatchSize,
    MaximumTime
):
    pass


DEFAULTS = Defaults()
