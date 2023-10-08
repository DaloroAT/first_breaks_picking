from first_breaks.data_models.dependent import Device, XAxis
from first_breaks.data_models.independent import (
    F1F2,
    F3F4,
    BatchSize,
    Clip,
    FillBlack,
    Gain,
    InvertX,
    InvertY,
    MaximumTime,
    Normalize,
    PicksColor,
    RegionContourColor,
    RegionContourWidth,
    RegionPolyColor,
    TracesPerGather,
    VSPView,
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
    MaximumTime,
):
    pass


DEFAULTS = Defaults()
