import uuid
from typing import Annotated, List, Literal, Optional, Union

import numpy as np
from pydantic import UUID4, Field, model_validator

from first_breaks.data_models.independent import (
    F1F2,
    F3F4,
    Clip,
    DefaultModel,
    Gain,
    MaximumTime,
    Normalize,
    TColor,
    TracesPerGather,
    TracesToInverse,
)
from first_breaks.utils.utils import UnitsConverter, generate_color

TValues = Union[np.ndarray, List[Union[int, float]]]


class PickingParameters(
    TracesPerGather,
    MaximumTime,
    TracesToInverse,
    F1F2,
    F3F4,
    Gain,
    Clip,
    Normalize,
):
    pass


DEFAULT_PICKS_WIDTH = 3.0


class Picks(DefaultModel):
    values: TValues
    unit: Literal["mcs", "ms", "sample"]

    dt_mcs: Optional[float] = None
    confidence: Optional[Union[np.ndarray, List[Union[int, float]]]] = None
    heatmap: Optional[np.ndarray] = None
    created_by_nn: Optional[bool] = None
    created_manually: Optional[bool] = None
    modified_manually: Optional[bool] = None
    picking_parameters: Optional[PickingParameters] = None
    color: TColor = Field(default_factory=generate_color, description="Color for picks")  # type: ignore
    width: Annotated[float, Field(description="Width of pick line")] = DEFAULT_PICKS_WIDTH

    active: Optional[bool] = None

    _units_converter: Optional[UnitsConverter] = None

    id: UUID4 = Field(default_factory=uuid.uuid4)

    def __len__(self) -> int:
        return len(self.values)

    def __hash__(self) -> int:
        return int(self.id)

    @model_validator(mode="after")  # type: ignore
    def _sync_units_converter_with_dt_mcs(self) -> "Picks":
        prev_assignment = self.model_config.get("validate_assignment", None)
        self.model_config["validate_assignment"] = False

        if self.dt_mcs is not None:
            if self._units_converter is None or self._units_converter.sgy_mcs != self.dt_mcs:
                self._units_converter = UnitsConverter(sgy_mcs=self.dt_mcs)
        else:
            self._units_converter = None

        self.model_config["validate_assignment"] = prev_assignment
        return self  # type: ignore

    def _raise_if_no_dt_mcs(self) -> None:
        if self.dt_mcs is None:
            raise ValueError("'dt_mcs' should be specified")

    @property
    def picks_in_ms(self) -> TValues:
        self._raise_if_no_dt_mcs()
        if self.unit == "sample":
            return self._units_converter.index2ms(self.values)
        elif self.unit == "ms":
            return self.values
        elif self.unit == "mcs":
            return self._units_converter.mcs2ms(self.values)
        else:
            raise ValueError("Wrong 'unit'")

    @property
    def picks_in_mcs(self) -> TValues:
        self._raise_if_no_dt_mcs()
        if self.unit == "sample":
            return self._units_converter.index2mcs(self.values)
        elif self.unit == "ms":
            return self._units_converter.ms2mcs(self.values)
        elif self.unit == "mcs":
            return self.values
        else:
            raise ValueError("Wrong 'unit'")

    @property
    def picks_in_samples(self) -> TValues:
        self._raise_if_no_dt_mcs()
        if self.unit == "sample":
            return self.values
        elif self.unit == "ms":
            return self._units_converter.ms2index(self.values)
        elif self.unit == "mcs":
            return self._units_converter.mcs2index(self.values)
        else:
            raise ValueError("Wrong 'unit'")

    def from_ms(self, values: TValues) -> None:
        self._raise_if_no_dt_mcs()
        if self.unit == "sample":
            self.values = self._units_converter.ms2index(values)
        elif self.unit == "ms":
            self.values = values
        elif self.unit == "mcs":
            self.values = self._units_converter.ms2mcs(values)
        else:
            raise ValueError("Wrong 'unit'")

    def from_mcs(self, values: TValues) -> None:
        self._raise_if_no_dt_mcs()
        if self.unit == "sample":
            self.values = self._units_converter.mcs2index(values)
        elif self.unit == "ms":
            self.values = self._units_converter.mcs2ms(values)
        elif self.unit == "mcs":
            self.values = values
        else:
            raise ValueError("Wrong 'unit'")

    def from_samples(self, values: TValues) -> None:
        self._raise_if_no_dt_mcs()
        if self.unit == "sample":
            self.values = values
        elif self.unit == "ms":
            self.values = self._units_converter.index2ms(values)
        elif self.unit == "mcs":
            self.values = self._units_converter.index2mcs(values)
        else:
            raise ValueError("Wrong 'unit'")

    def create_duplicate(self, keep_color: bool = False) -> "Picks":
        values = self.values.copy()
        confidence = self.confidence.copy() if self.confidence is not None else None
        heatmap = self.heatmap.copy() if self.heatmap is not None else None

        return Picks(
            values=values,
            confidence=confidence,
            heatmap=heatmap,
            dt_mcs=self.dt_mcs,
            unit=self.unit,
            created_manually=True,
            created_by_nn=self.created_by_nn,
            modified_manually=True,
            picking_parameters=self.picking_parameters,
            color=self.color if keep_color else generate_color(),
        )
