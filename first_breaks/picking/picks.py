import uuid
from typing import Optional, Union, Literal, List, Any

import numpy as np
from first_breaks.utils.utils import UnitsConverter, generate_color

from pydantic import model_validator, Field, UUID4

from first_breaks.data_models.independent import PicksColor, PicksWidth


class Picks(PicksColor, PicksWidth):
    values: Union[np.ndarray, List[Union[int, float]]]
    unit: Literal["mcs", "ms", "sample"]

    dt_mcs: Optional[float] = None
    confidence: Optional[Union[np.ndarray, List[Union[int, float]]]] = None
    created_by_nn: Optional[bool] = None
    created_manually: Optional[bool] = None
    modified_manually: Optional[bool] = None

    active: Optional[bool] = None

    _units_converter: Optional[UnitsConverter] = None

    id: UUID4 = Field(default_factory=uuid.uuid4)

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

    def _raise_if_no_dt_mcs(self):
        if self.dt_mcs is None:
            raise ValueError("'dt_mcs' should be specified")

    @property
    def picks_in_ms(self):
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
    def picks_in_mcs(self):
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
    def picks_in_samples(self):
        self._raise_if_no_dt_mcs()
        if self.unit == "sample":
            return self.values
        elif self.unit == "ms":
            return self._units_converter.ms2index(self.values)
        elif self.unit == "mcs":
            return self._units_converter.mcs2index(self.values)
        else:
            raise ValueError("Wrong 'unit'")

    def create_duplicate(self, keep_color: bool = False) -> "Picks":
        values = self.values.copy()
        confidence = self.confidence.copy() if self.confidence is not None else None

        return Picks(
            values=values,
            confidence=confidence,
            dt_mcs=self.dt_mcs,
            unit=self.unit,
            created_manually=self.created_manually,
            created_by_nn=self.created_by_nn,
            modified_manually=self.modified_manually,
            picks_color=self.picks_color if keep_color else generate_color(),
        )


