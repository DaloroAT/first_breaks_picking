from typing import Optional, Literal, Union, Any

from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import FieldValidationInfo

from first_breaks.data_models.independent import DefaultModel
from first_breaks.sgy.headers import TraceHeaders
from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import download_demo_sgy

TRACE_HEADER_NAMES = [v[1] for v in TraceHeaders().headers_schema]


class XAxis(BaseModel):
    x_axis: Optional[str] = Field(None,
                                  description="Source of labels for X axis")

    @field_validator("x_axis")
    def validate_x_axis(cls, v: Any) -> Any:
        if v is None:
            return v
        else:
            if v not in TRACE_HEADER_NAMES:
                raise ValueError(f"'x_axis' must be None or one of trace header name, got {v}")
            else:
                return v
