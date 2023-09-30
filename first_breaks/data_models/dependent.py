from pathlib import Path
from typing import Any, Optional, Union, Literal

from pydantic import AfterValidator, Field, field_validator, model_validator
from pydantic_core.core_schema import FieldValidationInfo

from first_breaks.data_models.independent import DefaultModel, TraceBytePosition
from first_breaks.sgy.headers import Headers, TraceHeaders
from first_breaks.sgy.reader import SGY
from first_breaks.utils.cuda import ONNX_CUDA_AVAILABLE

TRACE_HEADER_NAMES = [v[1] for v in TraceHeaders().headers_schema]


class XAxis(DefaultModel):
    x_axis: Optional[str] = Field(None, description="Source of labels for X axis")

    @field_validator("x_axis")
    def validate_x_axis(cls, v: Any) -> Any:
        if v is None:
            return v
        else:
            if v not in TRACE_HEADER_NAMES:
                raise ValueError(f"'x_axis' must be None or one of trace header name, got {v}")
            else:
                return v


class Encoding(DefaultModel):
    encoding: str = Field("i", description="How to encode value")

    @field_validator("encoding")
    def validate_encoding(cls, v: str) -> str:
        if v not in Headers().format2size.keys():
            raise ValueError(f"'encoding' must be one of {Headers().format2size.keys()}")
        else:
            return v


class TraceHeaderParams(TraceBytePosition, Encoding):
    @field_validator("byte_position")
    def validate_position_depends_on_encoding(cls, v: int, based_validation_info: FieldValidationInfo) -> int:
        encoding = based_validation_info.data["encoding"]
        size = Headers.format2size[encoding]
        if v + size > 240:
            raise ValueError(
                f"'byte_position' is greater than allowed for '{encoding}' encoding. "
                f"Maximum allowed: {240 - size}, got {v}"
            )
        return v


class SGYModel(DefaultModel):
    sgy: Union[SGY, str, Path, bytes] = Field(..., description="Source of SGY data")


class Device(DefaultModel):
    device: Literal["cpu", "cuda"] = Field(
        "cuda" if ONNX_CUDA_AVAILABLE else "cpu",
        description="Device to compute first breaks"
    )
