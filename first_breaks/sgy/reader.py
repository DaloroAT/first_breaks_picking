from __future__ import annotations

import io
import shutil
import struct
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from first_breaks.sgy.headers import FileHeaders, TraceHeaderEnum, TraceHeaders
from first_breaks.utils.utils import UnitsConverter, calc_hash, chunk_iterable, get_io

SizeHW = Tuple[int, int]


# =============================================================================
# Constants
# =============================================================================

# SGY file structure sizes (bytes)
FILE_HEADER_SIZE = 3600
TRACE_HEADER_SIZE = 240
MIN_SGY_FILE_SIZE = FILE_HEADER_SIZE + TRACE_HEADER_SIZE + 1  # At least one sample

# Byte positions in file header
DATA_FORMAT_BYTE_POSITION = 3224
NUM_SAMPLES_BYTE_POSITION = 3220

# Valid data format range for endianess detection
DATA_FORMAT_MIN = 1
DATA_FORMAT_MAX = 16

# IBM float constants
IBM_EXPONENT_BIAS = 64
IBM_MANTISSA_BITS = 24
IBM_SIGN_BIT = 31
IBM_EXPONENT_MASK = 0x7F
IBM_MANTISSA_MASK = 0x00FFFFFF
IBM_BASE = 16
IBM_BASE_LOG2 = 4  # log2(16)

# Conversion factors
MCS_TO_MS_FACTOR = 1e-3
MS_TO_HZ_FACTOR = 1000

# Array dimension constants
DIM_1D = 1
DIM_2D = 2

# Default values for new files
DEFAULT_ENDIANESS = ">"
DEFAULT_DATA_FORMAT = 5  # IEEE float
DEFAULT_SCALAR_VALUE = 1.0
DEFAULT_HEADER_VALUE = 0

# String padding
HEADER_STRING_PAD = b" "


class DataFormat(IntEnum):
    """SEG-Y data sample format codes."""

    IBM_FLOAT = 1  # 4-byte IBM floating point
    INT32 = 2  # 4-byte two's complement integer
    INT16 = 3  # 2-byte two's complement integer
    FIXED_POINT = 4  # 32-bit fixed point with gain (obsolete)
    IEEE_FLOAT = 5  # 4-byte IEEE floating point
    IEEE_DOUBLE = 6  # 8-byte IEEE floating point

    @classmethod
    def is_valid(cls, value: int) -> bool:
        """Check if value is a valid DataFormat."""
        return value in cls._value2member_map_

    @classmethod
    def is_supported_for_reading(cls, value: int) -> bool:
        """Check if format is supported for reading."""
        return value in (cls.IBM_FLOAT, cls.INT32, cls.INT16, cls.IEEE_FLOAT, cls.IEEE_DOUBLE)

    @classmethod
    def is_supported_for_writing(cls, value: int) -> bool:
        """Check if format is supported for writing."""
        return value in (cls.IBM_FLOAT, cls.INT32, cls.INT16, cls.IEEE_FLOAT, cls.IEEE_DOUBLE)


# Bytes per sample for each data format
FORMAT_TO_BYTES_PER_SAMPLE: Dict[int, int] = {
    DataFormat.IBM_FLOAT: 4,
    DataFormat.INT32: 4,
    DataFormat.INT16: 2,
    DataFormat.FIXED_POINT: 4,
    DataFormat.IEEE_FLOAT: 4,
    DataFormat.IEEE_DOUBLE: 8,
}

# Numpy dtype strings for each format
FORMAT_TO_DTYPE: Dict[int, str] = {
    DataFormat.INT32: "i4",
    DataFormat.INT16: "i2",
    DataFormat.IEEE_FLOAT: "f4",
    DataFormat.IEEE_DOUBLE: "f8",
}


class Endianess:
    """Byte order constants."""

    BIG = ">"
    LITTLE = "<"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if value is a valid endianess marker."""
        return value in (cls.BIG, cls.LITTLE)


# =============================================================================
# Exceptions
# =============================================================================


class NotImplementedReader(Exception):
    pass


class InvalidSGY(Exception):
    pass


class SGYInitParamsError(Exception):
    pass


class InvalidSamplesSlice(Exception):
    pass


# =============================================================================
# SGY Class
# =============================================================================


class SGY:
    """SEG-Y file reader and writer.

    Supports reading and writing SEG-Y format seismic data files with various
    data sample formats including IBM float, IEEE float, and integer formats.
    """

    # Class-level format mapping (for backward compatibility)
    fmt2bps = FORMAT_TO_BYTES_PER_SAMPLE

    # =========================================================================
    # Properties - Data attributes
    # =========================================================================

    @property
    def source(self) -> Union[str, Path, bytes, np.ndarray]:
        """The original source used to create this SGY instance."""
        return self.__source

    @property
    def dt(self) -> int:
        """Sample interval in microseconds."""
        return self.__dt

    @property
    def ns(self) -> int:
        """Number of samples per trace."""
        return self.__ns

    @property
    def ntr(self) -> int:
        """Number of traces."""
        return self.__ntr

    @property
    def dt_mcs(self) -> int:
        """Sample interval in microseconds (alias for dt)."""
        return self.__dt

    @property
    def dt_ms(self) -> float:
        """Sample interval in milliseconds."""
        return self.__dt * MCS_TO_MS_FACTOR

    @property
    def num_samples(self) -> int:
        """Number of samples per trace (alias for ns)."""
        return self.__ns

    @property
    def num_traces(self) -> int:
        """Number of traces (alias for ntr)."""
        return self.__ntr

    @property
    def shape(self) -> SizeHW:
        """Shape of the trace data as (num_samples, num_traces)."""
        return self.__ns, self.__ntr

    @property
    def fs(self) -> float:
        """Sampling frequency in Hz."""
        return MS_TO_HZ_FACTOR / self.dt_ms

    @property
    def max_time_ms(self) -> float:
        """Maximum recording time in milliseconds."""
        return self.__ns * self.dt_ms

    # =========================================================================
    # Properties - Format attributes
    # =========================================================================

    @property
    def endianess(self) -> str:
        """Byte order: '>' for big-endian, '<' for little-endian."""
        return self.__endianess

    @property
    def data_format(self) -> int:
        """Data sample format code (see DataFormat enum)."""
        return self.__data_fmt

    @property
    def is_source_ndarray(self) -> bool:
        """True if the source was a numpy array, False otherwise."""
        return self.__is_source_ndarray

    # =========================================================================
    # Properties - Headers
    # =========================================================================

    @property
    def general_headers(self) -> Dict[str, Any]:
        """File (general) headers dictionary."""
        return self.__general_headers

    @property
    def traces_headers(self) -> pd.DataFrame:
        """Trace headers as a DataFrame with scaled values."""
        return self.__traces_headers

    @property
    def traces_headers_raw(self) -> pd.DataFrame:
        """Trace headers as a DataFrame with raw (unscaled) values."""
        return self.__traces_headers_raw

    @property
    def traces_headers_schema(self) -> TraceHeaders:
        """Schema for trace headers."""
        return self.__traces_headers_schema

    # =========================================================================
    # Constructor
    # =========================================================================

    def __init__(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        dt_mcs: Optional[Union[int, float]] = None,
        general_headers_schema: FileHeaders = FileHeaders(),
        traces_headers_schema: TraceHeaders = TraceHeaders(),
        file_headers: Optional[Dict[str, Any]] = None,
        traces_headers: Optional[pd.DataFrame] = None,
    ):
        """Initialize SGY from a file path, bytes, or numpy array.

        Args:
            source: File path, bytes buffer, or numpy array with trace data
            dt_mcs: Sample interval in microseconds (required if source is ndarray)
            general_headers_schema: Schema for file headers
            traces_headers_schema: Schema for trace headers
            file_headers: Optional file headers dict (for ndarray source)
            traces_headers: Optional trace headers DataFrame (for ndarray source)
        """
        self.__source = source
        self.__dt_mcs_input = dt_mcs
        self.__file_headers_input = file_headers
        self.__traces_headers_input = traces_headers

        # Header schemas
        self.__general_headers_schema: FileHeaders = general_headers_schema
        self.__traces_headers_schema: TraceHeaders = traces_headers_schema

        # These will be set during initialization - declare with proper types
        self.__traces: Optional[np.ndarray] = None
        self.__dt: int = 0
        self.__ns: int = 0
        self.__ntr: int = 0
        self.__endianess: str = DEFAULT_ENDIANESS
        self.__bps: int = FORMAT_TO_BYTES_PER_SAMPLE[DEFAULT_DATA_FORMAT]
        self.__data_fmt: int = DEFAULT_DATA_FORMAT
        self.__general_headers: Dict[str, Any] = {}
        self.__traces_headers: pd.DataFrame = pd.DataFrame()
        self.__traces_headers_raw: pd.DataFrame = pd.DataFrame()
        self.__units_converter: UnitsConverter = UnitsConverter(sgy_mcs=1)  # Temporary, will be reset

        # I/O and state attributes
        self.__descriptor: Optional[Union[io.BytesIO, io.FileIO]] = None
        self.__is_source_ndarray: bool = False
        self.__hash_value: Optional[str] = None

        self.__init_sgy()

    # =========================================================================
    # Public methods
    # =========================================================================

    def ms2index(self, ms_value: float) -> int:
        """Convert milliseconds to sample index."""
        return self.__units_converter.ms2index(ms_value)

    def get_hash(self) -> Optional[str]:
        """Get MD5 hash of the source file (None if source is ndarray)."""
        if self.__is_source_ndarray:
            return None
        else:
            if self.__hash_value is None:
                self.__descriptor = get_io(self.__source, mode="rb")
                self.__hash_value = calc_hash(self.__descriptor)
                self.__descriptor.close()
                self.__descriptor = None
            return self.__hash_value

    def read(self, min_sample: Optional[int] = None, max_sample: Optional[int] = None) -> np.ndarray:
        """Read all traces from the SGY file.

        Args:
            min_sample: Start sample index (0-based, inclusive)
            max_sample: End sample index (exclusive)

        Returns:
            Numpy array of shape (num_samples, num_traces)
        """
        ids = list(range(self.__ntr))
        traces = self.read_traces_by_ids(ids, min_sample, max_sample)
        self.replace_traces(traces)
        return traces

    def read_traces_by_ids(
        self,
        ids: Sequence[int],
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> np.ndarray:
        """Read specific traces by their indices.

        Args:
            ids: Sequence of trace indices to read
            min_sample: Start sample index (0-based, inclusive)
            max_sample: End sample index (exclusive)

        Returns:
            Numpy array of shape (num_samples, len(ids))
        """
        if min_sample is not None:
            if min_sample < 0 or not isinstance(min_sample, int):
                raise InvalidSamplesSlice("Invalid minimum slice index")
            min_sample = int(np.clip(min_sample, 0, self.__ns))
        else:
            min_sample = 0

        if max_sample is not None:
            if max_sample < 1 or not isinstance(max_sample, int):
                raise InvalidSamplesSlice("Invalid maximum slice index")
            max_sample = int(np.clip(max_sample, 0, self.__ns))
        else:
            max_sample = self.__ns

        if min_sample >= max_sample:
            raise InvalidSamplesSlice("Minimum slice index is greater or equal to maximum index")

        len_slice = max_sample - min_sample
        ids = [idx for idx in ids if idx < self.__ntr]

        if len(ids) == 0:
            raise ValueError("The requested IDs were not found in the file")

        if self.__is_source_ndarray or (self.__traces is not None and self.__traces.shape == self.shape):
            return self.__read_block_ndarray(ids, min_sample, len_slice)
        else:
            return self.__read_block_external(ids, min_sample, len_slice)

    def get_chunked_reader(
        self,
        chunk_size: int,
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        """Get a generator that yields traces in chunks.

        Args:
            chunk_size: Number of traces per chunk
            min_sample: Start sample index
            max_sample: End sample index

        Yields:
            Numpy arrays of shape (num_samples, chunk_size)
        """
        chunk_size = min(chunk_size, self.__ntr)
        all_ids = list(range(self.__ntr))

        for ids in chunk_iterable(all_ids, chunk_size):
            yield self.read_traces_by_ids(ids, min_sample, max_sample)

    def replace_traces(self, traces: np.ndarray) -> None:
        """Replace the internal trace data with new data."""
        if traces.shape == self.shape:
            self.__traces = traces

    def read_custom_trace_header(self, byte_position: int, encoding: str) -> Tuple[Any, ...]:
        """Read a custom trace header field from all traces.

        Args:
            byte_position: Byte offset within trace header (0-239)
            encoding: Struct format character (e.g., 'i', 'h', 'f')

        Returns:
            Tuple of header values, one per trace
        """
        self.__descriptor = get_io(self.__source, mode="rb")
        result = self.__read_custom_trace_header_with_existed_descriptor(byte_position, encoding)
        self.__descriptor.close()
        self.__descriptor = None
        return result

    def write(
        self,
        output_path: Union[str, Path],
        data_format: Optional[int] = None,
        endianess: Optional[str] = None,
    ) -> None:
        """Write the SGY data to a file.

        Args:
            output_path: Path to the output file
            data_format: Data sample format (see DataFormat enum). If None, uses current format.
            endianess: Byte order ('>' for big-endian, '<' for little-endian). If None, uses current.
        """
        if self.__traces is None:
            raise ValueError("No trace data available. Call read() first or provide numpy array.")

        # Use provided values or fall back to current instance values
        write_data_fmt = data_format if data_format is not None else self.__data_fmt
        write_endianess = endianess if endianess is not None else self.__endianess

        # Validate data format
        if not DataFormat.is_supported_for_writing(write_data_fmt):
            raise NotImplementedError(f"Data format {write_data_fmt} not supported for writing")

        # Validate endianess
        if not Endianess.is_valid(write_endianess):
            raise ValueError(f"Invalid endianess '{write_endianess}'. Use '>' or '<'.")

        write_bps = FORMAT_TO_BYTES_PER_SAMPLE[write_data_fmt]

        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # Update file headers with write format (create a copy to avoid mutation)
        write_headers = self.__general_headers.copy()
        write_headers[self.__general_headers_schema.data_sample_format_name] = write_data_fmt

        with open(output_path, "wb") as f:
            self.__write_file_headers(f, write_headers, write_endianess)
            self.__write_traces(f, write_data_fmt, write_bps, write_endianess)

    def export_sgy_with_picks(
        self,
        output_fname: Union[str, Path],
        picks_in_mcs: List[float],
        byte_position: int = 236,
        encoding: Optional[str] = None,
        picks_unit: Optional[str] = "mcs",
    ) -> None:
        """Export SGY file with first break picks written to trace headers.

        Args:
            output_fname: Output file path
            picks_in_mcs: List of pick times in microseconds
            byte_position: Byte position in trace header to write picks (0-236)
            encoding: Struct format character for picks
            picks_unit: Unit for picks ('ms', 'mcs', or 'sample')
        """
        max_pick_byte_position = TRACE_HEADER_SIZE - 4  # Leave room for at least 4-byte value
        assert not self.__is_source_ndarray, "Only true SGY can be used for importing picks"
        assert 0 <= byte_position <= max_pick_byte_position, f"Only 0-{max_pick_byte_position} bytes can be used"
        assert len(picks_in_mcs) == self.__ntr, "Number of traces and picks differs"
        assert picks_unit in ["ms", "mcs", "sample"]

        Path(output_fname).parent.mkdir(exist_ok=True, parents=True)

        if isinstance(self.__source, (str, Path)):
            if Path(self.__source).resolve() != Path(output_fname).resolve():
                shutil.copyfile(str(self.__source), str(output_fname))
        elif isinstance(self.__source, bytes):
            with open(output_fname, "wb+") as f_output:
                f_output.write(self.__source)
        else:
            raise TypeError("Invalid type of source data")

        if encoding is None:
            encoding = [
                pack_type
                for _, header, pack_type in self.__traces_headers_schema.headers_schema
                if header == self.__traces_headers_schema.fb_pick_default
            ][0]

        cast_to = float if encoding in ["f", "d"] else int

        if picks_unit == "ms":
            picks = self.__units_converter.mcs2ms(picks_in_mcs, cast_to=cast_to)
        elif picks_unit == "mcs":
            picks = picks_in_mcs
        elif picks_unit == "sample":
            picks = self.__units_converter.mcs2index(picks_in_mcs, cast_to=cast_to)
        else:
            raise ValueError("Unsupported 'picking unit'")

        self.__descriptor = get_io(output_fname, mode="r+b")
        trace_block_size = TRACE_HEADER_SIZE + self.__ns * self.__bps
        for idx, pick in enumerate(picks):  # type: ignore
            pointer = FILE_HEADER_SIZE + trace_block_size * idx + byte_position
            pick_byte = struct.pack(f"{self.__endianess}{encoding}", pick)
            self.__descriptor.seek(pointer)
            self.__descriptor.write(pick_byte)
        self.__descriptor.close()
        self.__descriptor = None

    # =========================================================================
    # Private initialization methods
    # =========================================================================

    def __init_sgy(self) -> None:
        """Initialize the SGY instance based on source type."""
        if isinstance(self.__source, (str, Path, bytes)):
            if self.__dt_mcs_input is not None:
                raise SGYInitParamsError("Argument 'dt_mcs' must be empty if SGY created from external sources")
            self.__init_from_external()
            self.__is_source_ndarray = False
        elif isinstance(self.__source, np.ndarray):
            if self.__dt_mcs_input is None:
                raise SGYInitParamsError("Argument 'dt_mcs' is required if nd.array is used as input")
            self.__init_from_numpy()
            self.__is_source_ndarray = True
        else:
            raise SGYInitParamsError("Only `str, Path, bytes, np.ndarray` types are available as input")

        # Create units converter with actual dt value
        self.__units_converter = UnitsConverter(sgy_mcs=self.__dt)

    def __init_from_numpy(self) -> None:
        """Initialize from a numpy array."""
        source_array: np.ndarray = self.__source  # type: ignore

        if source_array.ndim not in (DIM_1D, DIM_2D):
            raise ValueError("Only 1D or 2D arrays are available")

        if source_array.ndim == DIM_1D:
            self.__ntr = 1
        else:
            self.__ntr = source_array.shape[1]

        self.__traces = source_array
        self.__dt = int(self.__dt_mcs_input)  # type: ignore
        self.__ns = source_array.shape[0]
        self.__endianess = DEFAULT_ENDIANESS
        self.__data_fmt = DataFormat.IEEE_FLOAT
        self.__bps = FORMAT_TO_BYTES_PER_SAMPLE[self.__data_fmt]

        # Handle file headers
        if self.__file_headers_input is not None:
            self.__general_headers = self.__file_headers_input.copy()
            # Validate dt_mcs matches file header if present
            header_dt = self.__general_headers.get(self.__general_headers_schema.dt_name)
            if header_dt is not None and header_dt != self.__dt:
                raise SGYInitParamsError(
                    f"dt_mcs ({self.__dt}) does not match file header dt ({header_dt})"
                )
            # Validate data format if present
            header_fmt = self.__general_headers.get(self.__general_headers_schema.data_sample_format_name)
            if header_fmt is not None:
                if not DataFormat.is_valid(header_fmt):
                    raise SGYInitParamsError(f"Invalid data format in file headers: {header_fmt}")
                self.__data_fmt = header_fmt
                self.__bps = FORMAT_TO_BYTES_PER_SAMPLE[self.__data_fmt]
        else:
            self.__general_headers = self.__generate_default_file_headers()

        # Handle trace headers
        if self.__traces_headers_input is not None:
            if len(self.__traces_headers_input) != self.__ntr:
                raise SGYInitParamsError(
                    f"traces_headers length ({len(self.__traces_headers_input)}) "
                    f"does not match number of traces ({self.__ntr})"
                )
            self.__traces_headers = self.__traces_headers_input.copy()
            self.__traces_headers_raw = self.__unscale_traces_headers(self.__traces_headers)
        else:
            self.__traces_headers = self.__generate_default_trace_headers()
            self.__traces_headers_raw = self.__traces_headers.copy()

    def __init_from_external(self) -> None:
        """Initialize from an external file or bytes."""
        self.__descriptor = get_io(self.__source, mode="rb")
        self.__read_endianess()
        self.__read_general_headers()
        self.__read_traces_headers()
        self.__scalar_raw_traces_headers()
        self.__descriptor.close()
        self.__descriptor = None

    # =========================================================================
    # Private header reading methods
    # =========================================================================

    def __read_endianess(self) -> None:
        """Determine file endianess from data format field."""
        num_bytes = self.__descriptor.seek(0, 2)
        if num_bytes < MIN_SGY_FILE_SIZE:
            raise InvalidSGY("Invalid structure of SGY file. File is small")

        self.__descriptor.seek(DATA_FORMAT_BYTE_POSITION)
        value = self.__descriptor.read(2)

        big = struct.unpack(f"{Endianess.BIG}H", value)[0]
        little = struct.unpack(f"{Endianess.LITTLE}H", value)[0]

        big_valid = DATA_FORMAT_MIN <= big <= DATA_FORMAT_MAX
        little_valid = DATA_FORMAT_MIN <= little <= DATA_FORMAT_MAX

        if big_valid or little_valid:
            self.__endianess = Endianess.BIG if big_valid else Endianess.LITTLE
        else:
            raise InvalidSGY("Invalid endianess of SGY file")

    def __read_general_headers(self) -> None:
        """Read file (general) headers."""
        gen_headers: Dict[str, Any] = {}
        for pointer, name, fmt in self.__general_headers_schema.headers_schema:
            self.__descriptor.seek(pointer)
            size = self.__general_headers_schema.get_num_bytes(fmt)
            gen_headers[name] = struct.unpack(f"{self.__endianess}{fmt}", self.__descriptor.read(size))[0]
        self.__general_headers = gen_headers

        self.__ns = self.__general_headers[self.__general_headers_schema.ns_name]
        if self.__ns < 1:
            raise InvalidSGY("Invalid number of samples")

        self.__dt = self.__general_headers[self.__general_headers_schema.dt_name]
        if self.__dt < 0:
            raise InvalidSGY("Invalid time discretization")

        self.__data_fmt = self.__general_headers[self.__general_headers_schema.data_sample_format_name]
        if not DataFormat.is_valid(self.__data_fmt):
            raise NotImplementedReader(f"Unknown data sample format '{self.__data_fmt}'")
        if not DataFormat.is_supported_for_reading(self.__data_fmt):
            raise NotImplementedReader(f"Not supported data sample format '{self.__data_fmt}'")
        self.__bps = FORMAT_TO_BYTES_PER_SAMPLE[self.__data_fmt]

        num_bytes = self.__descriptor.seek(0, 2)
        trace_block_size = TRACE_HEADER_SIZE + self.__ns * self.__bps
        self.__ntr = int((num_bytes - FILE_HEADER_SIZE) / trace_block_size)
        if num_bytes != (FILE_HEADER_SIZE + trace_block_size * self.__ntr):
            raise InvalidSGY("Invalid number of bytes")

    def __read_traces_headers(self) -> None:
        """Read all trace headers into a DataFrame."""
        trace_block_size = TRACE_HEADER_SIZE + self.__ns * self.__bps

        all_headers_buffer = bytearray(self.__ntr * TRACE_HEADER_SIZE)
        for idx in range(self.__ntr):
            pointer = FILE_HEADER_SIZE + trace_block_size * idx
            self.__descriptor.seek(pointer)
            all_headers_buffer[idx * TRACE_HEADER_SIZE : (idx + 1) * TRACE_HEADER_SIZE] = self.__descriptor.read(
                TRACE_HEADER_SIZE
            )

        all_headers_bytes = bytes(all_headers_buffer)
        traces_headers: Dict[str, List[Any]] = {}
        for offset, name, fmt in self.__traces_headers_schema.headers_schema:
            fmt_str = f"{self.__endianess}{fmt}"
            values = [
                struct.unpack_from(fmt_str, all_headers_bytes, idx * TRACE_HEADER_SIZE + offset)[0]
                for idx in range(self.__ntr)
            ]
            traces_headers[name] = values

        self.__traces_headers_raw = pd.DataFrame(data=traces_headers)

    def __scalar_raw_traces_headers(self) -> None:
        """Apply scalar fields to trace headers."""
        self.__traces_headers = self.__traces_headers_raw.copy()
        for scalar_from, apply_to_columns in self.__traces_headers_schema.scalar_from2apply.items():
            scalar = self.__traces_headers_raw[scalar_from].values.astype(np.float64)

            scalar[scalar == 0] = DEFAULT_SCALAR_VALUE
            negative_mask = scalar < 0
            scalar[negative_mask] = DEFAULT_SCALAR_VALUE / np.abs(scalar[negative_mask])

            for col in apply_to_columns:
                self.__traces_headers[col] = self.__traces_headers[col].values * scalar

    def __unscale_traces_headers(self, scaled_headers: pd.DataFrame) -> pd.DataFrame:
        """Reverse the scalar application to get raw header values for writing."""
        raw_headers = scaled_headers.copy()
        for scalar_from, apply_to_columns in self.__traces_headers_schema.scalar_from2apply.items():
            if scalar_from not in raw_headers.columns:
                continue
            scalar = raw_headers[scalar_from].values.astype(np.float64)

            for col in apply_to_columns:
                if col not in raw_headers.columns:
                    continue
                col_scalar = scalar.copy()
                col_scalar[col_scalar == 0] = DEFAULT_SCALAR_VALUE

                # Reverse the scaling:
                # Original: if scalar < 0, scaled = value / |scalar|, so raw = scaled * |scalar|
                # Original: if scalar > 0, scaled = value * scalar, so raw = scaled / scalar
                negative_mask = col_scalar < 0
                positive_mask = col_scalar > 0

                result = raw_headers[col].values.astype(np.float64)
                result[negative_mask] = result[negative_mask] * np.abs(col_scalar[negative_mask])
                result[positive_mask] = result[positive_mask] / col_scalar[positive_mask]

                raw_headers[col] = np.round(result).astype(np.int32)

        return raw_headers

    def __generate_default_file_headers(self) -> Dict[str, Any]:
        """Generate default file headers for a new SGY file."""
        headers: Dict[str, Any] = {}
        for offset, name, fmt in self.__general_headers_schema.headers_schema:
            if fmt.endswith("s"):
                # String field - fill with spaces
                size = int(fmt[:-1]) if fmt[:-1].isdigit() else 1
                headers[name] = HEADER_STRING_PAD * size
            else:
                headers[name] = DEFAULT_HEADER_VALUE

        # Set essential headers
        headers[self.__general_headers_schema.dt_name] = self.__dt
        headers[self.__general_headers_schema.dt_name_orig] = self.__dt
        headers[self.__general_headers_schema.ns_name] = self.__ns
        headers[self.__general_headers_schema.ns_name_orig] = self.__ns
        headers[self.__general_headers_schema.data_sample_format_name] = self.__data_fmt

        return headers

    def __generate_default_trace_headers(self) -> pd.DataFrame:
        """Generate default trace headers for a new SGY file."""
        headers: Dict[str, List[Any]] = {}
        for offset, name, fmt in self.__traces_headers_schema.headers_schema:
            headers[name] = [DEFAULT_HEADER_VALUE] * self.__ntr

        # Set trace numbers using names from schema
        name_map = TraceHeaders._ENUM_TO_NAME
        headers[name_map[TraceHeaderEnum.TRACENO]] = list(range(1, self.__ntr + 1))
        headers[name_map[TraceHeaderEnum.TRACE_SEQUENCE_FILE]] = list(range(1, self.__ntr + 1))
        headers[name_map[TraceHeaderEnum.NUMSMP]] = [self.__ns] * self.__ntr
        headers[name_map[TraceHeaderEnum.DT]] = [self.__dt] * self.__ntr

        return pd.DataFrame(headers)

    # =========================================================================
    # Private writing methods
    # =========================================================================

    def __write_file_headers(self, f: io.FileIO, headers: Dict[str, Any], endianess: str) -> None:
        """Write the 3600-byte file header block."""
        header_buffer = bytearray(FILE_HEADER_SIZE)

        for offset, name, fmt in self.__general_headers_schema.headers_schema:
            value = headers.get(name, DEFAULT_HEADER_VALUE)
            size = self.__general_headers_schema.get_num_bytes(fmt)

            if fmt.endswith("s"):
                # String field
                if isinstance(value, bytes):
                    data = value[:size].ljust(size, HEADER_STRING_PAD)
                else:
                    data = str(value).encode("ascii", errors="replace")[:size].ljust(size, HEADER_STRING_PAD)
                header_buffer[offset : offset + size] = data
            else:
                packed = struct.pack(f"{endianess}{fmt}", int(value))
                header_buffer[offset : offset + size] = packed

        f.write(bytes(header_buffer))

    def __write_traces(self, f: io.FileIO, data_fmt: int, bps: int, endianess: str) -> None:
        """Write all traces with their headers."""
        # Get raw (unscaled) trace headers for writing
        raw_headers = self.__traces_headers_raw

        for trace_idx in range(self.__ntr):
            self.__write_trace_header(f, raw_headers, trace_idx, endianess)
            self.__write_trace_data(f, trace_idx, data_fmt, endianess)

    def __write_trace_header(
        self, f: io.FileIO, raw_headers: pd.DataFrame, trace_idx: int, endianess: str
    ) -> None:
        """Write a single trace header (240 bytes)."""
        header_buffer = bytearray(TRACE_HEADER_SIZE)

        for offset, name, fmt in self.__traces_headers_schema.headers_schema:
            if name in raw_headers.columns:
                value = raw_headers.iloc[trace_idx][name]
            else:
                value = DEFAULT_HEADER_VALUE

            size = self.__traces_headers_schema.get_num_bytes(fmt)
            packed = struct.pack(f"{endianess}{fmt}", int(value))
            header_buffer[offset : offset + size] = packed

        f.write(bytes(header_buffer))

    def __write_trace_data(self, f: io.FileIO, trace_idx: int, data_fmt: int, endianess: str) -> None:
        """Write trace sample data."""
        if self.__traces.ndim == DIM_1D:
            trace_data = self.__traces
        else:
            trace_data = self.__traces[:, trace_idx]

        if data_fmt == DataFormat.IBM_FLOAT:
            data_bytes = self.__convert_to_ibm(trace_data, endianess)
        elif data_fmt == DataFormat.INT32:
            data_bytes = trace_data.astype(f"{endianess}{FORMAT_TO_DTYPE[DataFormat.INT32]}").tobytes()
        elif data_fmt == DataFormat.INT16:
            data_bytes = trace_data.astype(f"{endianess}{FORMAT_TO_DTYPE[DataFormat.INT16]}").tobytes()
        elif data_fmt == DataFormat.IEEE_FLOAT:
            data_bytes = trace_data.astype(f"{endianess}{FORMAT_TO_DTYPE[DataFormat.IEEE_FLOAT]}").tobytes()
        elif data_fmt == DataFormat.IEEE_DOUBLE:
            data_bytes = trace_data.astype(f"{endianess}{FORMAT_TO_DTYPE[DataFormat.IEEE_DOUBLE]}").tobytes()
        else:
            raise NotImplementedError(f"Data format {data_fmt} not supported for writing")

        f.write(data_bytes)

    def __convert_to_ibm(self, data: np.ndarray, endianess: str) -> bytes:
        """Convert IEEE float array to IBM float bytes.

        IBM format: 1 sign bit, 7 exponent bits (base 16, biased by 64), 24 mantissa bits.
        Value = mantissa * 16^(exponent-64), where mantissa is in [1/16, 1).
        """
        result = np.zeros(len(data), dtype=f"{endianess}u4")

        # Handle zeros separately
        non_zero_mask = data != 0

        if np.any(non_zero_mask):
            values = data[non_zero_mask].astype(np.float64)

            # Extract sign and work with absolute values
            sign = np.where(values < 0, 1, 0).astype(np.uint32)
            abs_values = np.abs(values)

            # Use frexp to get mantissa and exponent: value = m * 2^e where m in [0.5, 1)
            ieee_mant, ieee_exp = np.frexp(abs_values)

            # Convert base-2 exponent to base-16: 16^x = 2^(4x), so x = ieee_exp/4
            # IBM exponent is ceiling of ieee_exp/4 to ensure mantissa < 1
            ibm_exp = np.ceil(ieee_exp / float(IBM_BASE_LOG2)).astype(np.int32)

            # Recalculate mantissa for IBM: value / 16^ibm_exp = value / 2^(4*ibm_exp)
            # Using ldexp for numerical stability
            mantissa = np.ldexp(abs_values, -IBM_BASE_LOG2 * ibm_exp)

            # Normalize mantissa to [1/16, 1) - handle edge cases
            mantissa_upper_bound = 1.0
            mantissa_lower_bound = 1.0 / IBM_BASE

            while np.any(mantissa >= mantissa_upper_bound):
                too_large = mantissa >= mantissa_upper_bound
                mantissa[too_large] /= IBM_BASE
                ibm_exp[too_large] += 1

            while np.any((mantissa > 0) & (mantissa < mantissa_lower_bound)):
                too_small = (mantissa > 0) & (mantissa < mantissa_lower_bound)
                mantissa[too_small] *= IBM_BASE
                ibm_exp[too_small] -= 1

            # Add bias to exponent
            ibm_exp_biased = (ibm_exp + IBM_EXPONENT_BIAS).astype(np.uint32)

            # Convert mantissa to 24-bit integer
            mantissa_int = (mantissa * (1 << IBM_MANTISSA_BITS)).astype(np.uint32) & IBM_MANTISSA_MASK

            # Combine: sign (1 bit) | exponent (7 bits) | mantissa (24 bits)
            result[non_zero_mask] = (sign << IBM_SIGN_BIT) | (ibm_exp_biased << IBM_MANTISSA_BITS) | mantissa_int

        return result.tobytes()

    # =========================================================================
    # Private reading methods
    # =========================================================================

    def __read_custom_trace_header_with_existed_descriptor(self, byte_position: int, encoding: str) -> Tuple[Any, ...]:
        """Read a custom trace header field using existing descriptor."""
        size = self.__traces_headers_schema.get_num_bytes(encoding)
        buffer = []
        trace_block_size = TRACE_HEADER_SIZE + self.__ns * self.__bps
        for idx in range(self.__ntr):
            pointer = FILE_HEADER_SIZE + trace_block_size * idx + byte_position
            self.__descriptor.seek(pointer)
            buffer.append(self.__descriptor.read(size))
        return struct.unpack(f"{self.__endianess}{encoding * self.__ntr}", b"".join(buffer))

    def __read_block_ndarray(self, ids: Sequence[int], min_sample: int, length_slice: int) -> np.ndarray:
        """Read a block of traces from the internal numpy array."""
        return self.__traces[min_sample : min_sample + length_slice, ids]

    def __read_block_external(self, ids: Sequence[int], min_sample: int, length_slice: int) -> np.ndarray:
        """Read a block of traces from an external file."""
        trace_bytes = length_slice * self.__bps
        buffer = bytearray(len(ids) * trace_bytes)
        trace_block_size = TRACE_HEADER_SIZE + self.__ns * self.__bps

        self.__descriptor = get_io(self.__source, mode="rb")
        for i, idx in enumerate(ids):
            pointer = FILE_HEADER_SIZE + TRACE_HEADER_SIZE + trace_block_size * idx + min_sample * self.__bps
            self.__descriptor.seek(pointer)
            buffer[i * trace_bytes : (i + 1) * trace_bytes] = self.__descriptor.read(trace_bytes)
        self.__descriptor.close()
        self.__descriptor = None

        traces = self.__read_traces_from_buffer(bytes(buffer), (length_slice, len(ids)))
        return traces

    def __read_traces_from_buffer(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        """Dispatch trace reading to the appropriate format handler."""
        if self.__data_fmt == DataFormat.IBM_FLOAT:
            return self.__read_traces_ibm(buffer, shape)
        elif self.__data_fmt == DataFormat.INT32:
            return self.__read_traces_4b_compl_int(buffer, shape)
        elif self.__data_fmt == DataFormat.INT16:
            return self.__read_traces_2b_compl_int(buffer, shape)
        elif self.__data_fmt == DataFormat.FIXED_POINT:
            raise NotImplementedReader(
                "Not implemented 32-bit fixed point with gain values reader (format 4 of SGY specification)"
            )
        elif self.__data_fmt == DataFormat.IEEE_FLOAT:
            return self.__read_traces_float(buffer, shape)
        elif self.__data_fmt == DataFormat.IEEE_DOUBLE:
            return self.__read_traces_double(buffer, shape)
        else:
            raise ValueError(f"Not supported format: {self.__data_fmt}")

    def __read_traces_ibm(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        """Convert IBM floating point buffer to IEEE float array.

        IBM format: 1 sign bit, 7 exponent bits (base 16, biased by 64), 24 mantissa bits.
        """
        array = np.ndarray(shape, f"{self.__endianess}u4", buffer, order="F")

        zero_mask = array == 0

        sign = (array >> IBM_SIGN_BIT) & 0x01
        exp = (array >> IBM_MANTISSA_BITS) & IBM_EXPONENT_MASK
        frac = array & IBM_MANTISSA_MASK

        # Mantissa in [0, 1)
        mantissa = frac.astype(np.float64) / float(1 << IBM_MANTISSA_BITS)

        # +1 for sign=0, -1 for sign=1
        sign_mult = 1.0 - 2.0 * sign.astype(np.float64)

        # 16^(exp-64) == 2^(4*(exp-64)); ldexp is fast and numerically stable
        shift = (exp.astype(np.int32) - IBM_EXPONENT_BIAS) * IBM_BASE_LOG2
        result = np.ldexp(sign_mult * mantissa, shift).astype(np.float32)

        result[zero_mask] = 0.0
        return result

    def __read_traces_4b_compl_int(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        """Read 4-byte signed integer traces."""
        return np.ndarray(shape, f"{self.__endianess}{FORMAT_TO_DTYPE[DataFormat.INT32]}", buffer, order="F")

    def __read_traces_2b_compl_int(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        """Read 2-byte signed integer traces."""
        return np.ndarray(shape, f"{self.__endianess}{FORMAT_TO_DTYPE[DataFormat.INT16]}", buffer, order="F")

    def __read_traces_float(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        """Read IEEE 4-byte float traces."""
        return np.ndarray(shape, f"{self.__endianess}{FORMAT_TO_DTYPE[DataFormat.IEEE_FLOAT]}", buffer, order="F")

    def __read_traces_double(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        """Read IEEE 8-byte double traces."""
        return np.ndarray(shape, f"{self.__endianess}{FORMAT_TO_DTYPE[DataFormat.IEEE_DOUBLE]}", buffer, order="F")
