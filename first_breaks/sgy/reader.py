from __future__ import annotations

import io
import shutil
import struct
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from first_breaks.sgy.headers import FileHeaders, TraceHeaders
from first_breaks.utils.utils import UnitsConverter, calc_hash, chunk_iterable, get_io

SizeHW = Tuple[int, int]


class NotImplementedReader(Exception):
    pass


class InvalidSGY(Exception):
    pass


class SGYInitParamsError(Exception):
    pass


class InvalidSamplesSlice(Exception):
    pass


class SGY:
    fmt2bps = {
        1: 4,  # 4-byte hexadecimal exponent data (IBM single precision floating point)
        2: 4,  # 4-byte, two's complement integer
        3: 2,  # 2-byte, two's complement integer
        4: 4,  # 32-bit fixed point with gain values (Obsolete)
        5: 4,  # 4-byte, IEEE Floating Point
        6: 8,  # 8-byte, IEEE Floating Point
    }

    @property
    def dt(self) -> Union[int, float]:
        return self._dt

    @property
    def ns(self) -> int:
        return self._ns

    @property
    def ntr(self) -> int:
        return self._ntr

    @property
    def dt_mcs(self) -> float:
        return self.dt

    @property
    def dt_ms(self) -> float:
        return self.dt * 1e-3

    @property
    def num_samples(self) -> int:
        return self.ns

    @property
    def num_traces(self) -> int:
        return self.ntr

    @property
    def shape(self) -> SizeHW:
        return self.ns, self.ntr

    @property
    def fs(self) -> float:
        return 1000 / self.dt_ms

    def ms2index(self, ms_value: float) -> int:
        return self.units_converter.ms2index(ms_value)  # type: ignore

    @property
    def max_time_ms(self) -> float:
        return self.num_samples * self.dt_ms

    def __init__(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        dt_mcs: Optional[Union[int, float]] = None,
        general_headers_schema: FileHeaders = FileHeaders(),
        traces_headers_schema: TraceHeaders = TraceHeaders(),
    ):
        self.source = source
        self._dt_mcs_input = dt_mcs

        # data
        self._traces: Optional[np.ndarray] = None
        self._dt: Optional[int] = None
        self._ns: Optional[int] = None
        self._ntr: Optional[int] = None

        # headers for file
        self.general_headers: Optional[Dict[str, Any]] = None
        self._general_headers_schema: FileHeaders = general_headers_schema

        # headers for traces
        self.traces_headers: Optional[pd.DataFrame] = None
        self._traces_headers_raw: Optional[pd.DataFrame] = None
        self._traces_headers_schema: TraceHeaders = traces_headers_schema

        # other
        self._descriptor: Optional[Union[io.BytesIO, io.FileIO]] = None
        self.is_source_ndarray: Optional[bool] = None
        self._endianess: Optional[str] = None
        self._bps: Optional[int] = None
        self._data_fmt: Optional[int] = None
        self._hash_value: Optional[str] = None
        self.units_converter: Optional[UnitsConverter] = None

        self.__init()

    def get_hash(self) -> Optional[str]:
        if self.is_source_ndarray:
            return None
        else:
            if self._hash_value is None:
                self._descriptor = get_io(self.source, mode="rb")
                self._hash_value = calc_hash(self._descriptor)
                self._descriptor.close()
                self._descriptor = None
            return self._hash_value

    def __init(self) -> None:
        if isinstance(self.source, (str, Path, bytes)):
            if self._dt_mcs_input is not None:
                raise SGYInitParamsError("Argument 'dt_mcs' must be empty if SGY created from external sources")
            self._init_from_external()
            self.is_source_ndarray = False
        elif isinstance(self.source, np.ndarray):
            if self._dt_mcs_input is None:
                raise SGYInitParamsError("Argument 'dt_mcs' is required if nd.array is used as input")
            self._init_from_numpy()
            self.is_source_ndarray = True
        else:
            raise SGYInitParamsError("Only `str, Path, bytes, np.ndarray` types are available as input")
        self.units_converter = UnitsConverter(sgy_mcs=self.dt_mcs)

    def _init_from_numpy(self) -> None:
        assert self.source.ndim == 1 or self.source.ndim == 2, "Only arrays are available"  # type: ignore
        if self.source.ndim == 1:  # type: ignore
            self._ntr = 1
        elif self.source.ndim == 2:  # type: ignore
            self._ntr = self.source.shape[1]  # type: ignore
        else:
            raise ValueError("Only 1D or 2D arrays are available")

        self._traces = self.source  # type: ignore
        self._dt = self._dt_mcs_input  # type: ignore
        self._ns = self.source.shape[0]  # type: ignore

    def _init_from_external(self) -> None:
        self._descriptor = get_io(self.source, mode="rb")
        self._read_endianess()
        self._read_general_headers()
        self._read_traces_headers()
        self._scalar_raw_traces_headers()
        self._descriptor.close()
        self._descriptor = None

    def _read_endianess(self) -> None:
        num_bytes = self._descriptor.seek(0, 2)
        if num_bytes < 3600 + 240 + 1:  # at least general headers, 1 trace headers block and byte for 1 sample
            raise InvalidSGY("Invalid structure of SGY file. File is small")

        self._descriptor.seek(3224)
        value = self._descriptor.read(2)

        big = struct.unpack(">H", value)[0]
        little = struct.unpack("<H", value)[0]

        if 1 <= big <= 16 or 1 <= little <= 16:
            self._endianess = ">" if 1 <= big <= 16 else "<"
        else:
            raise InvalidSGY("Invalid endianess of SGY file")

    def _read_general_headers(self) -> None:
        gen_headers = {}
        for pointer, name, fmt in self._general_headers_schema.headers_schema:
            self._descriptor.seek(pointer)
            size = self._general_headers_schema.get_num_bytes(fmt)
            gen_headers[name] = struct.unpack(f"{self._endianess}{fmt}", self._descriptor.read(size))[0]
        self.general_headers = gen_headers

        self._ns = self.general_headers[self._general_headers_schema.ns_name]
        if self._ns < 1:
            raise InvalidSGY("Invalid number of samples")

        self._dt = self.general_headers[self._general_headers_schema.dt_name]
        if self._dt < 0:
            raise InvalidSGY("Invalid time discretization")

        self._data_fmt = self.general_headers[self._general_headers_schema.data_sample_format_name]
        if self._data_fmt < 1 or self._data_fmt > 6:
            raise NotImplementedReader(f"Not supported samples format '{self._data_fmt}'")
        self._bps = self.fmt2bps[self._data_fmt]

        num_bytes = self._descriptor.seek(0, 2)
        self._ntr = int((num_bytes - 3600) / (240 + self._ns * self._bps))
        if num_bytes != (3600 + (240 + self._ns * self._bps) * self._ntr):
            raise InvalidSGY("Invalid number of bytes")

    def _read_traces_headers(self) -> None:
        traces_headers = {}

        for offset, name, fmt in self._traces_headers_schema.headers_schema:
            header = self._read_custom_trace_header_with_existed_descriptor(offset, fmt)
            traces_headers[name] = header

        self._traces_headers_raw = pd.DataFrame(data=traces_headers)

    def _scalar_raw_traces_headers(self) -> None:
        self.traces_headers = self._traces_headers_raw.copy()
        for (
            scalar_from,
            apply_to_columns,
        ) in self._traces_headers_schema.scalar_from2apply.items():
            scalar = np.array(self._traces_headers_raw[scalar_from].copy())

            scalar[scalar == 0] = 1
            scalar[scalar < 0] = 1 / abs(scalar[scalar < 0])
            self.traces_headers[apply_to_columns] = self.traces_headers[apply_to_columns].apply(lambda x: scalar * x)

    def _read_custom_trace_header_with_existed_descriptor(self, byte_position: int, encoding: str) -> Tuple[Any, ...]:
        size = self._traces_headers_schema.get_num_bytes(encoding)
        buffer = []
        for idx in range(self._ntr):
            pointer = 3600 + (240 + self._ns * self._bps) * idx + byte_position
            self._descriptor.seek(pointer)
            buffer.append(self._descriptor.read(size))
        return struct.unpack(f"{self._endianess}{encoding * self._ntr}", b"".join(buffer))

    def read_custom_trace_header(self, byte_position: int, encoding: str) -> Tuple[Any, ...]:
        self._descriptor = get_io(self.source, mode="rb")
        result = self._read_custom_trace_header_with_existed_descriptor(byte_position, encoding)
        self._descriptor.close()
        self._descriptor = None
        return result

    def replace_traces(self, traces: np.ndarray) -> None:
        if traces.shape == self.shape:
            self._traces = traces

    def read(self, min_sample: Optional[int] = None, max_sample: Optional[int] = None) -> np.ndarray:
        ids = list(range(self.num_traces))
        traces = self.read_traces_by_ids(ids, min_sample, max_sample)
        self.replace_traces(traces)
        return traces

    def get_chunked_reader(
        self,
        chunk_size: int,
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        chunk_size = min(chunk_size, self.num_traces)
        all_ids = list(range(self.num_traces))

        for ids in chunk_iterable(all_ids, chunk_size):
            yield self.read_traces_by_ids(ids, min_sample, max_sample)

    def read_traces_by_ids(
        self,
        ids: Sequence[int],
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> np.ndarray:

        if min_sample is not None:
            if min_sample < 0 or not isinstance(min_sample, int):
                raise InvalidSamplesSlice("Invalid minimum slice index")
            min_sample = int(np.clip(min_sample, 0, self.num_samples))
        else:
            min_sample = 0

        if max_sample is not None:
            if max_sample < 1 or not isinstance(max_sample, int):
                raise InvalidSamplesSlice("Invalid maximum slice index")
            max_sample = int(np.clip(max_sample, 0, self.num_samples))
        else:
            max_sample = self.num_samples

        if min_sample >= max_sample:
            raise InvalidSamplesSlice("Minimum slice index is greater or equal to maximum index")

        len_slice = max_sample - min_sample
        ids = [idx for idx in ids if idx < self.num_traces]

        if len(ids) == 0:
            raise ValueError("The requested IDs were not found in the file")

        if self.is_source_ndarray or (self._traces is not None and self._traces.shape == self.shape):
            return self._read_block_ndarray(ids, min_sample, len_slice)
        else:
            return self._read_block_external(ids, min_sample, len_slice)

    def _read_block_ndarray(self, ids: Sequence[int], min_sample: int, length_slice: int) -> np.ndarray:
        return self._traces[min_sample : min_sample + length_slice, ids]

    def _read_block_external(self, ids: Sequence[int], min_sample: int, length_slice: int) -> np.ndarray:
        buffer = []

        self._descriptor = get_io(self.source, mode="rb")
        for idx in ids:
            pointer = 3600 + 240 + (240 + self._ns * self._bps) * idx + min_sample
            self._descriptor.seek(pointer)
            buffer.append(self._descriptor.read(length_slice * self._bps))
        self._descriptor.close()
        self._descriptor = None

        buffer_tr = b"".join(buffer)
        traces = self._read_traces_from_buffer(buffer_tr, (length_slice, len(ids)))
        return traces

    def _read_traces_from_buffer(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        if self._data_fmt == 1:
            reader_func = self._read_traces_ibm
        elif self._data_fmt == 2:
            reader_func = self._read_traces_4b_compl_int
        elif self._data_fmt == 3:
            reader_func = self._read_traces_2b_compl_int
        elif self._data_fmt == 4:
            raise NotImplementedReader(
                "Not implemented 32-bit fixed point with gain values reader (format 4 of SGY specification)"
            )
        elif self._data_fmt == 5:
            reader_func = self._read_traces_float
        elif self._data_fmt == 6:
            reader_func = self._read_traces_double
        else:
            raise ValueError("Not supported format")

        return reader_func(buffer, shape)

    @staticmethod
    def _read_ibm(ibm: int) -> float:
        """
        Converts an IBM floating point number into IEEE format.
        :param: ibm - 32 bit unsigned integer: unpack('>L', f.read(4))
        """
        if ibm == 0:
            return 0.0
        sign = ibm >> 31 & 0x01
        exponent = ibm >> 24 & 0x7F
        mantissa = (ibm & 0x00FFFFFF) / float(pow(2, 24))
        return (1 - 2 * sign) * mantissa * pow(16, exponent - 64)

    def _read_traces_ibm(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        reader = np.vectorize(self._read_ibm)
        array = np.ndarray(shape, f"{self._endianess}u4", buffer, order="F")
        return reader(array)

    @staticmethod
    def _read_compl_int(value: int, num_bits: int) -> int:
        return value - int((value << 1) & 2**num_bits)

    def _read_traces_4b_compl_int(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        reader = np.vectorize(self._read_compl_int)
        array = np.ndarray(shape, f"{self._endianess}u4", buffer, order="F")
        return reader(array, 32)

    def _read_traces_2b_compl_int(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        reader = np.vectorize(self._read_compl_int)
        array = np.ndarray(shape, f"{self._endianess}u2", buffer, order="F")
        return reader(array, 16)

    def _read_traces_float(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        return np.ndarray(shape, f"{self._endianess}f4", buffer, order="F")

    def _read_traces_double(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        return np.ndarray(shape, f"{self._endianess}f8", buffer, order="F")

    def export_sgy_with_picks(
        self,
        output_fname: Union[str, Path],
        picks_in_mcs: List[float],
        byte_position: int = 236,
        encoding: Optional[str] = None,
        picks_unit: Optional[str] = "mcs",
    ) -> None:
        assert not self.is_source_ndarray, "Only true SGY can be used for importing picks"
        assert 0 <= byte_position <= 236, "Only 0-236 bytes can be ised for writing"
        assert len(picks_in_mcs) == self.num_traces, "Number of traces and picks differs"
        assert picks_unit in ["ms", "mcs", "sample"]

        Path(output_fname).parent.mkdir(exist_ok=True, parents=True)

        if isinstance(self.source, (str, Path)):
            if Path(self.source).resolve() != Path(output_fname).resolve():
                shutil.copyfile(str(self.source), str(output_fname))
        elif isinstance(self.source, bytes):
            with open(output_fname, "wb+") as f_output:
                f_output.write(self.source)
        else:
            raise TypeError("Invalid type of source data")

        if encoding is None:
            encoding = [
                pack_type
                for _, header, pack_type in self._traces_headers_schema.headers_schema
                if header == self._traces_headers_schema.fb_pick_default
            ][0]

        cast_to = float if encoding in ["f", "d"] else int

        if picks_unit == "ms":
            picks = self.units_converter.mcs2ms(picks_in_mcs, cast_to=cast_to)
        elif picks_unit == "mcs":
            picks = picks_in_mcs
        elif picks_unit == "sample":
            picks = self.units_converter.mcs2index(picks_in_mcs, cast_to=cast_to)
        else:
            raise ValueError("Unsupported 'picking unit'")

        self._descriptor = get_io(output_fname, mode="r+b")
        for idx, pick in enumerate(picks):  # type: ignore
            pointer = 3600 + (240 + self.num_samples * self._bps) * idx + byte_position
            pick_byte = struct.pack(f"{self._endianess}{encoding}", pick)
            self._descriptor.seek(pointer)
            self._descriptor.write(pick_byte)
        self._descriptor.close()
