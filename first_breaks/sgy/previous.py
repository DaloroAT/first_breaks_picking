import io
import struct
from pathlib import Path
from struct import unpack
from typing import Any, Dict, Generator, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from first_breaks.sgy.headers import FileHeaders, TraceHeaders

from first_breaks.const import PROJECT_ROOT
from first_breaks.sgy.interface import SizeHW, InvalidSamplesSlice, ISGY
from first_breaks.utils.utils import chunk_iterable, get_io, calc_hash


class NotImplementedReader(Exception):
    pass


class InvalidSGY(Exception):
    pass


class SGY(ISGY):
    def __init__(self, source: Union[str, Path, bytes]):
        self.descriptor = get_io(source, mode='rb')

        self.endianess: Optional[str] = None
        self.num_bytes: Optional[int] = None
        self._ns: Optional[int] = None  # number on data per trace
        self._dt: Optional[int] = None  # time discretization in mcs
        self._bps: Optional[int] = None  # bytes per sample
        self._ntr: Optional[int] = None  # number of traces in file

        self._general_headers_info = FileHeaders()
        self._traces_headers_info = TraceHeaders()
        self._general_headers: Optional[Dict[str, Any]] = None
        self._traces_headers_df: Optional[pd.DataFrame] = None

        try:
            self.endianess = self.get_endianess()
            self.read_general_headers()
            self.read_traces_headers()
        except Exception:
            raise InvalidSGY("Invalid structure of SGY file")

        self.check_headers()

    def get_bytes(self) -> bytes:
        self.descriptor.seek(0)
        return self.descriptor.read()

    def close(self):
        self.descriptor.close()

    @property
    def content_hash(self) -> str:
        return calc_hash(self.descriptor)

    def __hash__(self) -> int:
        return hash(self.content_hash)

    def check_headers(self) -> None:
        problems = []
        if self._ns < 1:
            problems.append("number of data")
        if self._ntr < 1:
            problems.append("number of traces")
        if self._data_fmt < 1 or self._data_fmt > 6:
            problems.append("data data format")
        if self._dt < 0:
            problems.append("time discretization")
        if self.num_bytes != (3600 + (240 + self._ns * self._bps) * self._ntr):
            problems.append("number of bytes")

        if problems:
            message = f'Invalid headers of SGY. Problems: {", ".join(problems)}.'
            raise InvalidSGY(message)

    @property
    def dt(self) -> int:
        return self._dt

    @property
    def ns(self) -> int:
        return self._ns

    @property
    def ntr(self) -> int:
        return self._ntr

    def get_endianess(self) -> str:
        self.descriptor.seek(3224)
        value = self.descriptor.read(2)

        big = struct.unpack(">H", value)[0]
        little = struct.unpack("<H", value)[0]

        if 1 <= big <= 16 or 1 <= little <= 16:
            return ">" if 1 <= big <= 16 else "<"
        else:
            raise InvalidSGY("Invalid structure of SGY file")

    def read_general_headers(self) -> None:
        gen_headers = {}
        for pointer, name, fmt in self._general_headers_info.headers_schema:
            self.descriptor.seek(pointer)
            size = self._general_headers_info.get_num_bytes(fmt)
            gen_headers[name] = unpack(f"{self.endianess}{fmt}", self.descriptor.read(size))[0]

        self.num_bytes = self.descriptor.seek(0, 2)

        self._general_headers = gen_headers

        self._ns = self._general_headers[self._general_headers_info.ns_name]
        self._dt = self._general_headers[self._general_headers_info.dt_name]
        self._init_bps()
        self._ntr = int((self.num_bytes - 3600) / (240 + self._ns * self._bps))

    def _init_bps(self) -> None:
        self._data_fmt = self._general_headers[self._general_headers_info.data_sample_format_name]

        bps = {
            1: 4,  # 4-byte hexadecimal exponent data (IBM single precision floating point)
            2: 4,  # 4-byte, two's complement integer
            3: 2,  # 2-byte, two's complement integer
            4: 4,  # 32-bit fixed point with gain values (Obsolete)
            5: 4,  # 4-byte, IEEE Floating Point
            6: 8,
        }  # 8-byte, IEEE Floating Point

        if self._data_fmt < 1 or self._data_fmt > 6:
            raise NotImplementedReader(f"Not supported format '{self._data_fmt}'")

        self._bps = bps[self._data_fmt]

    def read_traces_headers(self) -> None:
        traces_headers = {}

        for offset, name, fmt in self._traces_headers_info.headers_schema:
            size = self._traces_headers_info.get_num_bytes(fmt)
            buffer = []
            for idx in range(self._ntr):
                pointer = 3600 + (240 + self._ns * self._bps) * idx + offset
                self.descriptor.seek(pointer)
                buffer.append(self.descriptor.read(size))

            traces_headers[name] = unpack(f"{self.endianess}{fmt * self._ntr}", b"".join(buffer))

        self._traces_headers_df = pd.DataFrame(data=traces_headers)

    def read(self, min_sample: Optional[int] = None, max_sample: Optional[int] = None) -> np.ndarray:
        ids = list(range(self._ntr))
        return self.read_traces_by_ids(ids, min_sample, max_sample)

    def get_chunked_reader(
        self, chunk_size: int, min_sample: Optional[int] = None, max_sample: Optional[int] = None
    ) -> Generator[np.ndarray, None, None]:
        chunk_size = min(chunk_size, self._ntr)
        all_ids = list(range(self._ntr))

        for ids in chunk_iterable(all_ids, chunk_size):
            yield self.read_traces_by_ids(ids, min_sample, max_sample)

    def read_traces_by_ids(
        self, ids: Sequence[int], min_sample: Optional[int] = None, max_sample: Optional[int] = None
    ) -> np.ndarray:
        ids, min_sample, len_slice = self.parse_reading_params(ids=ids, min_sample=min_sample, max_sample=max_sample)

        buffer = []

        for idx in ids:
            pointer = 3600 + 240 + (240 + self._ns * self._bps) * idx + min_sample
            self.descriptor.seek(pointer)
            buffer.append(self.descriptor.read(len_slice * self._bps))

        buffer_tr = b"".join(buffer)
        traces = self._read_traces_from_buffer(buffer_tr, (len_slice, len(ids)))
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
    def read_ibm(ibm: int) -> float:
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
        reader = np.vectorize(self.read_ibm)
        array = np.ndarray(shape, f"{self.endianess}u4", buffer, order="F")
        return reader(array)

    @staticmethod
    def read_compl_int(value: int, num_bits: int) -> int:
        return value - int((value << 1) & 2**num_bits)

    def _read_traces_4b_compl_int(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        reader = np.vectorize(self.read_compl_int)
        array = np.ndarray(shape, f"{self.endianess}u4", buffer, order="F")
        return reader(array, 32)

    def _read_traces_2b_compl_int(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        reader = np.vectorize(self.read_compl_int)
        array = np.ndarray(shape, f"{self.endianess}u2", buffer, order="F")
        return reader(array, 16)

    def _read_traces_float(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        return np.ndarray(shape, f"{self.endianess}f4", buffer, order="F")

    def _read_traces_double(self, buffer: bytes, shape: SizeHW) -> np.ndarray:
        return np.ndarray(shape, f"{self.endianess}f8", buffer, order="F")

    def disp_load_info(self) -> None:
        print("Number of data per trace: %s" % self._ns)
        print("dt: %s ms" % (self._dt / 1000))
        print("Number of traces: %s" % self._ntr)
        print("Trace format: %s" % self._data_fmt)
        print("Bytes per sample: %s\n" % self._bps)


# class SGYPicks(SGY):
#     def __init__(self, *args, **kwargs):
#         self.picks: Optional[np.ndarray] = None
#         super().__init__(*args, **kwargs)
#
#     def get_picks(self):
#         dt = self.dt
#         return np.array([int(headers[-1] / dt) for headers in self.tr_hdr_struct])
#
#     def read_sgy(self, *args, **kwargs):
#         super().read(*args, **kwargs)
#         dt = self.dt
#         self.picks = np.array([int(headers[-1] / dt) for headers in self.tr_hdr_struct])
#
#     def export_sgy_with_picks(self, output_fname: Path, picks: List[int]):
#         output_fname.parent.mkdir(exist_ok=True, parents=True)
#         shutil.copyfile(str(self.fname), str(output_fname))
#
#         with open(output_fname, "r+b") as file_io:
#             for idx, pick in enumerate(picks):
#                 pointer = 3600 + (240 + self._ns * self._bps) * idx + 236
#                 pick_byte = struct.pack(f"{self.endianess}i", int(pick))
#                 file_io.seek(pointer)
#                 file_io.write(pick_byte)


if __name__ == "__main__":
    from first_breaks.utils.visualizations import plotseis
    sgy = SGY(PROJECT_ROOT / "data/real_gather.sgy")

    print(sgy.content_hash)
    gather = sgy.read()
    plotseis(gather, normalizing='indiv', ampl=0.5)
