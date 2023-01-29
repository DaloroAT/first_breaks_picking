import shutil
import struct
import time
from hashlib import md5
from pathlib import Path
from pprint import pprint
from struct import unpack
from typing import List, Optional, Sequence, Tuple

import numpy as np
from headers import FileHeadersOld, TraceHeadersOld, FileHeaders, TraceHeaders
import pandas as pd

from first_breaks.const import PROJECT_ROOT
from first_breaks.utils.utils import chunk_iterable
from first_breaks.utils.visualizations import plotseis


class NotConsistent(Exception):
    pass


class NotImplementedReader(Exception):
    pass


class InvalidSGY(Exception):
    pass


class FileNotSGY(Exception):
    pass


class InvalidSamplesSlice(Exception):
    pass


SizeHW = Tuple[int, int]


class SGY:
    def __init__(self, fname: Path, verbose: bool = False):
        fname = Path(fname).resolve()
        if not fname.exists():
            raise FileNotFoundError("There is no file: ", str(fname))

        self.fname = fname
        self.verbose = verbose

        self.gen_hdr_dict = None  # general header info
        self.gen_hdr_fmt = None  # how to read general header

        self.tr_hdr_struct = None  # trace header info
        self.tr_hdr_fmt = None  # how to read trace header
        self.endianess = None  # little of big endian of byte order

        self.num_bytes = None
        self._ns = None  # number on data per trace
        self._dt = None  # time discretization
        self._bps = None  # bytes per sample
        self._ntr = None  # number of traces in file
        self._data_fmt = None  # traces format

        self._general_headers_info = FileHeaders()
        self._traces_headers_info = TraceHeaders()
        self._general_headers = None
        self._traces_headers_df = None

        try:
            self.endianess = self.get_endianess()
            self.read_general_headers()
            self.read_traces_headers()
            self._init_fields_header()
            self.read_general_headers_old()
            self.read_trace_headers_old()
        except Exception:
            raise InvalidSGY("Invalid structure of SGY file")

        self.check_headers()

    def check_headers(self):
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

    def shape(self) -> SizeHW:
        return self._ns, self._ntr

    def get_endianess(self) -> str:
        with open(self.fname, "rb") as file_io:
            file_io.seek(3224)
            value = file_io.read(2)

        big = struct.unpack(">H", value)[0]
        little = struct.unpack("<H", value)[0]

        if 1 <= big <= 16 or 1 <= little <= 16:
            return ">" if 1 <= big <= 16 else "<"
        else:
            raise InvalidSGY("Invalid structure of SGY file")

    def _init_fields_header(self):
        self.endianess = self.get_endianess()
        print(self.endianess)
        self.gen_hdr_fmt = f"{self.endianess}3200siiiHHHHHHHHhHHHHhhHHHHHHHHH240sHHH94s"
        file_headers = FileHeadersOld()
        self.gen_hdr_dict = file_headers.get_headers()
        trace_headers = TraceHeadersOld()
        self.tr_hdr_fmt = trace_headers.get_headers()
        self.tr_hdr_fmt = [
            (pair[0], pair[1].replace(">", self.endianess).replace("<", self.endianess)) for pair in self.tr_hdr_fmt
        ]

    def read_general_headers(self):
        gen_headers = {}
        with open(self.fname, "rb") as file_io:
            for pointer, name, fmt in self._general_headers_info.headers:
                file_io.seek(pointer)
                size = self._general_headers_info.get_num_bytes(fmt)
                gen_headers[name] = unpack(f"{self.endianess}{fmt}", file_io.read(size))[0]

            self.num_bytes = file_io.seek(0, 2)

        self._general_headers = gen_headers
        self._init_ns()
        self._init_dt()
        self._init_bps()
        self._ntr = int((self.num_bytes - 3600) / (240 + self._ns * self._bps))

    def read_general_headers_old(self):
        with open(self.fname, "rb") as file_io:
            gen_hdr_fromfile = unpack(self.gen_hdr_fmt, file_io.read(3600))
            for gen_idx, gen_key in enumerate(self.gen_hdr_dict):
                self.gen_hdr_dict[gen_key] = gen_hdr_fromfile[gen_idx]

            self.num_bytes = file_io.seek(0, 2)

        self._init_ns_old()
        self._init_dt_old()
        self._init_bps_old()

        self._ntr = int((self.num_bytes - 3600) / (240 + self._ns * self._bps))

    # def read_general_headers(self):
    #     gen_headers = {}
    #     with open(self.fname, "rb") as file_io:
    #         for pointer, name, fmt in self._general_headers_info.headers:
    #             file_io.seek(pointer)
    #             size = self._general_headers_info.get_num_bytes(fmt)
    #             gen_headers[name] = unpack(f"{self.endianess}{fmt}", file_io.read(size))[0]
    #
    #         self.num_bytes = file_io.seek(0, 2)
    #
    #     self._general_headers = gen_headers
    #     self._init_ns()
    #     self._init_dt()
    #     self._init_bps()
    #     self._ntr = int((self.num_bytes - 3600) / (240 + self._ns * self._bps))

    def read_traces_headers(self):
        traces_headers = {}

        with open(self.fname, "rb") as file_io:
            for offset, name, fmt in self._traces_headers_info.headers:
                size = self._traces_headers_info.get_num_bytes(fmt)
                buffer = []
                for idx in range(self._ntr):
                    pointer = 3600 + (240 + self._ns * self._bps) * idx + offset
                    file_io.seek(pointer)
                    buffer.append(file_io.read(size))

                traces_headers[name] = unpack(f"{self.endianess}{fmt * self._ntr}", b"".join(buffer))

        self._traces_headers_df = pd.DataFrame(data=traces_headers)

        print(self._traces_headers_df)

    def read_trace_headers_old(self):
        with open(self.fname, "rb") as file_io:
            buffer = []

            for idx in range(self._ntr):
                pointer = 3600 + (240 + self._ns * self._bps) * idx
                file_io.seek(pointer)
                buffer.append(file_io.read(240))

        self.tr_hdr_struct = np.ndarray(self._ntr, self.tr_hdr_fmt, b"".join(buffer))

    def read(self, min_sample: Optional[int] = None, max_sample: Optional[int] = None):
        ids = list(range(self._ntr))
        return self.read_traces_by_ids(ids, min_sample, max_sample)

    def get_chunked_reader(self, chunk_size: int, min_sample: Optional[int] = None, max_sample: Optional[int] = None):
        chunk_size = min(chunk_size, self._ntr)
        all_ids = list(range(self._ntr))

        for ids in chunk_iterable(all_ids, chunk_size):
            yield self.read_traces_by_ids(ids, min_sample, max_sample)

    def read_traces_by_ids(
        self, ids: Sequence[int], min_sample: Optional[int] = None, max_sample: Optional[int] = None
    ):

        if min_sample is not None:
            if min_sample < 0 or not isinstance(min_sample, int):
                raise InvalidSamplesSlice("Invalid minimum slice index")
            min_sample = int(np.clip(min_sample, 0, self._ns))
        else:
            min_sample = 0

        if max_sample is not None:
            if max_sample < 1 or not isinstance(max_sample, int):
                raise InvalidSamplesSlice("Invalid maximum slice index")
            max_sample = int(np.clip(max_sample, 0, self._ns))
        else:
            max_sample = self._ns

        if min_sample >= max_sample:
            raise InvalidSamplesSlice("Minimum slice index is greater or equal to maximum index")

        len_slice = max_sample - min_sample
        ids = [idx for idx in ids if idx < self._ntr]

        if len(ids) == 0:
            return None

        with open(self.fname, "rb") as file_io:
            buffer = []

            for idx in ids:
                pointer = 3600 + 240 + (240 + self._ns * self._bps) * idx + min_sample
                file_io.seek(pointer)
                buffer.append(file_io.read(len_slice * self._bps))

        buffer = b"".join(buffer)
        traces = self._read_traces_from_buffer(buffer, (len_slice, len(ids)))
        return traces

    def export_sgy_with_picks(self, output_fname: Path, picks: List[int]):
        output_fname.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(str(self.fname), str(output_fname))

        with open(output_fname, "r+b") as file_io:
            for idx, pick in enumerate(picks):
                pointer = 3600 + (240 + self._ns * self._bps) * idx + 236
                pick_byte = struct.pack(f"{self.endianess}i", int(pick))
                file_io.seek(pointer)
                file_io.write(pick_byte)

    def _init_ns(self):
        self._ns = self._general_headers[self._general_headers_info.ns_name]

    def _init_ns_old(self, key_ns="ns"):
        self._ns = self.gen_hdr_dict[key_ns]

    def _init_dt(self):
        self._dt = self._general_headers[self._general_headers_info.dt_name]

    def _init_dt_old(self, key_dt="dt"):
        self._dt = self.gen_hdr_dict[key_dt]

    def _init_bps(self):
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

    def _init_bps_old(self, key_bps="data_sample_format"):
        if self.gen_hdr_dict[key_bps] == 0:
            raise ValueError("Unexpected data sample format")

        bps = {
            1: 4,  # 4-byte hexadecimal exponent data (IBM single precision floating point)
            2: 4,  # 4-byte, two's complement integer
            3: 2,  # 2-byte, two's complement integer
            4: 4,  # 32-bit fixed point with gain values (Obsolete)
            5: 4,  # 4-byte, IEEE Floating Point
            6: 8,
        }  # 8-byte, IEEE Floating Point

        self._data_fmt = self.gen_hdr_dict[key_bps]

        if self._data_fmt < 1 or self._data_fmt > 6:
            raise ValueError("Not supported format")

        self._bps = bps[self._data_fmt]

    def _read_traces_from_buffer(self, buffer: bytes, shape: SizeHW):
        if self._data_fmt == 1:
            reader_func = self._read_traces_ibm
        elif self._data_fmt == 2:
            reader_func = self._read_traces_4b_compl_int
        elif self._data_fmt == 3:
            reader_func = self._read_traces_2b_compl_int
        elif self._data_fmt == 4:
            reader_func = self._read_traces_4b_fixed_point_gain
        elif self._data_fmt == 5:
            reader_func = self._read_traces_float
        elif self._data_fmt == 6:
            reader_func = self._read_traces_double
        else:
            raise ValueError("Not supported format")

        return reader_func(buffer, shape)

    @staticmethod
    def read_ibm(ibm):
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

    def _read_traces_ibm(self, buffer: bytes, shape: SizeHW):
        reader = np.vectorize(self.read_ibm)
        array = np.ndarray(shape, f"{self.endianess}u4", buffer, order="F")
        return reader(array)

    @staticmethod
    def read_compl_int(value: int, num_bits: int):
        return value - int((value << 1) & 2**num_bits)

    def _read_traces_4b_compl_int(self, buffer: bytes, shape: SizeHW):
        reader = np.vectorize(self.read_compl_int)
        array = np.ndarray(shape, f"{self.endianess}u4", buffer, order="F")
        return reader(array, 32)

    def _read_traces_2b_compl_int(self, buffer: bytes, shape: SizeHW):
        reader = np.vectorize(self.read_compl_int)
        array = np.ndarray(shape, f"{self.endianess}u2", buffer, order="F")
        return reader(array, 16)

    @staticmethod
    def _read_traces_4b_fixed_point_gain(*args, **kwargs):
        message = "Not implemented 32-bit fixed point with gain values reader (format 4 of SGY specification)"
        raise NotImplementedReader(message)

    def _read_traces_float(self, buffer: bytes, shape: SizeHW):
        return np.ndarray(shape, f"{self.endianess}f4", buffer, order="F")

    def _read_traces_double(self, buffer: bytes, shape: SizeHW):
        return np.ndarray(shape, f"{self.endianess}f8", buffer, order="F")

    def disp_load_info(self):
        if self._ntr is not None:
            print("\nFile path: %s" % self.fname)
            print("Number of data per trace: %s" % self._ns)
            print("dt: %s ms" % (self._dt / 1000))
            print("Number of traces: %s" % self._ntr)
            print("Trace format: %s" % self._data_fmt)
            print("Bytes per sample: %s\n" % self._bps)
        else:
            err_empty = "Data is not loaded"
            raise ValueError(err_empty)

    def get_dt(self):
        return self._dt

    @property
    def dt(self):
        return self.get_dt()


class SGYPicks(SGY):
    def __init__(self, *args, **kwargs):
        self.picks: Optional[np.ndarray] = None
        super().__init__(*args, **kwargs)

    def get_picks(self):
        dt = self.get_dt()
        return np.array([int(headers[-1] / dt) for headers in self.tr_hdr_struct])

    def read_sgy(self, *args, **kwargs):
        super().read(*args, **kwargs)
        dt = self.get_dt()
        self.picks = np.array([int(headers[-1] / dt) for headers in self.tr_hdr_struct])


if __name__ == "__main__":
    sgy = SGY(PROJECT_ROOT / "data/real_gather.sgy")
    # gather = sgy.read()
    # plotseis(gather, normalizing="indiv", dpi=200)
