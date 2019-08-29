from struct import *
import os.path
import os

import numpy as np
import pandas as pd
from pathlib import Path

from utils import plotseis


class SGY:

    def __init__(self):
        self.path_in = None
        self.path_out = None

        self.gen_hdr_dict = None  # general header info
        self.gen_hdr_fmt = None  # how to read general header

        self.tr_hdr_struct = None  # trace header info
        self.tr_hdr_fmt = None  # how to read trace header
        self.tr_data = None  # seismic data

        self._ns = None  # number on samples per trace
        self._dt = None  # time discretization
        self._bps = None  # bytes per sample
        self._ntr = None  # number of traces in file
        self._data_fmt = None  # traces format

        self._init_fields_header()

    def _init_fields_header(self):
        # file with keys and series
        filename_gen_hdr = 'file_header.csv'

        gen_frame = pd.read_csv(filename_gen_hdr, sep='\t')
        self.gen_hdr_fmt = '>' + "".join(gen_frame['type'].tolist())
        list_field_gen = gen_frame['field'].tolist()
        self.gen_hdr_dict = dict.fromkeys(list_field_gen)

        name_tr_hdr = 'trace_header.csv'
        tr_frame = pd.read_csv(name_tr_hdr, sep='\t')
        list_type_tr = tr_frame['type_big'].tolist()
        list_field_tr = tr_frame['field'].tolist()

        self.tr_hdr_fmt = list(zip(list_field_tr, list_type_tr))

    def read_sgy(self, path_in: str):
        path = Path(path_in).resolve()
        if not os.path.exists(path):
            raise FileNotFoundError('There is no file: ', path)

        self.path_in = path
        file_io = open(self.path_in, 'rb')
        gen_hdr_fromfile = unpack(self.gen_hdr_fmt, file_io.read(3600))
        for gen_idx, gen_key in enumerate(self.gen_hdr_dict):
            self.gen_hdr_dict[gen_key] = gen_hdr_fromfile[gen_idx]

        num_bytes = file_io.seek(0, 2)
        file_io.close()

        self._init_ns()
        self._init_dt()
        self._init_bps()

        self._ntr = int((num_bytes - 3600) / (240 + self._ns * self._bps))

        self._read_data()

        self.disp_load_info()

    def _init_ns(self, key_ns='ns'):
        self._ns = self.gen_hdr_dict[key_ns]

    def _init_dt(self, key_dt='dt'):
        self._dt = self.gen_hdr_dict[key_dt]

    def _init_bps(self, key_bps='data_sample_format'):
        if self.gen_hdr_dict[key_bps] == 0:
            raise ValueError('Unexpected data sample format')

        bps = {1: 4,  # 4-byte hexadecimal exponent data (IBM single precision floating point)
               2: 4,  # 4-byte, two's complement integer
               3: 2,  # 2-byte, two's complement integer
               4: 4,  # 32-bit fixed point with gain values (Obsolete)
               5: 4,  # 4-byte, IEEE Floating Point
               6: 8}  # 8-byte, IEEE Floating Point

        self._data_fmt = self.gen_hdr_dict[key_bps]

        if self._data_fmt < 1 or self._data_fmt > 6:
            raise ValueError('Not supported format')

        self._bps = bps[self._data_fmt]

    def _read_data(self):
        file_io = open(self.path_in, 'rb')
        file_io.seek(3600)  # skip textual header
        buffer_data_inp = file_io.read()
        # construct 'format_data_inp' as 'hdr_tr1 + data_tr1 + hdr_tr2 + data_tr2 + ...'
        format_data_inp = '>' + ('240s' + str(self._ns * self._bps) + 's') * self._ntr
        # unpack data to tuple buffer, where odd index - hdr, even - corresponding data
        buffer_data = unpack(format_data_inp, buffer_data_inp)
        file_io.close()
        if len(buffer_data) < 2:
            raise ValueError('Empty buffer for reading')

        del buffer_data_inp

        self.tr_hdr_struct = np.ndarray((self._ntr, 1), self.tr_hdr_fmt, b''.join(buffer_data[0::2]))

        self.tr_data = np.ndarray((self._ns * self._ntr * self._bps, 1), '>c', b''.join(buffer_data[1::2]), order='F')

        del buffer_data

        self._read_traces()

    def _read_traces(self):
        if self._data_fmt == 1:
            self._read_traces_ibm()
        elif self._data_fmt == 2:
            self._read_traces_4b_compl_int()
        elif self._data_fmt == 3:
            self._read_traces_2b_compl_int()
        elif self._data_fmt == 4:
            self._read_traces_4b_fixed_point_gain()
        elif self._data_fmt == 5:
            self._read_traces_float()
        elif self._data_fmt == 6:
            self._read_traces_double()
        else:
            raise ValueError('Not supported format')

    def _read_traces_ibm(self):
        pass

    def _read_traces_4b_compl_int(self):
        pass

    def _read_traces_2b_compl_int(self):
        pass

    def _read_traces_4b_fixed_point_gain(self):
        pass

    def _read_traces_float(self):
        self.tr_data = np.ndarray((self._ns, self._ntr), '>f', self.tr_data, order='F')

    def _read_traces_double(self):
        pass

    def disp_load_info(self):
        if self._ntr is not None:
            print('\nFile path: %s' % self.path_in)
            print('Number of samples per trace: %s' % self._ns)
            print('dt: %s ms' % (self._dt / 1000))
            print('Number of traces: %s' % self._ntr)
            print('Trace format: %s' % self._data_fmt)
            print('Bytes per sample: %s\n' % self._bps)
        else:
            err_empty = 'Data is not loaded'
            raise ValueError(err_empty)

    def get_traces(self, diap=None):
        if diap is None:
            return self.tr_data
        else:
            return self.tr_data[:, diap]

    def get_dt(self):
        return self._dt


if __name__ == '__main__':
    sgy = SGY()
    filename = './examples/real_gather.sgy'

    try:
        sgy.read_sgy(filename)
        data = sgy.get_traces()
        dt = sgy.get_dt() / 1e3

        plotseis(data, normalizing='indiv', colorseis=True, ampl=-2, dt=dt)
    except FileNotFoundError:
        print('File with example not found \nChange filename')

