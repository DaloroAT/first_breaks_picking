import os.path

import numpy as np
from scipy.signal import chirp
from pathlib import Path

from utils import sinc_interp_factor


class SeisModel:

    def __init__(self, smoothness: int = 5,
                 height: int = 1000,
                 width: int = 24,
                 min_thickness: int = 20,
                 max_thickness: int = 60,
                 interp_signal: int = None,
                 filename_signal: str = ''):

        self.smoothness = smoothness
        self.height_model = height
        self.width_model = width
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.interp_signal = interp_signal
        self.filename_signal = filename_signal
        self.path_signals = self._prepare_path()

    def get_random_model(self):
        model, picking = self._generate_model()
        # cut segment of model according to "width_model"
        start_offset = np.random.randint(0, np.shape(model)[1] - self.width_model)
        model_output = model[:, start_offset:start_offset + self.width_model]
        picking_output = picking[start_offset:start_offset + self.width_model]
        return model_output, picking_output

    def _generate_model(self):
        # number of base points for varying depths
        num_base_points = (np.ceil(self.width_model / self.smoothness + 3)).astype(int)

        # width of variable depths model
        width = ((num_base_points - 1) * self.smoothness + 1).astype(int)

        min_start_sample = 10
        max_start_sample = 30
        range_depth = int(2 / 3 * self.max_thickness)
        vel_hyp = np.random.uniform(8.5e-3, 20.0e-3)  # best
        t0_hyp = np.random.randint(10, 30)  # best
        x0 = np.random.randint(0, width)  # best

        # height of 1 layer
        heights_layers = [np.random.randint(min_start_sample, max_start_sample)]

        # height randomizing of other layers
        while heights_layers[-1] <= self.height_model - 1:
            heights_layers.append(heights_layers[-1]+np.random.randint(self.min_thickness, self.max_thickness))
        heights_layers = np.array(heights_layers[:-2:])[:, np.newaxis]
        num_layers = len(heights_layers)

        layers_shifts = np.zeros((len(heights_layers), width))

        # create smoothed variable depths
        for k_layer in range(num_layers):
            variable_base_depth = np.random.randint(-range_depth, range_depth, (1, num_base_points))
            layers_shifts[k_layer, :] = sinc_interp_factor(variable_base_depth, self.smoothness)

        # create hyperbolic trend
        offset = np.arange(width) - x0
        hyp_trend = np.sqrt(t0_hyp ** 2 + (offset / (vel_hyp * self.smoothness)) ** 2)[np.newaxis, :]

        # each row of 'layers_model' is coordinates of layer
        layers_model = np.round(heights_layers + layers_shifts + hyp_trend)
        layers_model[layers_model > 1.5 * self.height_model] = 1.5 * self.height_model
        layers_model = layers_model.astype(int)
        max_depth = (np.max(layers_model) + 1).astype(int)
        picking = np.min(layers_model, axis=0)

        # create matrix of model with amplitudes
        model_spike = np.zeros((max_depth, width))

        for k_layer in range(num_layers):
            ampl = np.random.uniform(-1.0, -0.3) if np.random.random() > 0.5 else np.random.uniform(0.3, 1.0)
            ampl = ampl * np.random.uniform(0.8, 1.0, (1, width))

            model_spike[layers_model[k_layer, :], np.arange(width)] = \
                model_spike[layers_model[k_layer, :], np.arange(width)] + ampl

        signal = self._get_signal()

        # model with signals
        model_refl = np.zeros((max_depth + len(signal) - 1, width))

        for k_trace in range(width):
            model_refl[:, k_trace] = np.convolve(model_spike[:, k_trace], signal)

        model_refl = self._limit_size_model(model_refl)

        # create and add surface wave
        model_surf = self._create_surface_wave(x0, width)

        return model_refl + model_surf, picking

    def _create_surface_wave(self, x0, width):
        vel_linear = np.random.uniform(5.0e-4, 15.0e-4)  # best

        min_samples = np.ceil(0.5 * self.height_model)
        max_samples = np.ceil(2 / 3 * self.height_model)

        len_chirp = np.random.randint(min_samples, max_samples)

        # time delays vector of surface wave
        offset = np.arange(width) - x0
        delay_surf_spike = np.round(np.abs(offset) / vel_linear / self.smoothness).astype(int)
        height_surf_model_spike = (np.max(delay_surf_spike) + 1).astype(int)

        # matrix of surface spike model
        model_surf_spike = np.zeros((height_surf_model_spike, width))
        model_surf_spike[delay_surf_spike, np.arange(width)] = 1

        # model with surface wave signal
        model_surf_wave = np.zeros((height_surf_model_spike + len_chirp - 1, width))

        # envelope
        envel_surf = np.linspace(np.random.uniform(0.2, 0.5), np.random.uniform(0.5, 1.0), len_chirp)
        envel_surf = np.random.uniform(2.0, 6.0) * envel_surf

        time_chirp = np.linspace(0, 1.0, len_chirp)
        min_freq = np.random.uniform(1.0, 5.0)
        max_freq = np.random.uniform(5.0, 10.0)

        for k_trace in range(width):
            surface_wave = envel_surf * chirp(time_chirp, max_freq, time_chirp[-1], min_freq, phi=-90)
            model_surf_wave[:, k_trace] = np.convolve(model_surf_spike[:, k_trace], surface_wave)

        return self._limit_size_model(model_surf_wave)

    def _prepare_path(self):
        path = Path('signals').resolve()
        if not os.path.exists(path):
            err_path_str = 'There is no "signals" folder in: ' + str(Path.cwd())
            raise FileNotFoundError(err_path_str)
        filelist = list(path.glob('*.txt'))
        return filelist

    def _get_signal(self):
        signal = self._read_signal()
        if self.interp_signal is None or self.interp_signal == 1:
            return signal
        elif self.interp_signal > 1 and isinstance(self.interp_signal, int):
            return sinc_interp_factor(signal, self.interp_signal)
        elif self.interp_signal == -1:
            return sinc_interp_factor(signal, np.random.randint(1, 6))
        else:
            err_interp_str = '"interp_signal" must be integer or -1'
            raise ValueError(err_interp_str)

    def _read_signal(self):
        filename = self._get_signal_filename()
        return np.loadtxt(str(filename))

    def _get_signal_filename(self):
        # if 'self.filename_signal' is empty - change random signal, otherwise try find file by name
        if not self.filename_signal:
            return self.path_signals[np.random.randint(0, len(self.path_signals))]
        elif self.filename_signal:
            filename_signal = self.filename_signal + '.txt'
            find_matching = [self.path_signals[k].match(filename_signal) for k in range(len(self.path_signals))]
            idx_matching = [idx for idx, val in enumerate(find_matching) if val]
            if idx_matching:
                return self.path_signals[idx_matching[0]]
            else:
                err_matching_str = 'There is no "' + self.filename_signal + '.txt" file'
                raise FileNotFoundError(err_matching_str)

    def _limit_size_model(self, raw_model):
        height_raw = np.shape(raw_model)[0]
        if height_raw > self.height_model:
            return raw_model[:self.height_model:, :]
        elif height_raw < self.height_model:
            width = np.shape(raw_model)[1]
            return np.vstack((raw_model, np.zeros((self.height_model - height_raw, width))))
        else:
            return raw_model

