import random
from math import ceil
from typing import Tuple, List, Any, Union, Optional

import numpy as np
import matplotlib
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def split_dataset(path: Path, fracs: Tuple[int, int, int]) -> Tuple[List[Path], List[Path], List[Path]]:
    filenames = list(path.glob('*.npy'))
    train_num = ceil(len(filenames) * fracs[0])
    valid_num = ceil(len(filenames) * fracs[1])
    test_num = ceil(len(filenames) * fracs[2])

    if train_num + valid_num + test_num > len(filenames):
        raise ValueError('Invalid fracs')

    if 0 in [train_num, valid_num, test_num]:
        raise ValueError('Insufficient size of dataset for split with such fractions')

    shuffled_names = random.choices(filenames, k=len(filenames))

    train_set, valid_set, test_set = shuffled_names[:train_num], \
                                     shuffled_names[train_num:train_num + valid_num], \
                                     shuffled_names[train_num + valid_num:]
    return train_set, valid_set, test_set


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AvgMoving:
    n: int
    abg: float

    def __init__(self):
        self.n = 0
        self.avg = 0

    def add(self, val: float) -> None:
        self.n += 1
        self.avg = val / self.n + (self.n - 1) / self.n * self.avg


class AvgMovingVector:
    n: np.ndarray
    avg: np.ndarray

    def __init__(self, num_elem: int):
        self.n = np.zeros(num_elem)
        self.avg = np.zeros(num_elem)

    def add(self, val: np.ndarray, idx: np.ndarray) -> None:
        if val.ndim == 1 and idx.ndim == 1 and np.shape(val) == np.shape(idx):
            self._add_one_vec(val, idx)
        if val.ndim == 2 and idx.ndim == 2 and np.shape(val) == np.shape(idx):
            for batch in range(np.shape(val)[0]):
                self._add_one_vec(val[batch, :], idx[batch, :])

    def _add_one_vec(self, val: np.ndarray, idx: np.ndarray) -> None:
        self.n[idx] += 1
        self.avg[idx] = val / self.n[idx] + (self.n[idx] - 1) / self.n[idx] * self.avg[idx]


class Stopper:
    max_wrongs: int
    n_obs_wrongs: int
    delta: float
    best_value: float

    def __init__(self, max_wrongs: int, delta: float):
        assert max_wrongs > 1 and delta > 0
        self.max_wrongs = max_wrongs
        self.n_obs_wrongs = 0
        self.delta = delta
        self.best_value = 0

    def update(self, new_value: float) -> None:
        if new_value - self.best_value < self.delta or new_value < self.best_value:
            self.n_obs_wrongs += 1
        else:
            self.n_obs_wrongs = 0
            self.best_value = new_value

    def is_need_stop(self) -> bool:
        return self.n_obs_wrongs >= self.max_wrongs


def sinc_interp(x: np.ndarray, t_prev: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    shape_x = np.shape(x)
    period = t_prev[1] - t_prev[0]

    if len(shape_x) == 1:
        t_prev = np.reshape(t_prev, (1, np.size(t_prev)))
        t_new = np.reshape(t_new, (1, np.size(t_new)))
        time_matrix = np.tile(t_new, (len(t_prev), 1)) - np.tile(t_prev.transpose(), (1, len(t_new)))
        return np.dot(x, np.sinc(time_matrix / period))
    elif shape_x[0] == 1 and shape_x[1] > 1:
        t_prev = np.reshape(t_prev, (1, np.size(t_prev)))
        t_new = np.reshape(t_new, (1, np.size(t_new)))
        time_matrix = np.tile(t_new, (len(t_prev), 1)) - np.tile(t_prev.transpose(), (1, len(t_new)))
        return np.dot(x, np.sinc(time_matrix / period))
    elif shape_x[0] > 1 and shape_x[1] == 1:
        t_prev = np.reshape(t_prev, (np.size(t_prev), 1))
        t_new = np.reshape(t_new, (np.size(t_new), 1))
        time_matrix = np.tile(t_new, (1, len(t_prev))) - np.tile(t_prev.transpose(), (len(t_new), 1))
        return np.dot(np.sinc(time_matrix / period), x)


def sinc_interp_factor(x: np.ndarray, factor: int) -> np.ndarray:
    num_elem = np.max(np.shape(x))
    t_prev = np.linspace(0, num_elem - 1, num_elem)
    t_new = np.linspace(0, num_elem - 1, (num_elem - 1) * factor + 1)
    return sinc_interp(x, t_prev, t_new)


def data_normalize_and_limiting(data: np.ndarray) -> np.ndarray:
    norma = np.max(np.abs(data), axis=0)
    norma[np.abs(norma) < 1e-9 * np.max(np.abs(norma))] = 1
    data = data / norma * 2
    data[data < -1] = -1
    data[data > 1] = 1
    return data


def plotseis(data: np.ndarray,
             picking: Optional[np.ndarray] = None,
             add_picking: Optional[np.ndarray] = None,
             normalizing: Union[str, int] = 'entire',
             clip: float = 0.9,
             ampl: float = 1.0,
             patch: bool = True,
             colorseis: bool = False,
             wiggle: bool = True,
             background: Optional[np.ndarray] = None,
             colorbar: bool = False,
             dt: float = 1.0,
             show: bool = True) -> matplotlib.figure.Figure:

    num_time, num_trace = np.shape(data)

    if normalizing == 'indiv':
        norm_factor = np.max(np.abs(data), axis=0)
        norm_factor[np.abs(norm_factor) < 1e-9 * np.max(np.abs(norm_factor))] = 1
    elif normalizing == 'entire':
        norm_factor = np.tile(np.max(np.abs(data)), (1, num_trace))
    elif np.size(normalizing) == 1 and not None:
        norm_factor = np.tile(normalizing, (1, num_trace))
    elif np.size(normalizing) == num_trace:
        norm_factor = np.reshape(normalizing, (1, num_trace))
    else:
        raise ValueError('Wrong value of "normalizing"')

    data = data / norm_factor * ampl

    mask_overflow = np.abs(data) > clip
    data[mask_overflow] = np.sign(data[mask_overflow]) * clip

    data_time = np.tile((np.arange(num_time) + 1)[:, np.newaxis], (1, num_trace)) * dt

    fig, ax = plt.subplots()

    plt.xlim((0, num_trace + 1))
    plt.ylim((0, num_time * dt))
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    if wiggle:
        data_to_wiggle = data + (np.arange(num_trace) + 1)[np.newaxis, :]

        ax.plot(data_to_wiggle, data_time,
                color=(0, 0, 0))

    if colorseis:
        if not (wiggle or patch):
            ax.imshow(data,
                      aspect='auto',
                      interpolation='bilinear',
                      alpha=1,
                      extent=(1, num_trace, (num_time - 0.5) * dt, -0.5 * dt),
                      cmap='gray')
        else:
            ax.imshow(data,
                      aspect='auto',
                      interpolation='bilinear',
                      alpha=1,
                      extent=(-0.5, num_trace + 2 - 0.5, (num_time - 0.5) * dt, -0.5 * dt),
                      cmap='gray')

    if patch:
        data_to_patch = data
        data_to_patch[data_to_patch < 0] = 0

        for k_trace in range(num_trace):
            patch_data = ((data_to_patch[:, k_trace] + k_trace + 1)[:, np.newaxis],
                          data_time[:, k_trace][:, np.newaxis])
            patch_data = np.hstack(patch_data)

            head = np.array((k_trace + 1, 0))[np.newaxis, :]
            tail = np.array((k_trace + 1, num_time * dt))[np.newaxis, :]
            patch_data = np.vstack((head, patch_data, tail))

            polygon = Polygon(patch_data,
                              closed=True,
                              facecolor='black',
                              edgecolor=None)
            ax.add_patch(polygon)

    if picking is not None:
        ax.plot(np.arange(num_trace) + 1, picking * dt,
                linewidth=1,
                color='blue')

    if add_picking is not None:
        ax.plot(np.arange(num_trace) + 1, add_picking * dt,
                linewidth=1,
                color='green')

    if background is not None:
        bg = ax.imshow(background,
                       aspect='auto',
                       extent=(0.5, num_trace + 1 - 0.5, (num_time - 0.5) * dt, -0.5 * dt),
                       cmap='YlOrRd')

        if colorbar:
            plt.colorbar(mappable=bg)

    if show:
        plt.show()
    return fig


def plotseis_batch(data_batch: np.ndarray,
                   picking_batch: Optional[np.ndarray] = None,
                   add_picking_batch: Optional[np.ndarray] = None,
                   normalizing: Union[str, int] = 'entire',
                   clip: float = 0.9,
                   ampl: float = 1.0,
                   patch: bool = True,
                   colorseis: bool = False,
                   wiggle: bool = True,
                   background_batch: Optional[np.ndarray] = None,
                   colorbar: bool = False,
                   dt: float = 1,
                   show: float = True) -> matplotlib.figure.Figure:

    *num_batch, num_time, num_trace = np.shape(data_batch)
    assert len(num_batch) == 1

    num_batch = num_batch[0]

    fig = plt.figure()

    if num_batch == 1:
        num_col = 1
        num_row = 1

    else:
        num_col = np.floor(np.sqrt(num_batch))
        num_col = int((num_col if num_col == np.sqrt(num_batch) else num_col + 1))
        num_row = np.array(num_batch // num_col)
        num_row = int((num_row if num_batch / num_col == num_row else num_row + 1))

    gs = fig.add_gridspec(num_row, num_col)

    for batch in range(num_batch):
        idx = np.unravel_index(batch, (num_row, num_col), order='C')
        ax = fig.add_subplot(gs[idx])
        data = data_batch[batch, :]

        if normalizing == 'indiv':
            norm_factor = np.max(np.abs(data), axis=0)
            norm_factor[np.abs(norm_factor) < 1e-9 * np.max(np.abs(norm_factor))] = 1
        elif normalizing == 'entire':
            norm_factor = np.tile(np.max(np.abs(data)), (1, num_trace))
        elif np.size(normalizing) == 1 and not None:
            norm_factor = np.tile(normalizing, (1, num_trace))
        elif np.size(normalizing) == num_trace:
            norm_factor = np.reshape(normalizing, (1, num_trace))
        else:
            raise ValueError('Wrong value of "normalizing"')

        data = data / norm_factor * ampl

        mask_overflow = np.abs(data) > clip
        data[mask_overflow] = np.sign(data[mask_overflow]) * clip

        data_time = np.tile((np.arange(num_time) + 1)[:, np.newaxis], (1, num_trace)) * dt

        plt.xlim((0, num_trace + 1))
        plt.ylim((0, num_time * dt))
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        if wiggle:
            data_to_wiggle = data + (np.arange(num_trace) + 1)[np.newaxis, :]

            ax.plot(data_to_wiggle, data_time,
                    color=(0, 0, 0))

        if colorseis:
            ax.imshow(data,
                      aspect='auto',
                      interpolation='bilinear',
                      alpha=1,
                      extent=(-0.5, num_trace + 2 - 0.5, (num_time - 0.5) * dt, -0.5 * dt),
                      cmap='gray')

        if patch:
            data_to_patch = data
            data_to_patch[data_to_patch < 0] = 0

            for k_trace in range(num_trace):
                patch_data = ((data_to_patch[:, k_trace] + k_trace + 1)[:, np.newaxis],
                              data_time[:, k_trace][:, np.newaxis])
                patch_data = np.hstack(patch_data)

                head = np.array((k_trace + 1, 0))[np.newaxis, :]
                tail = np.array((k_trace + 1, num_time * dt))[np.newaxis, :]
                patch_data = np.vstack((head, patch_data, tail))

                polygon = Polygon(patch_data,
                                  closed=True,
                                  facecolor='black',
                                  edgecolor=None)
                ax.add_patch(polygon)

        if picking_batch is not None:
            picking = picking_batch[batch, :]
            if picking is not None:
                ax.plot(np.arange(num_trace) + 1, picking * dt,
                        linewidth=1,
                        color='blue')

        if add_picking_batch is not None:
            add_picking = add_picking_batch[batch, :]
            if add_picking is not None:
                ax.plot(np.arange(num_trace) + 1, add_picking * dt,
                        linewidth=1,
                        color='green')

        if background_batch is not None:
            background = background_batch[batch, :]
            if background is not None:
                bg = ax.imshow(background,
                               aspect='auto',
                               extent=(0.5, num_trace + 1 - 0.5, (num_time - 0.5) * dt, -0.5 * dt),
                               cmap='Wistia')

                if colorbar:
                    plt.colorbar(mappable=bg)
    if show:
        plt.show()
    return fig


