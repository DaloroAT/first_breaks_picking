from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as t
from scipy.signal import ricker
from pathlib import Path
import matplotlib.pyplot as plt

from utils import data_normalize_and_limiting


class SeisDataset(Dataset):
    path_data: List[Path]
    height_model: int
    width_model: int
    _transforms: t.Compose

    def __init__(self, path_data: List[Path], height_model: int, width_model: int, prob_aug: float = 0.8):
        super(SeisDataset).__init__()
        self.path_data = path_data
        self.height_model = height_model
        self.width_model = width_model
        self._transforms = t.ToTensor()

        self.set_transforms(prob_aug)

    def __len__(self) -> int:
        return len(self.path_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        model, picking = _get_data(self.path_data[idx])

        model = (self._transforms(model)).float()
        picked_model, picking_new = self._get_picked_model(picking)
        return model, picked_model, picking_new

    def _get_picked_model(self, picking: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        pred_model = np.zeros((self.height_model, self.width_model))
        picking[picking < 0] = 0
        diap = 8
        for k in range(self.width_model):
            pick1 = (np.max(picking[k] - diap, 0)).astype(int)
            pick2 = picking[k]
            pred_model[pick1:pick2, k] = 2
            pred_model[pick2:, k] = 1
        return (t.functional.to_tensor(pred_model)).long(), picking

    def set_transforms(self, p: float) -> None:
        if p > 0:
            transforms_list = [self._add_random_const_transform,
                               self._ampl_inversion_transform,
                               self._sync_impulse_transform,
                               self._sine_distortion_transform,
                               self._energy_absorption_transform]
            random_transforms = t.RandomOrder([t.RandomApply([transform], p=p) for transform in transforms_list])

            self._transforms = t.Compose([random_transforms,
                                          self._add_random_noise_transform,
                                          t.RandomApply([self._ampl_zero_transform,
                                                         self._high_noise_trace_transform], p=p),
                                          data_normalize_and_limiting,
                                          t.ToTensor()])
        elif p == 0:
            self._transforms = t.Compose([data_normalize_and_limiting,
                                          t.ToTensor()])
        else:
            ValueError('Probability must be non-negative')

    def _add_random_const_transform(self, data: np.ndarray) -> np.ndarray:
        add_const = np.zeros(self.width_model)
        add_const[np.random.randint(0, self.width_model)] = np.random.uniform(-0.05, 0.05)
        return data + add_const[np.newaxis, :]

    def _add_random_noise_transform(self, data, max_ampl: float = 0.04) -> np.ndarray:
        return data + np.random.uniform(-max_ampl, max_ampl, (self.height_model, self.width_model))

    def _high_noise_trace_transform(self, data: np.ndarray) -> np.ndarray:
        rand_chan = np.random.randint(self.width_model)
        data[:, rand_chan] = np.random.uniform(-1, 1, self.height_model)
        return data

    def _ampl_inversion_transform(self, data: np.ndarray) -> np.ndarray:
        rand_chan = np.random.randint(self.width_model)
        data[:, rand_chan] = -data[:, rand_chan]
        return data

    def _ampl_zero_transform(self, data: np.ndarray) -> np.ndarray:
        data[:, np.random.randint(0, self.width_model)] = 0
        return data

    def _sync_impulse_transform(self, data: np.ndarray) -> np.ndarray:
        len_sync = 80
        alpha = np.random.uniform(1.0, 7.0)
        sync_impulse = ricker(len_sync, alpha)

        trace_with_sync = np.zeros(self.height_model - len_sync + 1)
        trace_with_sync[np.random.randint(0, np.ceil(self.height_model * 0.5))] = \
            np.random.uniform(-0.6, -0.3) if np.random.random() > 0.5 else np.random.uniform(0.3, 0.6)

        trace_with_sync = (np.convolve(trace_with_sync, sync_impulse))[:, np.newaxis]
        return data + trace_with_sync

    def _sine_distortion_transform(self, data: np.ndarray) -> np.ndarray:
        rand_channels = np.random.choice(range(self.width_model), 3, replace=False)
        sine = np.sin((np.arange(0, 1, self.height_model) * np.random.uniform(0.5, 1.0) + np.random.uniform())
                      * 2 * np.pi)
        sine *= np.random.uniform(-0.5, -0.2) if np.random.random() > 0.5 else np.random.uniform(0.2, 0.5)
        data[:, rand_channels] = data[:, rand_channels] + sine[:, np.newaxis]
        return data

    def _energy_absorption_transform(self, data: np.ndarray) -> np.ndarray:
        [xx, yy] = np.meshgrid(np.arange(self.width_model), np.arange(self.height_model))
        rand_chan = np.random.randint(self.width_model)
        spatial_absorp = np.random.randint(self.width_model // 2, self.width_model)
        time_absorp = np.random.randint(self.height_model // 2, self.height_model)
        absorp = np.exp(-((xx - rand_chan) / spatial_absorp) ** 2 - ((yy - self.height_model) / time_absorp) ** 2)
        absorp = 1 - absorp
        return data * absorp


# @lru_cache(maxsize=2**19)
def _get_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    return np.load(path, allow_pickle=True)
