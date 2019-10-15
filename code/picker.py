from typing import Tuple, Any

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as trans_func
from torch.nn.functional import softmax
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from metrics import fb_calculation
from utils import data_normalize_and_limiting, AvgMovingVector, plotseis


class Picker(nn.Module):
    _net: nn.Module

    def __init__(self, net: nn.Module):
        super(Picker, self).__init__()
        self._net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._net(x)
        return x

    def calc_fb_and_prob(self, data_batch: np.ndarray, raw_data: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        *num_batch, num_time, num_trace = np.shape(data_batch)
        device = next(self._net.parameters()).device

        if not num_batch:
            num_batch = 1
            data_batch = data_batch[np.newaxis, :]
        else:
            num_batch = num_batch[0]

        if raw_data:
            data_tensor = torch.zeros((num_batch, 1, num_time, num_trace))

            for batch in range(num_batch):
                data = data_batch[batch, :]

                data = data_normalize_and_limiting(data)

                data_tensor[batch, 0, :] = torch.from_numpy(data)
        else:
            data_tensor = torch.from_numpy(data_batch).unsqueeze(1)

        data_tensor = data_tensor.float().to(device)

        prev_train_mode = self._net.training
        self._net.eval()
        with torch.no_grad():
            outputs = self._net(data_tensor)
            outputs = (func.softmax(outputs.detach(), dim=1))[:, 1, :, :]

            pred_fb = fb_calculation(outputs).cpu().numpy()
            outputs = outputs.cpu().numpy()

        self._net.train(prev_train_mode)

        return pred_fb, outputs

    def calc_fb_high_acc(self, seismogram: np.ndarray, batch_size: int, raw_data: bool = False) -> np.ndarray:
        data_set = BatchFormerWithMirroring(seismogram, raw_data)
        dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
        picking_vector = AvgMovingVector(np.shape(seismogram)[1])

        device = next(self._net.parameters()).device
        prev_train_mode = self._net.training
        self._net.eval()

        with torch.no_grad():
            for model, idx in dataloader:
                model = model.to(device)
                outputs = self._net(model)

                outputs = (func.softmax(outputs.detach(), dim=1))[:, 1, :, :]
                pred_fb = fb_calculation(outputs).cpu().numpy()

                picking_vector.add(pred_fb, idx.numpy())

        self._net.train(prev_train_mode)

        return picking_vector

    def save(self, path: Path, meta: Any = '') -> None:
        checkpoint = {
            'state_dict': self._net.state_dict(),
            'meta': meta
        }
        torch.save(checkpoint, path)

    def load(self, path: Path) -> None:
        device = next(self._net.parameters()).device
        checkpoint = torch.load(path)
        if isinstance(checkpoint, dict):
            self._net.load_state_dict(checkpoint['state_dict'])
            meta = checkpoint['meta']
        elif isinstance(checkpoint, nn.Module):
            self._net = checkpoint
            meta = {}
        else:
            raise ValueError('Wrong file')

        self._net.to(device)
        return meta


class BatchFormerWithMirroring(Dataset):
    data: np.ndarray
    init_idx_vec: np.ndarray
    len: int

    def __init__(self, seismogram: np.ndarray, raw_data: bool):
        super(BatchFormerWithMirroring).__init__()
        assert seismogram.ndim == 2 or np.shape(seismogram)[0] == 1000 or np.shape(seismogram)[1] < 24
        self.data = seismogram
        self.init_idx_vec = np.arange(np.shape(seismogram)[1])
        self.len = np.shape(seismogram)[1] + 23

        self._preparation(raw_data)

    def _preparation(self, raw_data: bool) -> None:
        if raw_data:
            self.data = data_normalize_and_limiting(self.data)
        idx_array = np.arange(np.shape(self.data)[1])
        self.data = np.hstack((self.data[:, 23:0:-1], self.data, self.data[:, -2:-25:-1]))
        self.init_idx_vec = np.hstack((idx_array[23:0:-1], idx_array, idx_array[-2:-25:-1]))

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        data_by_start_idx = trans_func.to_tensor(self.data[:, idx:idx + 24].copy())
        idx_from_init_gather = self.init_idx_vec[np.arange(idx, idx + 24)].copy()
        return data_by_start_idx, idx_from_init_gather

