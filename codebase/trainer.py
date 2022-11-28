from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import plotseis_batch, AvgMoving, Stopper
from metrics import calc_fb_and_metrics
from focal_loss import FocalLoss
from picker import Picker
from seis_dataset import SeisDataset


class Trainer:
    _picker: Picker
    _results_path: Path
    _train_set: SeisDataset
    _valid_set: SeisDataset
    _test_set: SeisDataset
    _device: torch.device
    _batch_size: int
    _lr: float
    _num_workers: int
    _freq_valid: int
    _visual: Dict[str, List[int]]
    _dt_ms: float
    _height_model: int
    _width_model: int
    _stopper: Stopper
    _weights: torch.Tensor

    _criterion: nn.Module
    _net_path: Path
    _tensorboard_path: Path
    _writer_tb: SummaryWriter
    _num_batch: int

    def __init__(self,
                 picker: Picker,
                 results_path: Path,
                 train_set: SeisDataset,
                 valid_set: SeisDataset,
                 test_set: SeisDataset,
                 device: torch.device,
                 batch_size: int,
                 lr: float,
                 num_workers: int,
                 freq_valid: int,
                 visual: Dict[str, List[int]],
                 dt_ms: float,
                 height_model: int,
                 width_model: int,
                 stopper: Stopper,
                 weights: torch.Tensor):

        self._picker = picker
        self._results_path = results_path
        self._train_set = train_set
        self._valid_set = valid_set
        self._test_set = test_set
        self._device = device
        self._batch_size = batch_size
        self._lr = lr
        self._num_workers = num_workers
        self._freq_valid = freq_valid

        self._visual = visual
        self._dt_ms = dt_ms
        self._height_model = height_model
        self._width_model = width_model
        self._stopper = stopper
        self._weights = weights

        # self._criterion = nn.CrossEntropyLoss(weight=self._weights).to(self._device)
        self._criterion = FocalLoss(alpha=self._weights, gamma=2)

        self._optimizer = torch.optim.Adam(picker.parameters(), lr=self._lr)

        self._net_path, self._tensorboard_path = self._results_path / 'net', self._results_path / 'tensorboard'

        for folder in [self._net_path, self._tensorboard_path]:
            folder.mkdir(exist_ok=True, parents=True)

        self._writer_tb = SummaryWriter(log_dir=str(self._tensorboard_path), flush_secs=20)
        self._picker.to(self._device)

        self._num_batch = 0

        self._correct_visual()
        self._freq_valid = min((self._freq_valid, len(self._train_set) // self._batch_size + 1))

    def train(self, num_epoch: int) -> Tuple[float, float]:
        iou_max = 0
        train_loader = DataLoader(self._train_set,
                                  batch_size=self._batch_size,
                                  num_workers=self._num_workers,
                                  shuffle=True)
        for epoch in range(num_epoch):
            train_tqdm = tqdm(train_loader, total=len(train_loader), desc=f'train_{epoch}')
            loss_train_avg = AvgMoving()
            for curr_batch, (images, labels, true_fb) in enumerate(train_tqdm):

                self._num_batch = curr_batch + epoch * len(train_loader)

                self._optimizer.zero_grad()
                images = images.to(self._device)
                labels = labels.to(self._device)
                outputs = self._picker(images)
                loss_train = self._criterion(outputs, labels.view(-1, self._height_model, self._width_model))
                loss_train.backward()
                self._optimizer.step()

                loss_train = loss_train.detach().cpu().item()
                loss_train_avg.add(loss_train)

                outputs = (softmax(outputs.detach(), dim=1))[:, 1, :, :]
                labels = labels.squeeze(1)

                iou_train, min_diff_fb_train, max_diff_fb_train, mean_diff_fb_train, pred_fb = \
                    calc_fb_and_metrics(outputs.clone(), labels, true_fb.to(self._device), self._dt_ms)

                if (curr_batch + 1) % self._visual['train'][0] == 0:
                    self.visualize(images, outputs.clone(), true_fb, pred_fb, self._visual['train'][1], 'train')

                self._writer_tb_add_metrics(min_diff_fb=min_diff_fb_train,
                                            max_diff_fb=max_diff_fb_train,
                                            mean_diff_fb=mean_diff_fb_train,
                                            iou=iou_train,
                                            loss=loss_train,
                                            mode='train')

                train_tqdm.set_postfix({'Avg train loss': round(loss_train_avg.avg, 4)})

                if self._freq_valid != 0 and (curr_batch + 1) % self._freq_valid == 0:
                    loss_valid, *_, iou_valid = self.testvalid('valid')
                    self._stopper.update(iou_valid)
                    if iou_valid > iou_max:
                        iou_max = iou_valid
                        meta = {'loss_train': round(loss_train_avg.avg, 4),
                                'loss_valid': round(loss_valid, 4),
                                'iou_valid': round(iou_valid, 4),
                                'epoch': epoch}
                        self._picker.save(self._net_path / 'best.pth', meta)

                    if self._stopper.is_need_stop():
                        break

            else:
                meta = {'loss_train': round(loss_train_avg.avg, 4),
                        'epoch': epoch}
                self._picker.save(self._net_path / f'net_{epoch}.pth', meta)
                continue
            break

        loss_test, *_, iou_test = self.testvalid('test')

        self._writer_tb.close()
        return loss_test, iou_test

    def testvalid(self, mode: str) -> Tuple[float, float, float, float, float]:
        assert any((mode == 'valid', mode == 'test'))

        loader = DataLoader(self._valid_set if mode == 'valid' else self._test_set,
                            batch_size=self._batch_size,
                            num_workers=self._num_workers,
                            shuffle=True)

        loss_avg = AvgMoving()
        min_diff_fb_avg = AvgMoving()
        max_diff_fb_avg = AvgMoving()
        mean_diff_fb_avg = AvgMoving()
        iou_avg = AvgMoving()

        self._picker.eval()

        with torch.no_grad():
            testvalid_tqdm = tqdm(loader, total=len(loader), desc=mode, leave=False)
            for curr_batch, (images, labels, true_fb) in enumerate(testvalid_tqdm):
                images = images.to(self._device)
                labels = labels.to(self._device)

                outputs = self._picker(images)

                loss = self._criterion(outputs, labels.view(-1, self._height_model, self._width_model))

                outputs = (softmax(outputs, dim=1))[:, 1, :, :]
                labels = labels.squeeze(1)

                iou, min_diff_fb, max_diff_fb, mean_diff_fb, pred_fb = \
                    calc_fb_and_metrics(outputs.clone(), labels, true_fb.to(self._device), self._dt_ms)

                if (curr_batch + 1) % self._visual[mode][0] == 0:
                    self.visualize(images, outputs.clone(), true_fb, pred_fb, self._visual[mode][1], mode)

                loss = loss.detach().cpu().item()

                loss_avg.add(loss)
                min_diff_fb_avg.add(min_diff_fb)
                max_diff_fb_avg.add(max_diff_fb)
                mean_diff_fb_avg.add(mean_diff_fb)
                iou_avg.add(iou)

        self._picker.train()

        self._writer_tb_add_metrics(min_diff_fb=min_diff_fb_avg.avg,
                                    max_diff_fb=max_diff_fb_avg.avg,
                                    mean_diff_fb=mean_diff_fb_avg.avg,
                                    iou=iou_avg.avg,
                                    loss=loss_avg.avg,
                                    mode=mode)

        return (loss_avg.avg,
                min_diff_fb_avg.avg,
                max_diff_fb_avg.avg,
                mean_diff_fb_avg.avg,
                iou_avg.avg)

    def visualize(self, images: torch.Tensor, outputs: torch.Tensor, true_fb: torch.Tensor,
                  pred_fb: np.ndarray, num_pic: int, mode: str) -> None:
        size_batch = list(images.size())[0]

        max_visual = min((num_pic, size_batch))
        idx_batch = np.random.choice(range(size_batch), max_visual, replace=False)

        images = images[idx_batch, 0, :, :].cpu().clone().numpy()
        outputs = outputs[idx_batch, :, :].cpu().clone().numpy()
        true_fb = true_fb[idx_batch, :].cpu().clone().numpy()
        pred_fb = pred_fb[idx_batch, :]

        fig = plotseis_batch(images,
                             picking_batch=true_fb,
                             add_picking_batch=pred_fb,
                             background_batch=outputs,
                             show=False)

        # fig.set_size_inches(11.04, 6.9)  # 13" 16:10
        fig.set_dpi(150)

        self._writer_tb.add_figure(mode, fig, self._num_batch, close=False)
        self._writer_tb.flush()

        plt.close(fig)

    def _writer_tb_add_metrics(self, min_diff_fb: float, max_diff_fb: float, mean_diff_fb: float, iou: float,
                               loss: float, mode: str) -> None:
        self._writer_tb.add_scalars('diff_fb',
                                    {f'min_diff_{mode}': min_diff_fb,
                                     f'max_diff_{mode}': max_diff_fb,
                                     f'mean_diff_{mode}': mean_diff_fb},
                                    self._num_batch)

        self._writer_tb.add_scalars('iou',
                                    {f'iou_{mode}': iou},
                                    self._num_batch)

        self._writer_tb.add_scalars('losses',
                                    {f'loss_{mode}': loss},
                                    self._num_batch)
        self._writer_tb.flush()

    def _correct_visual(self) -> None:
        sets = {'train': self._train_set, 'valid': self._valid_set, 'test': self._test_set}
        for mode in self._visual:
            freq_visual = min((self._visual[mode][0], len(sets[mode]) // self._batch_size + 1))
            num_visual = min((self._visual[mode][1], self._batch_size, len(sets[mode])))
            self._visual[mode] = [freq_visual, num_visual]
