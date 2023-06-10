import os
from pathlib import Path
from typing import Any, Optional, Tuple, Union, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

from first_breaks.models.unet3plus import Unet3Plus
from first_breaks.picking.task import Task
from first_breaks.picking.utils import preprocess_gather
from first_breaks.utils.utils import calc_hash, download_model_torch, is_cuda_available


def build_prod_model():
    return Unet3Plus(resnet_type='resnet18',
                     in_channels=1,
                     out_channels=3,
                     inter_channels=64,
                     pretrained=False)


class PickingDataset(Dataset):
    def __init__(self, task: Task):
        self.task = task
        self.idx2gather_ids = {idx: gather_ids for idx, gather_ids in enumerate(task.get_gathers_ids())}
        self.transform = ToTensor()

    def __len__(self) -> int:
        return len(self.idx2gather_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        gather_ids = self.idx2gather_ids[idx]
        amplitudes = np.array(
            [-1 if idx in self.task.traces_to_inverse else 1 for idx in range(len(gather_ids))], dtype=np.float32
        )
        gather = self.task.sgy.read_traces_by_ids(gather_ids)
        gather = preprocess_gather(gather, self.task.gain, self.task.clip)
        gather = amplitudes[None, :] * gather
        gather = gather[: self.task.maximum_time_sample, :]
        gather = self.transform(gather)

        return {"gather": gather, "gather_ids": torch.tensor(gather_ids)}


class PickerTorch:
    def __init__(self,
                 model_path: Optional[Union[str, Path]] = None,
                 segmentation_hw: Tuple[int, int] = (1024, 128),
                 show_progressbar: bool = True,
                 num_workers: int = int(os.cpu_count() * 2 / 3),
                 device: str = 'cuda' if is_cuda_available else 'cpu',
                 batch_size: int = 64):
        if model_path is None:
            model_path = download_model_torch()
        self.model_path = model_path
        self.model_hash = calc_hash(self.model_path)

        self.model = build_prod_model()
        ckpt = torch.load(self.model_path)
        self.model.load_state_dict(ckpt)
        self.model.eval()
        self.model.to(device)

        self.segmentation_hw = segmentation_hw
        self.num_workers = num_workers
        self.device = device
        self.batch_size = batch_size

        self.show_progressbar = show_progressbar
        self.progressbar: Optional[tqdm] = None

    def callback_processing_started(self, length: int) -> Any:
        if self.show_progressbar:
            self.progressbar = tqdm(desc="Picking", total=length)

    def callback_processing_finished(self) -> Any:
        if self.show_progressbar:
            self.progressbar.close()

    def callback_step_finished(self, step: int) -> Any:
        if self.show_progressbar:
            self.progressbar.update(step)

    @torch.no_grad()
    def process_task(self, task: Task) -> Task:
        self.model.eval()
        dataset = PickingDataset(task)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers)

        task_picks_in_sample = torch.zeros(task.sgy.num_traces).long()
        task_confidence = torch.zeros(task.sgy.num_traces)

        self.callback_processing_started(len(dataset))

        for batch_dict in dataloader:
            x = batch_dict['gather']
            x = x.to(self.device)

            src_hw = x.shape[2:]
            x = interpolate(x, self.segmentation_hw, mode='bicubic')
            x = self.model(x)
            x = interpolate(x, src_hw, mode='bicubic')

            preds = torch.softmax(x, 1)[:, 0, :, :]  # 0 channel - FB
            confs, picks = torch.max(preds, 1)

            indices = batch_dict['gather_ids']

            task_picks_in_sample[indices.flatten()] = picks.cpu().flatten().long()
            task_confidence[indices.flatten()] = confs.cpu().flatten()

            self.callback_step_finished(len(x))

        self.callback_processing_finished()

        task.success = True
        task.picks_in_samples = task_picks_in_sample.numpy().tolist()
        task.confidence = task_confidence.numpy().tolist()
        task.model_hash = self.model_hash

        return task
