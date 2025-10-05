from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort

from first_breaks.picking.ipicker import IPicker
from first_breaks.picking.picks import Picks
from first_breaks.picking.task import Task
from first_breaks.picking.utils import preprocess_gather
from first_breaks.utils.cuda import ONNX_DEVICE2PROVIDER, get_recommended_device
from first_breaks.utils.utils import (
    calc_hash,
    chunk_iterable,
    download_model_onnx,
    generate_color,
)


class IteratorOfTask:
    gather_key = "gather"
    gather_ids_key = "gather_ids"

    def __init__(self, task: Task):
        self.task = task
        self.idx2gather_ids = {idx: gather_ids for idx, gather_ids in enumerate(self.task.get_gathers_ids())}
        if self.task.normalize == "gather" and len(self.idx2gather_ids) > 1:
            raise AssertionError(
                "'gather' normalization can't be used for picking when number of gathers > 1. "
                "Use 'gather' normalization it only for visualization"
            )

    def __len__(self) -> int:
        return len(self.idx2gather_ids)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        gather_ids = self.idx2gather_ids[idx]
        amplitudes = np.array(
            [-1 if idx in self.task.traces_to_inverse else 1 for idx in range(len(gather_ids))],
            dtype=np.float32,
        )
        gather = self.task.sgy.read_traces_by_ids(gather_ids)

        gather = preprocess_gather(
            data=gather,
            gain=self.task.gain,
            clip=self.task.clip,
            normalize=self.task.normalize,
            f1_f2=self.task.f1_f2,
            f3_f4=self.task.f3_f4,
            fs=self.task.sgy.fs,
        )

        gather = amplitudes[None, :] * gather
        gather = gather[: self.task.maximum_time_sample, :]

        return {self.gather_key: gather, self.gather_ids_key: np.array(gather_ids)}

    def get_batch_generator(self, batch_size: int = 1) -> Generator[Dict[str, np.ndarray], None, None]:
        for ids in chunk_iterable(range(len(self)), batch_size):
            gather_batch = []
            gather_ids_batch = []
            for idx in ids:
                item = self[idx]
                gather_batch.append(item[self.gather_key][None, ...])
                gather_ids_batch.append(item[self.gather_ids_key])

            gather_batch = np.stack(gather_batch, axis=0)
            gather_ids_batch = np.hstack(gather_ids_batch)
            batch = {
                self.gather_key: gather_batch,
                self.gather_ids_key: gather_ids_batch,
            }
            yield batch


class PickerONNX(IPicker):
    OUTPUT_PICKS_KEY = "picks"
    OUTPUT_CONFS_KEY = "confs"
    OUTPUT_HEATMAP_KEY = "heatmap"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        show_progressbar: bool = True,
        device: str = get_recommended_device(),
        batch_size: int = 1,
    ):
        super().__init__(show_progressbar=show_progressbar)
        assert device in ["cpu", "cuda"]

        if model_path is None:
            model_path = download_model_onnx()
        self.model_path = model_path
        self.model_hash = calc_hash(self.model_path)

        self.device = device
        self.batch_size = batch_size

        self.model: Optional[ort.InferenceSession] = None
        self.init_model()

        self._available_outputs = sorted(o.name for o in self.model.get_outputs())
        self._input_name = self.model.get_inputs()[0].name

        for mandatory_key in [self.OUTPUT_PICKS_KEY, self.OUTPUT_CONFS_KEY]:
            assert mandatory_key in self._available_outputs, f"`mandatory_key` not found in model outputs `{self._available_outputs}`"

    def is_heatmap_available(self) -> bool:
        return self.OUTPUT_HEATMAP_KEY in self._available_outputs

    def init_model(self) -> None:
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.device == "cuda":
            sess_opt = ort.SessionOptions()
            sess_opt.intra_op_num_threads = 2
            sess_opt.inter_op_num_threads = 2
        self.model = ort.InferenceSession(
            str(self.model_path),
            providers=[ONNX_DEVICE2PROVIDER[self.device]],
            sess_options=sess_opt,
        )

    def change_settings(  # type: ignore
        self, *args: Any, device: Optional[str] = None, batch_size: Optional[int] = None
    ) -> "PickerONNX":
        if args:
            raise ValueError("Use named arguments instead of positional")

        if device and device != self.device:
            self.device = device
            self.init_model()

        if batch_size:
            self.batch_size = batch_size

        return self

    def pick_batch_of_gathers(self, gather: np.ndarray) -> Dict[str, np.ndarray]:
        assert gather.ndim == 4
        assert all(dim > 0 for dim in gather.shape)
        outputs_list = self.model.run(self._available_outputs, {self._input_name: gather})
        return dict(zip(self._available_outputs, outputs_list))

    def process_task(self, task: Task) -> Task:
        task_picks_in_sample = np.zeros(task.sgy.num_traces)
        task_confidence = np.zeros(task.sgy.num_traces)

        if self.is_heatmap_available():
            task_heatmap = np.zeros((task.sgy.num_samples, task.sgy.num_traces))
        else:
            task_heatmap = None

        task_iterator = IteratorOfTask(task)
        counter_step_finished = 0
        self.callback_processing_started(len(task_iterator))

        for idx, batch in enumerate(task_iterator.get_batch_generator(batch_size=self.batch_size)):
            self.interrupt_if_need()
            data = batch["gather"].astype(np.float32)
            results = self.pick_batch_of_gathers(data)

            picks = results[self.OUTPUT_PICKS_KEY]
            confidence = results[self.OUTPUT_CONFS_KEY]


            indices = batch["gather_ids"]

            task_picks_in_sample[indices.flatten()] = picks.flatten()  # (B, W) -> (BxW)
            task_confidence[indices.flatten()] = confidence.flatten()  # (B, W) -> (BxW)

            if self.is_heatmap_available():
                heatmap = results[self.OUTPUT_HEATMAP_KEY]
                heatmap = heatmap.transpose(1, 0, 2)  # (B, H, W) -> (H, B, W)
                heatmap = heatmap.reshape(heatmap.shape[0], -1)  # (H, B, W) -> (H, BxW)
                task_heatmap[: len(heatmap), indices.flatten()] = heatmap

            counter_step_finished += len(data)
            self.callback_step_finished(counter_step_finished)

        self.callback_processing_finished()

        picks = Picks(
            values=task_picks_in_sample.astype(int).tolist(),
            heatmap=task_heatmap,
            unit="sample",
            dt_mcs=task.sgy.dt_mcs,
            confidence=task_confidence.tolist(),
            modified_manually=False,
            created_manually=False,
            created_by_nn=True,
            color=generate_color(),
            picking_parameters=task.picking_parameters,
        )

        task.success = True
        task.picks = picks
        task.model_hash = self.model_hash
        task.assign_new_picks_id()

        self.need_interrupt = False

        return task
