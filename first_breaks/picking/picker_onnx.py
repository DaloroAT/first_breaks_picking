from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader

from first_breaks.picking.ipicker import IPicker
from first_breaks._pytorch.picker_torch import PickingDataset
from first_breaks.picking.task import Task
from first_breaks.utils.utils import calc_hash, download_model_onnx, is_onnx_cuda_available


class PickerONNX(IPicker):
    def __init__(self,
                 model_path: Optional[Union[str, Path]] = None,
                 show_progressbar: bool = True,
                 device: str = 'cuda' if is_onnx_cuda_available() else 'cpu',
                 batch_size: int = 1):
        super().__init__(show_progressbar=show_progressbar)
        if model_path is None:
            model_path = download_model_onnx()
        self.model_path = model_path
        self.model_hash = calc_hash(self.model_path)
        self.batch_size = batch_size
        # print(_pybind_state.get_available_providers())
        # print(Session().get_providers())
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = 2
        sess_opt.inter_op_num_threads = 2
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        print(ort.get_available_providers())
        self.model = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"],
                                          sess_options=sess_opt
                                          )
        print(self.model.get_providers())

    def pick_gather(self, gather: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert gather.ndim == 4
        assert all(dim > 0 for dim in gather.shape)
        outputs = self.model.run(None, {"input": gather})
        return outputs[0], outputs[1]

    def process_task(self, task: Task) -> Task:
        # self.model.eval()
        dataset = PickingDataset(task)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=0,)

        task_picks_in_sample = np.zeros(task.sgy.num_traces)
        task_confidence = np.zeros(task.sgy.num_traces)

        self.callback_processing_started(len(dataset))
        counter_step_finished = 0

        for batch_dict in dataloader:
            data = batch_dict['gather'].numpy()
            picks, confidence = self.pick_gather(data)
            indices = batch_dict['gather_ids']

            task_picks_in_sample[indices.flatten()] = picks.flatten()
            task_confidence[indices.flatten()] = confidence.flatten()

            counter_step_finished += len(data)
            self.callback_step_finished(counter_step_finished)

        self.callback_processing_finished()

        task.success = True
        task.picks_in_samples = task_picks_in_sample.tolist()
        task.confidence = task_confidence.tolist()
        task.model_hash = self.model_hash

        return task


    # def pick_gather(self, gather: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     assert gather.ndim == 2
    #     assert all(dim > 0 for dim in gather.shape)
    #     outputs = self.model.run(None, {"input": gather[None, None, ...]})
    #     return outputs[0][0], outputs[1][0]
    #
    # def process_task(self, task: Task) -> Task:
    #     task_picks_in_sample = []
    #     task_confidence = []
    #
    #     self.callback_processing_started(task.num_gathers)
    #
    #     for step, gather_ids in enumerate(task.get_gathers_ids()):
    #         amplitudes = np.array(
    #             [-1 if idx in task.traces_to_inverse else 1 for idx in range(len(gather_ids))], dtype=np.float32
    #         )
    #
    #         gather = task.sgy.read_traces_by_ids(gather_ids)
    #
    #         gather = preprocess_gather(gather, task.gain, task.clip)
    #         gather = amplitudes[None, :] * gather
    #         gather = gather[: task.maximum_time_sample, :]
    #
    #         picks, confidence = self.pick_gather(gather)
    #
    #         task_picks_in_sample.extend(picks.tolist())
    #         task_confidence.extend(confidence.tolist())
    #
    #         self.callback_step_finished(step)
    #
    #     self.callback_processing_finished()
    #
    #     task.success = True
    #     task.picks_in_samples = task_picks_in_sample
    #     task.confidence = task_confidence
    #     task.model_hash = self.model_hash
    #
    #     return task


    # def pick_gather(self, gather: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     assert gather.ndim == 2
    #     assert all(dim > 0 for dim in gather.shape)
    #     outputs = self.model.run(None, {"input": gather[None, None, ...]})
    #     return outputs[0][0], outputs[1][0]
    #
    # def process_task(self, task: Task) -> Task:
    #     task_picks_in_sample = []
    #     task_confidence = []
    #
    #     self.callback_processing_started(task.num_gathers)
    #
    #     for step, gather_ids in enumerate(task.get_gathers_ids()):
    #         amplitudes = np.array(
    #             [-1 if idx in task.traces_to_inverse else 1 for idx in range(len(gather_ids))], dtype=np.float32
    #         )
    #
    #         gather = task.sgy.read_traces_by_ids(gather_ids)
    #
    #         gather = preprocess_gather(gather, task.gain, task.clip)
    #         gather = amplitudes[None, :] * gather
    #         gather = gather[: task.maximum_time_sample, :]
    #
    #         picks, confidence = self.pick_gather(gather)
    #
    #         task_picks_in_sample.extend(picks.tolist())
    #         task_confidence.extend(confidence.tolist())
    #
    #         self.callback_step_finished(step)
    #
    #     self.callback_processing_finished()
    #
    #     task.success = True
    #     task.picks_in_samples = task_picks_in_sample
    #     task.confidence = task_confidence
    #     task.model_hash = self.model_hash
    #
    #     return task
