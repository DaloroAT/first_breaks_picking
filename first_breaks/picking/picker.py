from pathlib import Path
from typing import Union, Tuple, Any, Optional

import numpy as np
from tqdm.auto import tqdm

from first_breaks.picking.task import Task
from first_breaks.picking.utils import preprocess_gather
from first_breaks.utils.utils import download_model_onnx, calc_hash

import onnxruntime as ort


class PickerONNX:
    def __init__(self, onnx_path: Optional[Union[str, Path]] = None, show_progressbar: bool = True):
        if onnx_path is None:
            onnx_path = download_model_onnx()
        self.onnx_path = onnx_path
        self.model_hash = calc_hash(self.onnx_path)
        self.model = ort.InferenceSession(onnx_path)
        self.show_progressbar = show_progressbar
        self.progressbar: Optional[tqdm] = None

    def callback_processing_started(self, length: int) -> Any:
        if self.show_progressbar:
            self.progressbar = tqdm(desc='Picking', total=length)

    def callback_processing_finished(self) -> Any:
        if self.show_progressbar:
            self.progressbar.close()

    def callback_step_started(self, step: int) -> Any:
        pass

    def callback_step_finished(self, step: int) -> Any:
        if self.show_progressbar:
            self.progressbar.update(1)

    def pick_gather(self, gather: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert gather.ndim == 2
        assert all(dim > 0 for dim in gather.shape)
        outputs = self.model.run(None, {"input": gather[None, None, ...]})
        return outputs[0][0], outputs[1][0]

    def process_task(self, task: Task) -> Task:
        task_picks_in_sample = []
        task_confidence = []

        self.callback_processing_started(task.num_gathers)

        for step, gather_ids in enumerate(task.get_gathers_ids()):
            self.callback_step_started(step)

            amplitudes = np.array([-1 if idx in task.traces_to_inverse else 1 for idx in range(len(gather_ids))],
                                  dtype=np.float32)

            gather = task.sgy.read_traces_by_ids(gather_ids)

            gather = preprocess_gather(gather, task.gain, task.clip)
            gather = amplitudes[None, :] * gather
            gather = gather[:task.maximum_time_sample, :]

            picks, confidence = self.pick_gather(gather)

            task_picks_in_sample.extend(picks.tolist())
            task_confidence.extend(confidence.tolist())

            self.callback_step_finished(step)

        self.callback_processing_finished()

        task.success = True
        task.picks_in_samples = task_picks_in_sample
        task.confidence = task_confidence
        task.model_hash = self.model_hash

        return task
