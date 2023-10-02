from pathlib import Path
from typing import Optional, Union

from PyQt5.QtCore import QThreadPool, QObject, pyqtSignal, pyqtBoundSignal
from PyQt5.QtWidgets import (
    QLabel,
    QProgressBar,
)

from first_breaks.desktop.threads import CallInThread, PickerQRunnable
from first_breaks.desktop.settings_processing_widget import (
    PickingSettings,
)
from first_breaks.picking.ipicker import IPicker
from first_breaks.picking.picker_onnx import PickerONNX
from first_breaks.picking.task import Task
from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import (
    remove_unused_kwargs,
)


class NNManager(QObject):
    parsing_picking_parameters_error_signal = pyqtSignal()
    picking_started_signal = pyqtSignal()
    picking_finished_signal = pyqtSignal(Task)

    def __init__(
            self,
            status_progress: QProgressBar,
            status_message: QLabel,
            threadpool: QThreadPool,
            interrupt_on: Union[pyqtSignal, pyqtBoundSignal]
    ):
        super().__init__()
        self.status_progress = status_progress
        self.status_message = status_message
        self.threadpool = threadpool
        self.interrupt_on = interrupt_on

        self.picker_class = PickerONNX
        self.picker_kwargs_init = {"show_progressbar": False, "device": "cpu"}

        self.picker: Optional[IPicker] = None

    def init_net(self, weights: Union[str, Path]) -> None:
        task = CallInThread(self.picker_class, model_path=weights, **self.picker_kwargs_init)
        task.signals.result.connect(self._store_picker)
        self.threadpool.start(task)

    def _store_picker(self, picker: PickerONNX) -> None:
        self.picker = picker

    def pick_fb(self, sgy: SGY, settings: PickingSettings):
        try:
            settings = settings.model_dump()
            change_settings = remove_unused_kwargs(settings, self.picker.change_settings)
            self.picker.change_settings(**change_settings)

            task_kwargs = remove_unused_kwargs(settings, Task)
            task = Task(sgy=sgy, **task_kwargs)

            worker = PickerQRunnable(picker=self.picker, task=task, interrpution_signal=self.interrupt_on)
            worker.signals.started.connect(self.on_start_task)
            worker.signals.result.connect(self.on_result_task)
            worker.signals.progress.connect(self.on_progressbar_task)
            worker.signals.message.connect(self.on_message_task)
            worker.signals.result.connect(self.on_finish_task)
            self.threadpool.start(worker)
            self.picking_started_signal.emit()
        except Exception:
            self.parsing_picking_parameters_error_signal.emit()

    def on_start_task(self) -> None:
        self.status_progress.show()

    def on_message_task(self, message: str) -> None:
        self.status_message.setText(message)

    def on_progressbar_task(self, value: int) -> None:
        self.status_progress.setValue(value)

    def on_finish_task(self) -> None:
        self.status_progress.hide()

    def on_result_task(self, result: Task) -> None:
        self.picking_finished_signal.emit(result)
