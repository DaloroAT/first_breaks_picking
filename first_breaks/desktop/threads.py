from typing import Any

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot

from first_breaks.picking.ipicker import IPicker
from first_breaks.picking.task import Task


class PickerSignals(QObject):
    started = pyqtSignal()
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    result = pyqtSignal(Task)
    message = pyqtSignal(str)


class PickerQRunnable(QRunnable):
    def __init__(self, picker: IPicker, task: Task):
        super().__init__()

        self.signals = PickerSignals()

        self.picker = picker
        self.task = task

        self.picker.callback_step_finished = self.callback_step_finished  # type: ignore
        self.picker.callback_processing_started = self.callback_processing_started  # type: ignore
        self.picker.callback_processing_finished = self._do_nothing  # type: ignore

        self.len = 0

    def _do_nothing(self, *args: Any, **kwargs: Any) -> None:
        pass

    def callback_step_finished(self, idx_batch: int) -> None:
        progress = int(100 * (idx_batch + 1) / self.len)
        self.signals.progress.emit(progress)

    def callback_processing_started(self, length: int) -> None:  # type: ignore
        self.len = length
        self.signals.progress.emit(0)
        self.signals.message.emit("Picking")

    @pyqtSlot()
    def run(self) -> None:
        self.signals.started.emit()
        self.signals.message.emit("Started")

        try:
            self.task = self.picker.process_task(self.task)
        except Exception as e:
            self.task.success = False
            self.task.error_message = str(e)
        finally:
            message = "Completed" if self.task.success else "Picking Error"
            self.signals.progress.emit(100)
            self.signals.finished.emit()
            self.signals.message.emit(message)
            self.signals.result.emit(self.task)


class CallInThreadSignals(QObject):
    started = pyqtSignal()
    finished = pyqtSignal()
    success = pyqtSignal(bool)
    result = pyqtSignal(object)


class CallInThread(QRunnable):
    def __init__(self, callable_obj: Any, *args: Any, **kwargs: Any):
        super().__init__()
        self.callable_obj = callable_obj
        self.args = args
        self.kwargs = kwargs
        self.signals = CallInThreadSignals()

    @pyqtSlot()
    def run(self) -> None:
        self.signals.started.emit()
        try:
            result = self.callable_obj(*self.args, **self.kwargs)
            success = True
        except Exception as e:
            result = e
            success = False

        self.signals.finished.emit()
        self.signals.success.emit(success)
        self.signals.result.emit(result)
