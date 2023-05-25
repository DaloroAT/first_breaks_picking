from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, pyqtSlot

from first_breaks.picker.picker import PickerONNX, Task


class PickerSignals(QObject):
    started = pyqtSignal()
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    result = pyqtSignal(Task)
    message = pyqtSignal(str)


class PickerQRunnable(QRunnable):
    def __init__(self, picker: PickerONNX, task: Task):
        super().__init__()

        self.signals = PickerSignals()

        self.picker = picker
        self.task = task

        self.picker.callback_step_finished = self.callback_step_finished
        self.picker.callback_processing_started = self.callback_processing_started

        self.len = task.num_gathers

    def callback_step_finished(self, idx_batch: int):
        progress = int(100 * (idx_batch + 1) / self.len)
        self.signals.progress.emit(progress)

    def callback_processing_started(self, length: int):
        self.signals.progress.emit(0)
        self.signals.message.emit('Picking')

    @pyqtSlot()
    def run(self):
        self.signals.started.emit()
        self.signals.message.emit('Started')

        try:
            self.task = self.picker.process_task(self.task, return_picks_in_ms=True)
        except Exception as e:
            self.task.success = False
            self.task.error_message = str(e)
        finally:
            message = 'Completed' if self.task.success else 'Picking Error'
            self.signals.progress.emit(100)
            self.signals.finished.emit()
            self.signals.message.emit(message)
            self.signals.result.emit(self.task)


class InitNetSignals(QObject):
    finished = pyqtSignal(PickerONNX)


class InitNet(QRunnable):
    def __init__(self, weights: Path):
        super().__init__()
        self.weights = weights
        self.signals = InitNetSignals()

    @pyqtSlot()
    def run(self):
        picker = PickerONNX(self.weights, show_progressbar=False)
        self.signals.finished.emit(picker)
