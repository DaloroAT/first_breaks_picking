raise_if_no_torch()

import warnings
from os import cpu_count
from typing import Optional, Dict, Any

from PyQt5.QtWidgets import QApplication, QLabel, QComboBox, QSpinBox

from first_breaks.const import is_windows
from first_breaks._pytorch.utils import MODEL_TORCH_HASH
from first_breaks.desktop.main_gui import MainWindow
from first_breaks.desktop.picking_widget import PickingWindow, get_current_value_default
from first_breaks.desktop.utils import set_geometry, QHSeparationLine
from first_breaks._pytorch.picker_torch import PickerTorch
from first_breaks.picking.task import Task
from first_breaks.utils.utils import remove_unused_kwargs
from first_breaks.utils.availability import is_torch_cuda_available

warnings.filterwarnings("ignore")


class PickingWindowTorch(PickingWindow):
    def __init__(self,
                 task: Optional[Task] = None,
                 device: str = 'cuda' if is_torch_cuda_available else 'cpu',
                 num_workers: int = 0,
                 batch_size: int = 1):
        super().__init__(task)

        self.setWindowTitle("Picking and PyTorch settings")
        set_geometry(self, width_rel=0.22, height_rel=0.3, fix_size=True, centralize=True)

        assert device in ['cuda', 'cpu']
        self.hide()
        self.layout.removeWidget(self.button_box)
        self.layout.addWidget(QHSeparationLine(), 4, 0, 1, 2)

        self.device_label = QLabel('Device')
        self.device = QComboBox()
        self.storage['device'] = [self.device_label, self.device, self.get_current_value_combobox, str]
        self.device_idx2labelvalue = {}
        idx = 0
        if is_torch_cuda_available():
            self.device.addItem('GPU/CUDA')
            self.device_idx2labelvalue[idx] = ['GPU/CUDA', 'cuda']
            idx += 1
        self.device.addItem('CPU')
        self.device_idx2labelvalue[idx] = ['CPU', 'cpu']
        current_device_idx = [idx for idx, (_, value) in self.device_idx2labelvalue.items() if value == device][0]
        self.device.setCurrentIndex(current_device_idx)
        self.device.activated.connect(self.set_enabled_batch_size_depend_on_device)
        self.layout.addWidget(self.device_label, 5, 0)
        self.layout.addWidget(self.device, 5, 1)

        self.num_workers_label = QLabel("Number of I/O workers (No Windows)")
        self.num_workers = QSpinBox()
        self.storage['num_workers'] = [self.num_workers_label,
                                       self.num_workers,
                                       get_current_value_default,
                                       int]
        self.num_workers.setRange(0, cpu_count())
        if is_windows():
            self.num_workers.setValue(0)
            self.num_workers.setEnabled(False)
        else:
            self.num_workers.setValue(num_workers)
        self.layout.addWidget(self.num_workers_label, 6, 0)
        self.layout.addWidget(self.num_workers, 6, 1)

        self.batch_size_label = QLabel("Batch size (GPU)")
        self.batch_size = QSpinBox()
        self.storage['batch_size'] = [self.batch_size_label,
                                      self.batch_size,
                                      get_current_value_default,
                                      int]
        self.batch_size.setRange(1, 100)
        self.batch_size.setValue(batch_size)
        self.set_enabled_batch_size_depend_on_device()
        self.layout.addWidget(self.batch_size_label, 7, 0)
        self.layout.addWidget(self.batch_size, 7, 1)

        self.layout.addWidget(self.button_box)
        self.show()

    def set_enabled_batch_size_depend_on_device(self, _: Optional[int] = None) -> None:
        current_device = self.get_current_value_combobox()
        if current_device == 'cpu':
            self.batch_size.setEnabled(False)
        else:
            self.batch_size.setEnabled(True)

    def get_current_value_combobox(self, _: Optional[QComboBox] = None) -> str:
        return self.device_idx2labelvalue[self.device.currentIndex()][1]


class MainWindowTorch(MainWindow):
    def __init__(self):
        super().__init__()

        self.picker_class = PickerTorch
        self.picker: Optional[PickerTorch] = None
        self.picker_hash = MODEL_TORCH_HASH
        self.extra_kwargs_for_picker_init = {'show_progressbar': False,
                                             "device": "cuda",
                                             "batch_size": 1,
                                             "num_workers": 0}

        self.picking_window_class = PickingWindowTorch

    def receive_settings(self, settings: Dict[str, Any]) -> None:
        self.picking_window_extra_kwargs = remove_unused_kwargs(settings, PickerTorch.__init__)
        return super().receive_settings(settings)


def run_app_torch() -> None:
    from first_breaks.const import DEMO_SGY_PATH
    from first_breaks._pytorch.utils import MODEL_TORCH_PATH
    app = QApplication([])
    window = MainWindowTorch()
    window.load_nn(MODEL_TORCH_PATH)
    window.get_filename(DEMO_SGY_PATH)
    app.exec_()


if __name__ == "__main__":
    run_app_torch()
