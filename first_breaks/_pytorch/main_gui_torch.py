import warnings
from typing import Optional, Dict, Any

from PyQt5.QtWidgets import QApplication

from first_breaks._pytorch.picker_torch import PickerTorch
from first_breaks._pytorch.utils import MODEL_TORCH_HASH
from first_breaks.desktop.main_gui import MainWindow
from first_breaks.utils.utils import remove_unused_kwargs

warnings.filterwarnings("ignore")


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
