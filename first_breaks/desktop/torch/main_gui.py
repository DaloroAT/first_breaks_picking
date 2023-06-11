import warnings
from typing import Any, Dict, Optional

from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QStyle,
)

from first_breaks.const import MODEL_TORCH_HASH
from first_breaks.picking.picker.picker_torch import PickerTorch

from first_breaks.desktop.main_gui import MainWindow

warnings.filterwarnings("ignore")


class MainWindowTorch(MainWindow):
    def __init__(self):
        super().__init__()

        icon_runtime_settings = self.style().standardIcon(QStyle.SP_DirIcon)
        self.button_runtime_settings = QAction(icon_runtime_settings, "Runtime parameters", self)
        self.button_runtime_settings.triggered.connect(self.set_runtime_settings)
        self.button_runtime_settings.setEnabled(True)
        self.toolbar.insertAction(self.button_fb, self.button_runtime_settings)

        self.torch_runtime_settings: Optional[Dict[str, Any]] = None
        self.model_hash = MODEL_TORCH_HASH
        self.picker: Optional[PickerTorch] = None

    def set_runtime_settings(self):
        pass


def run_torch_app() -> None:
    app = QApplication([])
    _ = MainWindowTorch()
    app.exec_()


if __name__ == "__main__":
    run_torch_app()


