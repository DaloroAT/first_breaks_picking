import warnings
from typing import Optional, Any, Dict, Tuple, Union, List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from first_breaks.const import HIGH_DPI
from first_breaks.desktop.utils import MessageBox, QHSeparationLine, set_geometry
from first_breaks.picking.task import Task
from first_breaks.utils.utils import is_onnx_cuda_available

if HIGH_DPI:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

warnings.filterwarnings("ignore")


class QComboBoxMapping(QComboBox):
    text_changed_signal = pyqtSignal(str)

    def __init__(self,
                 mapping: Dict[int, Union[Tuple[str, str], List[str]]],
                 current_index: Optional[int] = 0,
                 current_text: Optional[str] = None,
                 *args: Any,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)

        if (current_index is None and current_text is None) or (current_index is not None and current_text is not None):
            assert ValueError("Either 'current_index' or 'current_text' have to be specified")

        if current_text:
            corresponding_ids = [k for k, v in mapping.items() if v[1] == current_text]
            if not corresponding_ids:
                raise KeyError(f"Text '{current_text}' not found in mapping. "
                               f"Available texts {[v[1] for v in mapping.values()]}"
                               )
            current_index = corresponding_ids[0]

        assert 0 <= current_index < len(mapping)

        self.mapping = mapping
        self.current_index = current_index

        items = [v[0] for v in self.mapping.values()]
        self.addItems(items)

        self.setCurrentIndex(current_index)
        self.currentIndexChanged.connect(self.text_changed)

    def text_changed(self):
        self.text_changed_signal.emit(self.text())

    def value(self) -> str:
        return self.mapping[self.currentIndex()][1]

    def text(self) -> str:
        return str(self.value())
