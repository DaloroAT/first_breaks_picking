import warnings
from typing import Any, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QComboBox

from first_breaks.const import HIGH_DPI
from first_breaks.desktop.utils import (
    TMappingSetup,
    validate_mapping_setup_and_get_current_index,
)

if HIGH_DPI:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

warnings.filterwarnings("ignore")


class QComboBoxMapping(QComboBox):
    index_changed_signal = pyqtSignal(int)
    value_changed_signal = pyqtSignal(object)
    label_changed_signal = pyqtSignal(str)
    changed_signal = pyqtSignal()

    def __init__(
        self,
        mapping: TMappingSetup,
        current_index: Optional[int] = None,
        current_label: Optional[str] = None,
        current_value: Optional[str] = None,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)

        current_index = validate_mapping_setup_and_get_current_index(
            mapping,
            current_index=current_index,
            current_label=current_label,
            current_value=current_value,
        )

        self.mapping = mapping
        self.current_index = current_index

        items = [v[0] for v in self.mapping.values()]
        self.addItems(items)

        self.setCurrentIndex(current_index)
        self.currentIndexChanged.connect(self.changed)

    def changed(self):
        self.index_changed_signal.emit(self.currentIndex())
        self.value_changed_signal.emit(self.value())
        self.label_changed_signal.emit(self.label())
        self.changed_signal.emit()

    def value(self) -> str:
        return self.mapping[self.currentIndex()][1]

    def text(self) -> str:
        return str(self.value())

    def label(self) -> str:
        return self.mapping[self.currentIndex()][0]
