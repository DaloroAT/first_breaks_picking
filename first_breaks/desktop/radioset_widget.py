from typing import Any, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from first_breaks.desktop.utils import (
    TMappingSetup,
    validate_mapping_setup_and_get_current_index,
)


class QRadioSetWidget(QWidget):
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
        orientation: str = "horizontal",
        margins: Optional[int] = None,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        assert orientation in ["horizontal", "vertical"]

        current_index = validate_mapping_setup_and_get_current_index(
            mapping,
            current_index=current_index,
            current_label=current_label,
            current_value=current_value,
        )

        self.mapping = mapping
        self.current_index = current_index

        if orientation == "horizontal":
            self.layout = QHBoxLayout()
        else:
            self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        if margins is not None:
            self.layout.setContentsMargins(margins, margins, margins, margins)

        self.storage = {}
        self.selected_idx = current_index

        for idx, (label, value) in mapping.items():
            radio = QRadioButton(label)
            self.storage[radio] = idx
            if idx == current_index:
                radio.setChecked(True)
            radio.toggled.connect(lambda *lambda_args, **lambda_kwargs: self.changed())
            self.layout.addWidget(radio)

    def changed(self) -> None:
        radio = self.sender()
        if radio.isChecked():
            self.selected_idx = self.storage[radio]

            self.index_changed_signal.emit(self.selected_idx)
            self.value_changed_signal.emit(self.value())
            self.label_changed_signal.emit(self.label())
            self.changed_signal.emit()

    def value(self) -> Any:
        return self.mapping[self.selected_idx][1]

    def text(self) -> str:
        return str(self.value())

    def label(self) -> Any:
        return self.mapping[self.selected_idx][0]


if __name__ == "__main__":
    app = QApplication([])
    window = QRadioSetWidget({0: ("A", "aa"), 1: ("B", "bb"), 2: ("C", "cc")})
    window.show()
    app.exec_()
