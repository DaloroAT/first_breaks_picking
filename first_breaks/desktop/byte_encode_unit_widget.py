import warnings
from typing import Any, Dict, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from first_breaks.const import HIGH_DPI, FIRST_BYTE
from first_breaks.desktop.combobox_with_mapping import QComboBoxMapping
from first_breaks.sgy.headers import Headers

if HIGH_DPI:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

warnings.filterwarnings("ignore")


def build_encoding_mapping() -> Dict[int, Tuple[str, str]]:
    mapping = {
        0: ("Integer 4", "i"),
        1: ("UInteger 4", "I"),
        2: ("Short 2", "h"),
        3: ("UShort 2", "H"),
        4: ("Long 4", "l"),
        5: ("ULong 4", "L"),
        6: ("Float 4", "f"),
        7: ("Double 8", "d"),
    }
    assert all(v[1] in Headers.format2size.keys() for v in mapping.values())
    return mapping


ENCODING_MAPPING = build_encoding_mapping()


class QByteEncodeUnitWidget(QWidget):
    values_changed_signal = pyqtSignal(dict)

    def __init__(
        self,
        byte_position: int = 0,
        encoding: str = "I",
        first_byte: int = FIRST_BYTE,
        picks_unit: str = "mcs",
        margins: Optional[int] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.layout = QHBoxLayout()
        if margins is not None:
            self.layout.setContentsMargins(margins, margins, margins, margins)
        self.setLayout(self.layout)

        assert byte_position >= first_byte
        self.byte_position_value = byte_position
        self.encoding_value = encoding
        self.first_byte = first_byte
        self.byte_position_maximum = 240 + self.first_byte
        self.picks_unit_value = picks_unit

        self.byte_position_label = QLabel("Byte")
        self.byte_position_widget = QSpinBox()
        self.byte_position_widget.setRange(self.first_byte, self.byte_position_maximum)
        self.byte_position_widget.setValue(self.byte_position_value)
        self.byte_position_widget.valueChanged.connect(self.values_changed)

        self.layout.addWidget(self.byte_position_label)
        self.layout.addWidget(self.byte_position_widget)

        self.encoding_label = QLabel("Encoding")
        self.encoding_widget = QComboBoxMapping(ENCODING_MAPPING, current_value=self.encoding_value)  # type: ignore

        self.layout.addWidget(self.encoding_label)
        self.layout.addWidget(self.encoding_widget)

        self.picks_unit_label = QLabel("Unit")
        self.picks_unit_widget = QComboBoxMapping(
            {0: ("Microseconds", "mcs"), 1: ("Milliseconds", "ms"), 2: ("Samples", "sample")},
            current_value=self.picks_unit_value,
        )

        self.layout.addWidget(self.picks_unit_label)
        self.layout.addWidget(self.picks_unit_widget)

        self._previous_values = self.get_values()
        self.align_and_update_values()

        # connect after aligning
        self.picks_unit_widget.value_changed_signal.connect(self.values_changed)
        self.encoding_widget.value_changed_signal.connect(self.values_changed)

    def align_and_update_values(self) -> None:
        self.encoding_value = self.encoding_widget.value()
        self.byte_position_value = self.byte_position_widget.value()
        self.picks_unit_value = self.picks_unit_widget.value()

        byte_position_maximum_aligned = self.first_byte + 240 - Headers.format2size[self.encoding_value]
        byte_position_value_aligned = min(self.byte_position_value, byte_position_maximum_aligned)

        if byte_position_maximum_aligned != self.byte_position_maximum:
            self.byte_position_maximum = byte_position_maximum_aligned
            self.byte_position_widget.setMaximum(self.byte_position_maximum)

        if byte_position_value_aligned != self.byte_position_value:
            self.byte_position_value = byte_position_value_aligned
            self.byte_position_widget.setValue(self.byte_position_value)

    def get_values(self) -> Dict[str, Any]:
        return {
            "byte_position": self.byte_position_widget.value() - self.first_byte,
            "encoding": self.encoding_widget.value(),
            "picks_unit": self.picks_unit_widget.value(),
        }

    def values_changed(self) -> None:
        self.align_and_update_values()
        values = self.get_values()
        if values != self._previous_values:  # avoid emit extra signal if chained triggers happen
            self._previous_values = values
            self.values_changed_signal.emit(values)


class QDialogByteEncodeUnit(QDialog):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.widget = QByteEncodeUnitWidget(*args, **kwargs)
        self.layout.addWidget(self.widget)

        self.button_box = QDialogButtonBox()
        self.button_box.addButton("Ok", QDialogButtonBox.AcceptRole)
        self.button_box.accepted.connect(self.accept)

        self.layout.addWidget(self.button_box)

    def get_values(self) -> Dict[str, Any]:
        return self.widget.get_values()


if __name__ == "__main__":
    app = QApplication([])
    window = QDialogByteEncodeUnit(first_byte=1, byte_position=237, encoding="I", picks_unit="mcs")
    window.show()
    app.exec_()
    print(window.get_values())
