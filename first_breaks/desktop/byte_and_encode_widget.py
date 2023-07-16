import warnings
from typing import Optional, Dict, Tuple, Any

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
    QWidget, QHBoxLayout, QDial,
)

from first_breaks.const import HIGH_DPI
from first_breaks.desktop.combobox_with_mapping import QComboBoxMapping
from first_breaks.desktop.utils import MessageBox, QHSeparationLine, set_geometry
from first_breaks.picking.task import Task
from first_breaks.sgy.headers import Headers
from first_breaks.utils.utils import is_onnx_cuda_available

if HIGH_DPI:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

warnings.filterwarnings("ignore")


def build_encoding_mapping() -> Dict[int, Tuple[str, str]]:
    mapping = {0: ("Integer 4", "i"),
               1: ("UInteger 4", "I"),
               2: ("Short 2", "h"),
               3: ("UShort 2", "H"),
               4: ("Long 4", "l"),
               5: ("ULong 4", "L"),
               6: ("Float 4", "f"),
               7: ("Double 8", 'd')}
    assert all(v[1] in Headers.format2size.keys() for v in mapping.values())
    return mapping


ENCODING_MAPPING = build_encoding_mapping()


class QByteEncodeWidget(QWidget):
    values_changed_signal = pyqtSignal(dict)

    def __init__(self,
                 byte_position: int = 1,
                 encoding: str = 'I',
                 first_byte: int = 1,
                 margins: Optional[int] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.layout = QHBoxLayout()
        if margins is not None:
            self.layout.setContentsMargins(margins, margins, margins, margins)
        self.setLayout(self.layout)

        assert byte_position >= first_byte
        self.byte_position_value = byte_position
        self.encoding_value = encoding
        self.first_byte = first_byte
        self.byte_position_maximum = 239 + self.first_byte

        self.byte_position_label = QLabel("Byte")
        self.byte_position_widget = QSpinBox()
        self.byte_position_widget.setRange(self.first_byte, self.byte_position_maximum)
        self.byte_position_widget.valueChanged.connect(self.values_changed)

        self.layout.addWidget(self.byte_position_label)
        self.layout.addWidget(self.byte_position_widget)

        self.encoding_label = QLabel("Encoding")
        self.encoding_widget = QComboBoxMapping(ENCODING_MAPPING, current_text="i")
        # self.encoding_widget.currentIndexChanged.connect(self.align_byte_position)
        self.encoding_widget.text_changed_signal.connect(self.values_changed)
        self.align_and_update_values()

        self.layout.addWidget(self.encoding_label)
        self.layout.addWidget(self.encoding_widget)

        self._previous_values = self.get_values()

    def align_and_update_values(self):
        self.encoding_value = self.encoding_widget.value()
        self.byte_position_value = self.byte_position_widget.value()

        byte_position_maximum_aligned = self.first_byte + 239 - Headers.format2size[self.encoding_value]
        byte_position_value_aligned = min(self.byte_position_value, byte_position_maximum_aligned)

        if byte_position_maximum_aligned != self.byte_position_maximum:
            self.byte_position_maximum = byte_position_maximum_aligned
            self.byte_position_widget.setMaximum(self.byte_position_maximum)

        if byte_position_value_aligned != self.byte_position_value:
            self.byte_position_value = byte_position_value_aligned
            self.byte_position_widget.setValue(self.byte_position_value)

    def get_values(self) -> Dict[str, Any]:
        return {"bytes_position": self.byte_position_widget.value(), "encoding": self.encoding_widget.value()}

    def values_changed(self):
        self.align_and_update_values()
        values = self.get_values()
        if values != self._previous_values:  # avoid emit extra signal if chained triggers happen
            self._previous_values = values
            self.values_changed_signal.emit(values)


if __name__ == "__main__":
    app = QApplication([])
    window = QByteEncodeWidget()
    window.show()
    app.exec_()


