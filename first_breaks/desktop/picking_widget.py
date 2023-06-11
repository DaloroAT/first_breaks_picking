import warnings
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QSpinBox, QWidget,
)

from first_breaks.const import HIGH_DPI
from first_breaks.desktop.utils import WarnBox, set_geometry
from first_breaks.picking.task import Task

if HIGH_DPI:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

warnings.filterwarnings("ignore")


def get_current_value_default(widget: QWidget) -> str:
    return widget.text()


class PickingWindow(QDialog):

    export_settings_signal = pyqtSignal(dict)

    def __init__(self, task: Optional[Task] = None):
        super().__init__()

        self.setWindowTitle("Picking settings")
        set_geometry(self, width_rel=0.2, height_rel=0.2, fix_size=True, centralize=True)

        self.storage = {}

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.traces_per_gather_label = QLabel("Traces per gather")
        self.traces_per_gather = QSpinBox()
        self.storage['traces_per_gather'] = [self.traces_per_gather_label,
                                             self.traces_per_gather,
                                             get_current_value_default,
                                             int]
        self.traces_per_gather.setRange(2, 999)
        value_traces_per_gather = task.traces_per_gather if task else 48
        self.traces_per_gather.setValue(value_traces_per_gather)
        self.layout.addWidget(self.traces_per_gather_label, 0, 0)
        self.layout.addWidget(self.traces_per_gather, 0, 1)

        self.maximum_time_label = QLabel("Maximum time, ms")
        self.maximum_time = QLineEdit()
        self.storage['maximum_time'] = [self.maximum_time_label,
                                        self.maximum_time,
                                        get_current_value_default,
                                        float]
        maximum_time_validator = QDoubleValidator()
        maximum_time_validator.setBottom(0.0)
        self.maximum_time.setValidator(maximum_time_validator)
        value_maximum_time = task.maximum_time if task else 100.0
        self.maximum_time.setText(str(value_maximum_time))
        self.layout.addWidget(self.maximum_time_label, 1, 0)
        self.layout.addWidget(self.maximum_time, 1, 1)

        self.gain_label = QLabel("Gain")
        self.gain = QLineEdit()
        self.storage['gain'] = [self.gain_label, self.gain, get_current_value_default, float]
        gain_validator = QDoubleValidator()
        self.gain.setValidator(gain_validator)
        value_gain = task.gain if task else 1
        self.gain.setText(str(value_gain))
        self.layout.addWidget(self.gain_label, 2, 0)
        self.layout.addWidget(self.gain, 2, 1)

        self.clip_label = QLabel("Clip")
        self.clip = QLineEdit()
        self.storage['clip'] = [self.clip_label, self.clip, get_current_value_default, float]
        clip_validator = QDoubleValidator()
        clip_validator.setBottom(0.1)
        self.clip.setValidator(clip_validator)
        value_clip = task.clip if task else 1
        self.clip.setText(str(value_clip))
        self.layout.addWidget(self.clip_label, 3, 0)
        self.layout.addWidget(self.clip, 3, 1)

        self.button_box = QDialogButtonBox()
        self.button_box.addButton("Run picking", QDialogButtonBox.AcceptRole)
        self.button_box.addButton("Cancel", QDialogButtonBox.RejectRole)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.layout.addWidget(self.button_box)
        self.show()

    def mark_as_invalid_if_empty(self, edit: QLineEdit, label: str) -> None:
        if edit.text().strip():
            if label in self.invalid_fields:
                self.invalid_fields.remove(label)
        else:
            self.invalid_fields.add(label)

    def accept(self) -> None:
        empty_fields = [label.split(",")[0] for label, widget, getter, _ in self.storage.values()
                        if not str(getter(widget)).strip()]

        if empty_fields:
            template = "Fields" if len(empty_fields) > 1 else "Field"
            template += " {} must be filled"
            invalid_fields_str = ",".join(empty_fields)
            window_error = WarnBox(self, title="Input error", message=template.format(invalid_fields_str))
            window_error.exec_()
        else:
            settings = {name: cast(getter(widget)) for name, (_, widget, getter, cast) in self.storage.items()}
            print(settings)
            self.export_settings_signal.emit(settings)
            super().accept()

    def reject(self) -> None:
        self.export_settings_signal.emit({})
        super().reject()


if __name__ == "__main__":
    app = QApplication([])
    window = PickingWindow()
    app.exec_()
