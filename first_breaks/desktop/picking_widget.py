import warnings
from typing import Optional

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QApplication, QLabel, QDialogButtonBox, QSpinBox, QDialog, QGridLayout, QLineEdit

from first_breaks.const import HIGH_DPI
from first_breaks.desktop.warn_widget import WarnBox
from first_breaks.picking.task import Task

if HIGH_DPI:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

warnings.filterwarnings("ignore")


class PickingWindow(QDialog):

    export_settings_signal = pyqtSignal(dict)

    def __init__(self, task: Optional[Task] = None):
        super().__init__()

        self.setWindowTitle("Picking settings")
        h, w = self.screen().size().height(), self.screen().size().width()
        left = int(0.4 * w)
        top = int(0.4 * h)
        width = int(0.2 * w)
        height = int(0.2 * h)
        self.setGeometry(left, top, width, height)
        self.setFixedSize(width, height)

        self.label2widget = {}
        layout = QGridLayout()
        self.setLayout(layout)

        self.traces_per_gather_label = QLabel('Traces per gather')
        self.traces_per_gather = QSpinBox()
        self.label2widget[self.traces_per_gather_label] = self.traces_per_gather
        self.traces_per_gather.setRange(2, 999)
        value_traces_per_gather = task.traces_per_gather if task else 48
        self.traces_per_gather.setValue(value_traces_per_gather)
        layout.addWidget(self.traces_per_gather_label, 0, 0)
        layout.addWidget(self.traces_per_gather, 0, 1)

        self.maximum_time_label = QLabel('Maximum time, ms')
        self.maximum_time = QLineEdit()
        self.label2widget[self.maximum_time_label] = self.maximum_time
        maximum_time_validator = QDoubleValidator()
        maximum_time_validator.setBottom(0.0)
        self.maximum_time.setValidator(maximum_time_validator)
        value_maximum_time = task.maximum_time if task else 100.0
        self.maximum_time.setText(str(value_maximum_time))
        layout.addWidget(self.maximum_time_label, 1, 0)
        layout.addWidget(self.maximum_time, 1, 1)

        self.gain_label = QLabel('Gain')
        self.gain = QLineEdit()
        self.label2widget[self.gain_label] = self.gain
        gain_validator = QDoubleValidator()
        self.gain.setValidator(gain_validator)
        value_gain = task.gain if task else 1
        self.gain.setText(str(value_gain))
        layout.addWidget(self.gain_label, 2, 0)
        layout.addWidget(self.gain, 2, 1)

        self.clip_label = QLabel('Clip')
        self.clip = QLineEdit()
        self.label2widget[self.clip_label] = self.clip
        clip_validator = QDoubleValidator()
        clip_validator.setBottom(0.1)
        self.clip.setValidator(clip_validator)
        value_clip = task.clip if task else 1
        self.clip.setText(str(value_clip))
        layout.addWidget(self.clip_label, 3, 0)
        layout.addWidget(self.clip, 3, 1)

        self.buttonBox = QDialogButtonBox()
        self.buttonBox.addButton("Run picking", QDialogButtonBox.AcceptRole)
        self.buttonBox.addButton("Cancel", QDialogButtonBox.RejectRole)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.buttonBox)
        self.show()

    def mark_as_invalid_if_empty(self, edit: QLineEdit, label: str):
        if edit.text().strip():
            if label in self.invalid_fields:
                self.invalid_fields.remove(label)
        else:
            self.invalid_fields.add(label)

    def accept(self) -> None:
        invalid_fields = [label.text().split(',')[0] for label, widget in self.label2widget.items()
                          if not widget.text().strip()]

        if invalid_fields:
            template = "Fields" if len(invalid_fields) > 1 else "Field"
            template += " {} must be filled"
            invalid_fields = ','.join(invalid_fields)
            window_error = WarnBox(self, title='Input error', message=template.format(invalid_fields))
            window_error.exec_()
        else:
            settings = {'traces_per_gather': self.traces_per_gather.value(),
                        'maximum_time': float(self.maximum_time.text()),
                        'gain': float(self.gain.text()),
                        'clip': float(self.clip.text())}
            self.export_settings_signal.emit(settings)
            super().accept()

    def reject(self) -> None:
        self.export_settings_signal.emit({})
        super().reject()


if __name__ == '__main__':
    app = QApplication([])
    window = PickingWindow()
    app.exec_()
