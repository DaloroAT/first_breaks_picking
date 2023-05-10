import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Union, Dict

from PyQt5.QtCore import QSize, QThreadPool, pyqtSignal
from PyQt5.QtGui import QIcon, QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QWidget, QSizePolicy, QApplication, QMainWindow, QToolBar, QAction, QFileDialog, QLabel, \
    QDesktopWidget, QProgressBar, QHBoxLayout, QDialogButtonBox, QSpinBox, QCheckBox, QDialog, QGridLayout, QLineEdit

from first_breaks.const import CKPT_HASH
from first_breaks.desktop.warn_widget import WarnBox
from first_breaks.desktop.graph import GraphWidget
from first_breaks.desktop.threads import InitNet, PickerQRunnable
from first_breaks.picker.picker import PickerONNX, Task
from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import calc_hash

warnings.filterwarnings("ignore")


class PickingWindow(QDialog):

    export_settings_signal = pyqtSignal(dict)

    def __init__(self, task: Optional[Task] = None):
        super().__init__()

        self.setWindowTitle("Settings")
        x, y, width, height = 500, 500, 300, 200
        self.setGeometry(x, y, width, height)
        self.setFixedSize(width, height)

        layout = QGridLayout()
        self.setLayout(layout)

        self.traces_per_gather = QSpinBox()
        self.traces_per_gather.setMinimum(2)
        value_traces_per_gather = task.traces_per_gather if task else 24
        self.traces_per_gather.setValue(value_traces_per_gather)
        layout.addWidget(QLabel('Traces per gather'), 0, 0)
        layout.addWidget(self.traces_per_gather, 0, 1)

        self.maximum_time = QLineEdit()
        maximum_time_validator = QDoubleValidator()
        maximum_time_validator.setBottom(0.0)
        self.maximum_time.setValidator(maximum_time_validator)
        value_maximum_time = task.maximum_time if task else 0.0
        self.maximum_time.setText(str(value_maximum_time))
        layout.addWidget(QLabel('Maximum time, ms'), 1, 0)
        layout.addWidget(self.maximum_time, 1, 1)

        self.gain = QLineEdit()
        gain_validator = QDoubleValidator()
        self.gain.setValidator(gain_validator)
        value_gain = task.gain if task else 1
        self.gain.setText(str(value_gain))
        layout.addWidget(QLabel('Gain'), 2, 0)
        layout.addWidget(self.gain, 2, 1)

        self.clip = QLineEdit()
        clip_validator = QDoubleValidator()
        clip_validator.setBottom(0.1)
        self.clip.setValidator(clip_validator)
        value_clip = task.clip if task else 1
        self.clip.setText(str(value_clip))
        layout.addWidget(QLabel('Clip'), 3, 0)
        layout.addWidget(self.clip, 3, 1)

        # labels = ["Keep aspect ratio", "Autosave", "Extend borders"]
        #
        # self.box_keep_aspect = QCheckBox()
        # layout.addWidget(QLabel(labels[0]), 0, 0)
        # layout.addWidget(self.box_keep_aspect, 0, 1)
        #
        # self.box_autosave = QCheckBox()
        # self.box_autosave.setEnabled(False)
        # self.label_save = QLabel(labels[1])
        # self.label_save.setEnabled(False)
        # layout.addWidget(self.label_save, 1, 0)
        # layout.addWidget(self.box_autosave, 1, 1)
        #
        # self.spin_extend = QSpinBox()
        # self.spin_extend.setRange(0, 50)
        # layout.addWidget(QLabel(labels[2]), 2, 0)
        # layout.addWidget(self.spin_extend, 2, 1)

        # self.json_specified = False

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.buttonBox)
        self.show()

    def accept(self) -> None:
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
