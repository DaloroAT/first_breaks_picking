import sys
import time
from typing import Optional, List

from PyQt5.QtWidgets import QWidget, QSizePolicy, QApplication, QMainWindow, QToolBar, QAction, QFileDialog, QLabel, \
    QDialog, QDialogButtonBox, QDesktopWidget, QVBoxLayout, QProgressBar, QHBoxLayout
from PyQt5.QtCore import QSize, Qt, QUrl, QThreadPool, QObject, pyqtSignal, QRunnable, pyqtSlot, QPoint
from PyQt5.QtGui import QIcon, QFont, QDesktopServices, QPixmap, QImage, QPolygonF, QPen, QPainterPath, QColor
import numpy as np
from pathlib import Path

from seismic.api.desktop.extra_widgets import WarnBox
from seismic.api.desktop.graph import GraphWidget, export_image_with_pyqt
from seismic.api.desktop.threads import InitNet, PickerServiceQRunnable
from seismic.api.nn_service import PickerService
from seismic.api.tasks import BaseTask
from seismic.config import common_config
from seismic.segmentation.picker import Picker
from seismic.sgy.sgy_reader import SGY
from seismic.utils.train_utils import chunk_iterable
from seismic.utils.visualizations import plotseis
import pyqtgraph as pg


import warnings
warnings.filterwarnings("ignore")


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        if getattr(sys, 'frozen', False):
            self.main_folder = Path(sys._MEIPASS)
        else:
            self.main_folder = Path(__file__).parent

        # main window settings
        left = 100
        top = 100
        width = 700
        height = 700
        self.setGeometry(left, top, width, height)

        qt_rectangle = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        qt_rectangle.moveCenter(center_point)
        self.move(qt_rectangle.topLeft())

        self.setWindowTitle('First break picking')

        # toolbar
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(30, 30))
        self.addToolBar(toolbar)

        # buttons on toolbar
        self.button_open_filename = QAction(QIcon(str(self.main_folder / "icons" / "sgy.png")), "Open SGY-file", self)
        self.button_open_filename.triggered.connect(self.get_filename)
        self.button_open_filename.setEnabled(False)
        toolbar.addAction(self.button_open_filename)

        self.button_fb = QAction(QIcon(str(self.main_folder / "icons" / "picking.png")), "Neural network FB picking", self)
        self.button_fb.triggered.connect(self.calc_fb)
        self.button_fb.setEnabled(False)
        toolbar.addAction(self.button_fb)

        self.button_export = QAction(QIcon(str(self.main_folder / "icons" / "export.png")), "Export picks to file", self)
        # self.button_export.triggered.connect(self.export)
        self.button_export.setEnabled(False)
        toolbar.addAction(self.button_export)

        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar.addWidget(spacer)

        self.button_git = QAction(QIcon(str(self.main_folder / "icons" / "github.png")), "Open Github repo with project", self)
        # self.button_git.triggered.connect(self.open_github)
        toolbar.addAction(self.button_git)

        self.name_input = None
        self.picks = None

        # sgy placeholders
        self.sgy = None
        self.fn = None
        self.traces = None
        self.norm_traces = None
        self.picks = None
        self.num_traces_to_render = None
        self.traces_rendered = 0

        self.status = self.statusBar()
        self.status_progress = QProgressBar()
        self.status_progress.hide()

        self.status_message = QLabel()

        self.status_message.setText('Initialize model')

        status_widget = QWidget()
        status_layout = QHBoxLayout()
        status_widget.setLayout(status_layout)
        status_layout.addWidget(self.status_progress)
        status_layout.addWidget(self.status_message)

        self.status.addPermanentWidget(status_widget)

        # graph widget
        self.graph = GraphWidget(background='w')
        self.setCentralWidget(self.graph)

        # self.image = QLabel()
        # self.setCentralWidget(self.image)

        self.threadpool = QThreadPool()
        print(
            "Multithreading with maximum %d threads" % self.
            threadpool.maxThreadCount()
        )
        self.picker: Optional[PickerService] = None
        self.graph.hide()
        self.show()
        self._thread_init_net()

    def _thread_init_net(self):
        worker = InitNet()
        worker.signals.finished.connect(self.init_net)
        self.threadpool.start(worker)

    def init_net(self, picker: PickerService):
        self.picker = picker
        self.button_open_filename.setEnabled(True)
        self.status_message.setText('Ready')

    def calc_fb(self):
        self.button_fb.setEnabled(False)
        task = BaseTask(path_sgy=self.fn)
        worker = PickerServiceQRunnable(self.picker, task)
        worker.signals.started.connect(self.start_fb)
        worker.signals.result.connect(self.result_fb)
        worker.signals.progress.connect(self.progress_fb)
        worker.signals.message.connect(self.message_fb)
        worker.signals.finished.connect(self.finish_fb)
        self.threadpool.start(worker)

    def start_fb(self):
        self.status_progress.show()

    def message_fb(self, message: str):
        self.status_message.setText(message)

    def finish_fb(self):
        self.status_progress.hide()
        self.button_fb.setEnabled(True)

    def progress_fb(self, value: int):
        self.status_progress.setValue(value)

    def result_fb(self, task: BaseTask):
        self.button_fb.setEnabled(True)
        if task.success:
            self.graph.plot_picks(task)
        else:
            window_error = WarnBox(self, title='InternalError', message=task.error_message)
            window_error.exec_()

    def show_sgy(self):
        try:
            self.sgy = SGY(self.fn)
            self.graph.clear()
            self.graph.plotseis_sgy(self.fn, negative_patch=True)
            self.graph.show()

        except Exception as e:
            window_err = WarnBox(self,
                                 title=e.__class__.__name__,
                                 message=str(e))
            window_err.exec_()
        finally:
            self.button_open_filename.setEnabled(True)

    def get_filename(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Open SGY-file", "",
                                                  "SGY-file (*.sgy)", options=options)
        if filename:
            self.fn = Path(filename)
            self.picks = None
            self.show_sgy()
            self.button_fb.setEnabled(True)


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    app.exec_()
