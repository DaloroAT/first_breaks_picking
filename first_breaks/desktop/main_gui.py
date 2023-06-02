import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Union, Dict, Any

from PyQt5.QtCore import QSize, QThreadPool, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QSizePolicy, QApplication, QMainWindow, QToolBar, QAction, QFileDialog, QLabel, \
    QDesktopWidget, QProgressBar, QHBoxLayout, QStyle, QSlider
from PyQt5.uic.properties import QtWidgets

from first_breaks.const import MODEL_ONNX_HASH, HIGH_DPI, MODEL_ONNX_PATH, DEMO_SGY_PATH
from first_breaks.desktop.picking_widget import PickingWindow
from first_breaks.desktop.warn_widget import WarnBox
from first_breaks.desktop.graph import GraphWidget
from first_breaks.desktop.threads import InitNet, PickerQRunnable
from first_breaks.picking.picker import PickerONNX
from first_breaks.picking.task import Task
from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import calc_hash

warnings.filterwarnings("ignore")

if HIGH_DPI:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


class FileState:
    valid_file = 0
    file_not_exists = 1
    file_changed = 2

    @classmethod
    def get_file_state(cls, fname: Union[str, Path], fhash: str):
        if not Path(fname).is_file():
            return cls.file_not_exists
        else:
            return cls.valid_file if calc_hash(fname) == fhash else cls.file_changed


class ReadyToProcess:
    sgy_selected: bool = False
    model_loaded: bool = False

    def is_ready(self) -> bool:
        return (self.sgy_selected == self.model_loaded) is True


class SliderConverter:
    multiplier = 10

    @classmethod
    def slider2value(cls, slider_value: int) -> float:
        a = slider_value / cls.multiplier
        return a

    @classmethod
    def value2slider(cls, value: float) -> int:
        a = int(cls.multiplier * value)
        return a


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        if getattr(sys, 'frozen', False):
            self.main_folder = Path(sys._MEIPASS)
        else:
            self.main_folder = Path(__file__).parent

        # main window settings
        h, w = self.screen().size().height(), self.screen().size().width()
        left = int(0.2 * w)
        top = int(0.2 * h)
        width = int(0.6 * w)
        height = int(0.6 * h)
        self.setGeometry(left, top, width, height)

        qt_rectangle = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        qt_rectangle.moveCenter(center_point)
        self.move(qt_rectangle.topLeft())

        self.setWindowTitle('First breaks picking')

        # toolbar
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(30, 30))
        self.addToolBar(toolbar)

        # buttons on toolbar
        icon_load_nn = self.style().standardIcon(QStyle.SP_ComputerIcon)
        # icon_load_nn = QIcon(str(self.main_folder / "icons" / "nn.png"))
        self.button_load_nn = QAction(icon_load_nn, "Load model", self)
        self.button_load_nn.triggered.connect(self.load_nn)
        self.button_load_nn.setEnabled(True)
        toolbar.addAction(self.button_load_nn)

        icon_get_filename = self.style().standardIcon(QStyle.SP_DirIcon)
        # icon_get_filename = QIcon(str(self.main_folder / "icons" / "sgy.png"))
        self.button_get_filename = QAction(icon_get_filename, "Open SGY-file", self)
        self.button_get_filename.triggered.connect(self.get_filename)
        self.button_get_filename.setEnabled(True)
        toolbar.addAction(self.button_get_filename)

        toolbar.addSeparator()

        icon_fb = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        # icon_fb = QIcon(str(self.main_folder / "icons" / "picking.png"))
        self.button_fb = QAction(icon_fb, "Neural network FB picking", self)
        self.button_fb.triggered.connect(self.pick_fb)
        self.button_fb.setEnabled(False)
        toolbar.addAction(self.button_fb)

        icon_export = self.style().standardIcon(QStyle.SP_DialogSaveButton)
        # icon_export = QIcon(str(self.main_folder / "icons" / "export.png"))
        self.button_export = QAction(icon_export, "Export picks to file", self)
        # self.button_export.triggered.connect(self.export)
        self.button_export.setEnabled(False)
        toolbar.addAction(self.button_export)

        # icon_export = self.style().standardIcon(QStyle.SP_DialogSaveButton)
        # icon_export = QIcon(str(self.main_folder / "icons" / "export.png"))
        # self.button_export = QAction(icon_export, "Export picks to file", self)
        # self.button_export.triggered.connect(self.export)
        # self.button_export.setEnabled(False)

        toolbar.addSeparator()

        default_gain_value = 1
        self.gain_value = default_gain_value
        self.gain_label = QLabel(str(default_gain_value))
        self.slider_gain = QSlider(Qt.Horizontal)
        self.slider_gain.setRange(SliderConverter.value2slider(-5), SliderConverter.value2slider(5))
        self.slider_gain.setValue(SliderConverter.value2slider(1))
        self.slider_gain.setSingleStep(SliderConverter.value2slider(0.1))
        self.slider_gain.wheelEvent = lambda *args: args[-1].ignore()  # block scrolling with wheel
        self.slider_gain.setMaximumWidth(150)
        self.slider_gain.valueChanged.connect(self.gain_changed)
        self.slider_gain.sliderReleased.connect(self.update_plot)
        toolbar.addWidget(self.slider_gain)
        toolbar.addWidget(self.gain_label)

        self.need_processing_region = True
        icon_processing_show = self.style().standardIcon(QStyle.SP_FileDialogListView)
        # icon_export = QIcon(str(self.main_folder / "icons" / "export.png"))
        self.button_processing_show = QAction(icon_processing_show, "Show processing grid", self)
        self.button_processing_show.triggered.connect(self.processing_region_changed)
        self.button_processing_show.setChecked(self.need_processing_region)
        self.button_processing_show.setEnabled(True)
        self.button_processing_show.setCheckable(True)
        if self.need_processing_region:
            self.button_processing_show.toggle()
        toolbar.addAction(self.button_processing_show)

        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar.addWidget(spacer)

        # self.button_git = QAction(QIcon(str(self.main_folder / "icons" / "github.png")),
        #                           "Open Github repo with project", self)
        # self.button_git.triggered.connect(self.open_github)
        # toolbar.addAction(self.button_git)

        self.status = self.statusBar()
        self.status_progress = QProgressBar()
        self.status_progress.hide()

        self.status_message = QLabel()
        self.status_message.setText('Open SGY file or load model')

        status_widget = QWidget()
        status_layout = QHBoxLayout()
        status_widget.setLayout(status_layout)
        status_layout.addWidget(self.status_progress)
        status_layout.addWidget(self.status_message)

        self.status.addPermanentWidget(status_widget)

        # graph widget
        self.graph = GraphWidget(background='w')
        self.graph.hide()
        self.setCentralWidget(self.graph)

        # picking widget
        self.picking = PickingWindow()
        self.picking.hide()

        # placeholders
        self.sgy = None
        self.fn = None
        self.picks = None
        self.ready_to_process = ReadyToProcess()
        self.picker: Optional[PickerONNX] = None
        self.start_time = None
        self.end_time = None
        self.last_task = None
        self.settings = None

        self.threadpool = QThreadPool()

        self.load_nn(str(MODEL_ONNX_PATH))
        self.get_filename(str(DEMO_SGY_PATH))

        self.show()

    def gain_changed(self, gain_from_slider: int):
        self.gain_value = SliderConverter.slider2value(gain_from_slider)
        self.gain_label.setText(str(self.gain_value))

    def _thread_init_net(self, weights: Union[str, Path]):
        worker = InitNet(weights)
        worker.signals.finished.connect(self.init_net)
        self.threadpool.start(worker)

    def init_net(self, picker: PickerONNX):
        self.picker = picker

    def receive_settings(self, settings: Dict[str, Any]):
        self.settings = settings

    def pick_fb(self):
        settings = PickingWindow(self.last_task)
        settings.export_settings_signal.connect(self.receive_settings)
        settings.exec_()

        if not self.settings:
            return

        try:
            task = Task(self.sgy, **self.settings)
            self.process_task(task)
        except Exception as e:
            window_err = WarnBox(self,
                                 title=e.__class__.__name__,
                                 message=str(e))
            window_err.exec_()

    def process_task(self, task: Task):
        self.button_fb.setEnabled(False)
        self.button_get_filename.setEnabled(False)
        worker = PickerQRunnable(self.picker, task)
        worker.signals.started.connect(self.on_start_task)
        worker.signals.result.connect(self.on_result_task)
        worker.signals.progress.connect(self.on_progressbar_task)
        worker.signals.message.connect(self.on_message_task)
        worker.signals.finished.connect(self.on_finish_task)
        self.threadpool.start(worker)
        self.start_time = time.perf_counter()

    def store_task(self, task: Task):
        self.last_task = task

    def on_start_task(self):
        self.status_progress.show()

    def on_message_task(self, message: str):
        self.status_message.setText(message)

    def on_finish_task(self):
        self.status_progress.hide()
        self.button_fb.setEnabled(True)

    def on_progressbar_task(self, value: int):
        self.status_progress.setValue(value)

    def on_result_task(self, result: Task):
        self.store_task(result)
        if result.success:
            self.graph.plot_picks(self.last_task.picks_in_ms)
            self.run_processing_region()
        else:
            window_error = WarnBox(self, title='InternalError', message=result.error_message)
            window_error.exec_()

        self.button_get_filename.setEnabled(True)
        self.button_fb.setEnabled(True)

    def processing_region_changed(self, toggle: bool):
        self.need_processing_region = toggle
        self.run_processing_region()

    def run_processing_region(self):
        if self.need_processing_region:
            self.show_processing_region()
        else:
            self.hide_processing_region()

    def show_processing_region(self):
        if self.last_task and self.last_task.success:
            self.graph.plot_processing_region(self.last_task.traces_per_gather_parsed,
                                              self.last_task.maximum_time_parsed)

    def hide_processing_region(self):
        if self.last_task and self.last_task.success:
            self.graph.remove_processing_region()

    def show_picks(self):
        if self.last_task and self.last_task.success:
            self.graph.plot_picks(self.last_task.picks_in_ms)

    def update_plot(self, refresh_view: bool = False):
        self.graph.plotseis(self.sgy, gain=self.gain_value, refresh_view=refresh_view)
        self.show_processing_region()
        self.show_picks()

    def unlock_pickng_if_ready(self):
        if self.ready_to_process.is_ready():
            self.button_fb.setEnabled(True)
            self.status_message.setText('Click on picking to start processing')

    def load_nn(self, filename: Optional[str] = None):
        if not filename:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getOpenFileName(self,
                                                      "Select file with NN weights",
                                                      directory=str(Path.home()),
                                                      options=options)

        if filename:
            if FileState.get_file_state(filename, MODEL_ONNX_HASH) == FileState.valid_file:
                self._thread_init_net(weights=filename)
                self.button_load_nn.setEnabled(False)
                self.ready_to_process.model_loaded = True

                status_message = 'Model loaded successfully'
                if not self.ready_to_process.sgy_selected:
                    status_message += ". Open SGY file to start picking"
                self.status_message.setText(status_message)

                self.unlock_pickng_if_ready()
            else:
                window_err = WarnBox(self,
                                     title="Model loading error",
                                     message="The file cannot be used as model weights. "
                                             "Download the file according to the manual and select it.")
                window_err.exec_()

    def get_filename(self, filename: Optional[str] = None):
        if not filename:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getOpenFileName(self,
                                                      "Open SGY-file",
                                                      directory=str(Path.home()),
                                                      filter="SGY-file (*.sgy)",
                                                      options=options)
        if filename:
            try:
                self.fn = Path(filename)
                self.picks = None
                self.last_task = None
                self.sgy = SGY(self.fn, use_delayed_init=False)

                self.graph.clear()
                self.update_plot(refresh_view=True)
                self.graph.show()

                self.button_get_filename.setEnabled(True)
                self.ready_to_process.sgy_selected = True

                if not self.ready_to_process.model_loaded:
                    status_message = "Load model to start picking"
                    self.status_message.setText(status_message)

                self.unlock_pickng_if_ready()

            except Exception as e:
                window_err = WarnBox(self,
                                     title=e.__class__.__name__,
                                     message=str(e))
                window_err.exec_()


def run_app():
    app = QApplication([])
    _ = MainWindow()
    app.exec_()


if __name__ == '__main__':
    run_app()
