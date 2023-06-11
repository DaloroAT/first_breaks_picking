import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PyQt5.QtCore import QSize, Qt, QThreadPool
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDesktopWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QSizePolicy,
    QSlider,
    QStyle,
    QToolBar,
    QWidget,
)

from first_breaks.const import HIGH_DPI, MODEL_ONNX_HASH
from first_breaks.desktop.graph import GraphWidget
from first_breaks.desktop.picking_widget import PickingWindow
from first_breaks.desktop.threads import InitNet, PickerQRunnable
from first_breaks.desktop.warn_widget import WarnBox
from first_breaks.picking.picker.picker_onnx import PickerONNX
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
    def get_file_state(cls, fname: Union[str, Path], fhash: str) -> int:
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
    def __init__(self):  # type: ignore
        super(MainWindow, self).__init__()

        if getattr(sys, "frozen", False):
            self.main_folder = Path(sys._MEIPASS)  # type: ignore
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

        self.setWindowTitle("First breaks picking")

        # toolbar
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(QSize(30, 30))
        self.addToolBar(self.toolbar)

        # buttons on toolbar
        icon_load_nn = self.style().standardIcon(QStyle.SP_ComputerIcon)
        # icon_load_nn = QIcon(str(self.main_folder / "icons" / "nn.png"))
        self.button_load_nn = QAction(icon_load_nn, "Load model", self)
        self.button_load_nn.triggered.connect(self.load_nn)
        self.button_load_nn.setEnabled(True)
        self.toolbar.addAction(self.button_load_nn)

        icon_get_filename = self.style().standardIcon(QStyle.SP_DirIcon)
        # icon_get_filename = QIcon(str(self.main_folder / "icons" / "sgy.png"))
        self.button_get_filename = QAction(icon_get_filename, "Open SGY-file", self)
        self.button_get_filename.triggered.connect(self.get_filename)
        self.button_get_filename.setEnabled(True)
        self.toolbar.addAction(self.button_get_filename)

        self.toolbar.addSeparator()

        icon_fb = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        # icon_fb = QIcon(str(self.main_folder / "icons" / "picking.png"))
        self.button_fb = QAction(icon_fb, "Neural network FB picking", self)
        self.button_fb.triggered.connect(self.pick_fb)
        self.button_fb.setEnabled(False)
        self.toolbar.addAction(self.button_fb)

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
        self.toolbar.addAction(self.button_processing_show)

        self.toolbar.addSeparator()

        default_gain_value = 1.0
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
        self.toolbar.addWidget(self.slider_gain)
        self.toolbar.addWidget(self.gain_label)

        icon_export = self.style().standardIcon(QStyle.SP_DialogSaveButton)
        # icon_export = QIcon(str(self.main_folder / "icons" / "export.png"))
        self.button_export = QAction(icon_export, "Export picks to file", self)
        self.button_export.triggered.connect(self.export)
        self.button_export.setEnabled(False)
        self.toolbar.addAction(self.button_export)

        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar.addWidget(spacer)

        self.status = self.statusBar()
        self.status_progress = QProgressBar()
        self.status_progress.hide()

        self.status_message = QLabel()
        self.status_message.setText("Open SGY file or load model")

        status_widget = QWidget()
        status_layout = QHBoxLayout()
        status_widget.setLayout(status_layout)
        status_layout.addWidget(self.status_progress)
        status_layout.addWidget(self.status_message)

        self.status.addPermanentWidget(status_widget)

        # graph widget
        self.graph = GraphWidget(background="w")
        self.graph.hide()
        self.setCentralWidget(self.graph)

        # picking widget
        self.picking = PickingWindow()
        self.picking.hide()

        # placeholders
        self.sgy: Optional[SGY] = None
        self.fn_sgy: Optional[Union[str, Path]] = None
        self.ready_to_process = ReadyToProcess()
        self.picker: Optional[PickerONNX] = None
        self.last_task: Optional[Task] = None
        self.settings: Optional[Dict[str, Any]] = None
        self.last_folder: Optional[Union[str, Path]] = None
        self.model_hash = MODEL_ONNX_HASH

        self.threadpool = QThreadPool()

        self.show()

    def get_last_folder(self) -> str:
        if self.last_folder is None:
            return str(Path.home())
        else:
            if Path(self.last_folder).exists():
                return str(Path(self.last_folder).resolve())
            else:
                return str(Path.home())

    def set_last_folder_based_on_file(self, file: Union[str, Path]) -> None:
        file = Path(file)
        if file.exists():
            self.last_folder = str(file.parent)
        else:
            self.last_folder = None

    def gain_changed(self, gain_from_slider: int) -> None:
        self.gain_value = SliderConverter.slider2value(gain_from_slider)
        self.gain_label.setText(str(self.gain_value))

    def _thread_init_net(self, weights: Union[str, Path]) -> None:
        worker = InitNet(weights)
        worker.signals.finished.connect(self.init_net)
        self.threadpool.start(worker)

    def init_net(self, picker: PickerONNX) -> None:
        self.picker = picker

    def receive_settings(self, settings: Dict[str, Any]) -> None:
        self.settings = settings

    def pick_fb(self) -> None:
        settings = PickingWindow(self.last_task)
        settings.export_settings_signal.connect(self.receive_settings)
        settings.exec_()

        if not self.settings:
            return

        try:
            task = Task(self.sgy, **self.settings)
            self.process_task(task)
        except Exception as e:
            window_err = WarnBox(self, title=e.__class__.__name__, message=str(e))
            window_err.exec_()

    def process_task(self, task: Task) -> None:
        self.button_fb.setEnabled(False)
        self.button_get_filename.setEnabled(False)
        worker = PickerQRunnable(self.picker, task)
        worker.signals.started.connect(self.on_start_task)
        worker.signals.result.connect(self.on_result_task)
        worker.signals.progress.connect(self.on_progressbar_task)
        worker.signals.message.connect(self.on_message_task)
        worker.signals.finished.connect(self.on_finish_task)
        self.threadpool.start(worker)

    def store_task(self, task: Task) -> None:
        self.last_task = task

    def on_start_task(self) -> None:
        self.status_progress.show()

    def on_message_task(self, message: str) -> None:
        self.status_message.setText(message)

    def on_finish_task(self) -> None:
        self.status_progress.hide()
        self.button_fb.setEnabled(True)

    def on_progressbar_task(self, value: int) -> None:
        self.status_progress.setValue(value)

    def on_result_task(self, result: Task) -> None:
        self.store_task(result)
        if result.success:
            self.graph.plot_picks(self.last_task.picks_in_ms)
            self.run_processing_region()
            self.button_export.setEnabled(True)
        else:
            window_error = WarnBox(self, title="InternalError", message=result.error_message)
            window_error.exec_()

        self.button_get_filename.setEnabled(True)
        self.button_fb.setEnabled(True)

    def processing_region_changed(self, toggle: bool) -> None:
        self.need_processing_region = toggle
        self.run_processing_region()

    def run_processing_region(self) -> None:
        if self.need_processing_region:
            self.show_processing_region()
        else:
            self.hide_processing_region()

    def show_processing_region(self) -> None:
        if self.last_task and self.last_task.success:
            self.graph.plot_processing_region(
                self.last_task.traces_per_gather_parsed, self.last_task.maximum_time_parsed
            )

    def hide_processing_region(self) -> None:
        if self.last_task and self.last_task.success:
            self.graph.remove_processing_region()

    def show_picks(self) -> None:
        if self.last_task and self.last_task.success:
            self.graph.plot_picks(self.last_task.picks_in_ms)

    def update_plot(self, refresh_view: bool = False) -> None:
        self.graph.plotseis(self.sgy, gain=self.gain_value, refresh_view=refresh_view)
        self.show_processing_region()
        self.show_picks()

    def unlock_pickng_if_ready(self) -> None:
        if self.ready_to_process.is_ready():
            self.button_fb.setEnabled(True)
            self.status_message.setText("Click on picking to start processing")

    def load_nn(self, filename: Optional[str] = None) -> None:
        if not filename:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getOpenFileName(
                self, "Select file with NN weights", directory=self.get_last_folder(), options=options
            )

        if filename:
            if FileState.get_file_state(filename, self.model_hash) == FileState.valid_file:
                self._thread_init_net(weights=filename)
                self.button_load_nn.setEnabled(False)
                self.ready_to_process.model_loaded = True

                status_message = "Model loaded successfully"
                if not self.ready_to_process.sgy_selected:
                    status_message += ". Open SGY file to start picking"
                self.status_message.setText(status_message)

                self.unlock_pickng_if_ready()
                self.set_last_folder_based_on_file(filename)
            else:
                window_err = WarnBox(
                    self,
                    title="Model loading error",
                    message="The file cannot be used as model weights. "
                    "Download the file according to the manual and select it.",
                )
                window_err.exec_()

    def get_filename(self, filename: Optional[str] = None) -> None:
        if not filename:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getOpenFileName(
                self, "Open SEGY-file", directory=self.get_last_folder(), filter="SEGY-file (*.segy *.sgy);; Any file (*)", options=options
            )
        if filename:
            try:
                self.fn_sgy = Path(filename)
                self.last_task = None
                self.sgy = SGY(self.fn_sgy)

                self.graph.clear()
                self.update_plot(refresh_view=True)
                self.graph.show()
                self.button_export.setEnabled(False)

                self.button_get_filename.setEnabled(True)
                self.ready_to_process.sgy_selected = True

                if not self.ready_to_process.model_loaded:
                    status_message = "Load model to start picking"
                    self.status_message.setText(status_message)

                self.unlock_pickng_if_ready()
                self.set_last_folder_based_on_file(filename)

            except Exception as e:
                window_err = WarnBox(self, title=e.__class__.__name__, message=str(e))
                window_err.exec_()

    def export(self) -> None:
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save result", directory=self.get_last_folder(), filter="TXT (*.txt)", options=options
        )

        if filename:
            if self.last_task is not None and self.last_task.success:
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                self.last_task.export_result(str(Path(filename).resolve()), as_plain=True)
                self.set_last_folder_based_on_file(filename)


def run_app() -> None:
    app = QApplication([])
    _ = MainWindow()
    app.exec_()


if __name__ == "__main__":
    run_app()
