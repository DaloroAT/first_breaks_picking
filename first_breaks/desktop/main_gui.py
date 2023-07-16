import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from PyQt5.QtCore import QSize, Qt, QThreadPool
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
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

from first_breaks.const import DEMO_SGY_PATH, HIGH_DPI, MODEL_ONNX_HASH, MODEL_ONNX_PATH
from first_breaks.desktop.graph import GraphWidget
from first_breaks.desktop.picking_widget import PickingWindow
from first_breaks.desktop.threads import CallInThread, PickerQRunnable
from first_breaks.desktop.utils import MessageBox, set_geometry
from first_breaks.desktop.visualization_settings_widget import VisualizationSettingsWindow, PlotseisSettings
from first_breaks.picking.ipicker import IPicker
from first_breaks.picking.picker_onnx import PickerONNX
from first_breaks.picking.task import Task
from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import (
    calc_hash,
    download_demo_sgy,
    download_model_onnx,
    multiply_iterable_by,
    remove_unused_kwargs,
)

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


class MainWindow(QMainWindow):
    def __init__(self, use_open_gl: bool = True):  # type: ignore
        super(MainWindow, self).__init__()

        if getattr(sys, "frozen", False):
            self.main_folder = Path(sys._MEIPASS)  # type: ignore
        else:
            self.main_folder = Path(__file__).parent

        set_geometry(self, width_rel=0.6, height_rel=0.6, fix_size=False, centralize=True)
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

        icon_visual_settings = self.style().standardIcon(QStyle.SP_BrowserReload)
        self.button_visual_settings = QAction(icon_visual_settings, "Show visual settings", self)
        self.button_visual_settings.triggered.connect(self.show_visual_settings_window)
        self.button_visual_settings.setEnabled(False)
        self.toolbar.addAction(self.button_visual_settings)

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
        self.graph = GraphWidget(use_open_gl=use_open_gl, background="w")
        self.graph.hide()
        self.setCentralWidget(self.graph)

        # visual settings widget
        self.plotseis_settings = {"clip": 0.9, "gain": 1, "normalize": "trace", "x_axis": None}
        self.visual_settings_widget = VisualizationSettingsWindow(hide_on_close=True, **self.plotseis_settings)
        self.visual_settings_widget.hide()
        self.visual_settings_widget.export_plotseis_settings_signal.connect(self.update_plotseis_settings)

        # placeholders
        self.sgy: Optional[SGY] = None
        self.fn_sgy: Optional[Union[str, Path]] = None
        self.ready_to_process = ReadyToProcess()
        self.last_task: Optional[Task] = None
        self.settings: Optional[Dict[str, Any]] = None
        self.last_folder: Optional[Union[str, Path]] = None

        self.picker_class: Type[IPicker] = PickerONNX
        self.picker: Optional[IPicker] = None
        self.picker_hash = MODEL_ONNX_HASH
        self.picker_extra_kwargs_init = {"show_progressbar": False, "device": "cpu"}

        self.picking_window_class = PickingWindow
        self.picking_window_extra_kwargs: Dict[str, Any] = {}

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

    def _thread_init_net(self, weights: Union[str, Path]) -> None:
        task = CallInThread(self.picker_class, model_path=weights, **self.picker_extra_kwargs_init)
        task.signals.result.connect(self.init_net)
        self.threadpool.start(task)

    def init_net(self, picker: PickerONNX) -> None:
        self.picker = picker

    def receive_settings(self, settings: Dict[str, Any]) -> None:
        self.picking_window_extra_kwargs = remove_unused_kwargs(settings, self.picker.change_settings)
        self.settings = settings

    def pick_fb(self) -> None:
        if self.graph.is_picks_modified_manually:
            overwrite_manual_changes_dialog = MessageBox(
                self,
                title="Overwrite manual picking",
                message="There are manual modifications in the current picks. "
                "They will be lost when the new picking starts. "
                "Do you agree?",
                add_cancel_option=True,
            )
            reply = overwrite_manual_changes_dialog.exec_()
            if reply == QDialog.Accepted:
                is_accepted_open_picking_settings = True
            else:
                is_accepted_open_picking_settings = False
        else:
            is_accepted_open_picking_settings = True

        if is_accepted_open_picking_settings:
            picking_settings = self.picking_window_class(task=self.last_task, **self.picking_window_extra_kwargs)
            picking_settings.export_settings_signal.connect(self.receive_settings)
            picking_settings.exec_()
        else:
            return

        if not self.settings:
            return

        try:
            task_kwargs = remove_unused_kwargs(self.settings, Task)
            task = Task(self.sgy, **task_kwargs)
            change_settings_kwargs = remove_unused_kwargs(self.settings, self.picker_class.change_settings)
            self.picker.change_settings(**change_settings_kwargs)
            self.process_task(task)
        except Exception as e:
            window_err = MessageBox(self, title=e.__class__.__name__, message=str(e))
            window_err.exec_()

    def process_task(self, task: Task) -> None:
        self.button_fb.setEnabled(False)
        self.button_get_filename.setEnabled(False)
        worker = PickerQRunnable(self.picker, task)
        worker.signals.started.connect(self.on_start_task)
        worker.signals.result.connect(self.on_result_task)
        worker.signals.progress.connect(self.on_progressbar_task)
        worker.signals.message.connect(self.on_message_task)
        worker.signals.result.connect(self.on_finish_task)
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
            window_error = MessageBox(self, title="InternalError", message=result.error_message)
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

    # def update_plotseis_settings(self, new_settings: Dict[str, Any]) -> None:
    #     self.plotseis_settings = new_settings
    #     self.update_plot(False)

    def update_plotseis_settings(self, new_settings: PlotseisSettings) -> None:
        self.plotseis_settings = new_settings.model_dump()
        self.update_plot(False)

    def update_plot(self, refresh_view: bool = False) -> None:
        self.graph.plotseis(self.sgy, refresh_view=refresh_view, **self.plotseis_settings)
        self.show_processing_region()
        self.show_picks()

    def show_visual_settings_window(self):
        self.visual_settings_widget.show()
        self.visual_settings_widget.focusWidget()

    def unlock_pickng_if_ready(self) -> None:
        if self.ready_to_process.is_ready():
            self.button_fb.setEnabled(True)
            self.status_message.setText("Click on picking to start processing")

    def load_nn(self, filename: Optional[Union[str, Path]] = None) -> None:
        if not filename:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getOpenFileName(
                self, "Select file with NN weights", directory=self.get_last_folder(), options=options
            )

        if filename:
            if FileState.get_file_state(filename, self.picker_hash) == FileState.valid_file:
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
                window_err = MessageBox(
                    self,
                    title="Model loading error",
                    message="The file cannot be used as model weights. "
                    "Download the file according to the manual and select it.",
                )
                window_err.exec_()

    def get_filename(self, filename: Optional[Union[str, Path]] = None) -> None:
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Open SEGY-file",
                directory=self.get_last_folder(),
                filter="SEGY-file (*.segy *.sgy);; Any file (*)",
            )
        if filename:
            try:
                self.fn_sgy = Path(filename)
                self.last_task = None
                self.sgy = SGY(self.fn_sgy)

                self.graph.full_clean()
                self.update_plot(refresh_view=True)
                self.graph.show()
                self.button_export.setEnabled(False)
                self.button_visual_settings.setEnabled(True)

                self.button_get_filename.setEnabled(True)
                self.ready_to_process.sgy_selected = True

                if not self.ready_to_process.model_loaded:
                    status_message = "Load model to start picking"
                    self.status_message.setText(status_message)

                self.unlock_pickng_if_ready()
                self.set_last_folder_based_on_file(filename)

            except Exception as e:
                window_err = MessageBox(self, title=e.__class__.__name__, message=str(e))
                window_err.exec_()

    def export(self) -> None:
        formats = ["SEGY-file (*.segy *.sgy)", "JSON-file (*.json)", "TXT-file (*.txt)"]
        formats = ";; ".join(formats)
        filename, _ = QFileDialog.getSaveFileName(self, "Save result", directory=self.get_last_folder(), filter=formats)

        if filename:
            filename = Path(filename).resolve()
            if self.last_task is not None and self.last_task.success:
                filename.parent.mkdir(parents=True, exist_ok=True)

                picks_in_samples_prev = self.last_task.picks_in_samples
                if self.graph.is_picks_modified_manually:
                    self.last_task.picks_in_samples = multiply_iterable_by(
                        self.graph.picks_in_ms, 1 / self.sgy.dt_ms, int
                    )
                if filename.suffix.lower() in (".sgy", ".segy"):
                    self.last_task.export_result(str(filename), as_sgy=True)  # type: ignore
                elif filename.suffix.lower() == ".txt":
                    self.last_task.export_result(str(filename), as_plain=True)
                elif filename.suffix.lower() == ".json":
                    self.last_task.export_result(str(filename), as_json=True)
                else:
                    message_er = "The file can only be saved in '.sgy', '.segy', '.txt, or '.json' formats"
                    window_err = MessageBox(self, title="Wrong filename", message=message_er)
                    window_err.exec_()
                if self.graph.is_picks_modified_manually:
                    self.last_task.picks_in_samples = picks_in_samples_prev


def run_app() -> None:
    app = QApplication([])
    _ = MainWindow()
    app.exec_()


def fetch_data_and_run_app() -> None:
    download_model_onnx(MODEL_ONNX_PATH)
    download_demo_sgy(DEMO_SGY_PATH)
    app = QApplication([])
    window = MainWindow()
    window.load_nn(MODEL_ONNX_PATH)
    window.get_filename(DEMO_SGY_PATH)
    app.exec_()


if __name__ == "__main__":
    fetch_data_and_run_app()
