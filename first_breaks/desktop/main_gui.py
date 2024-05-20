import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from PyQt5.QtCore import QSize, Qt, QThreadPool
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QSizePolicy,
    QStyle,
    QToolBar,
    QWidget,
)

from first_breaks.const import DEMO_SGY_PATH, HIGH_DPI, MODEL_ONNX_HASH, MODEL_ONNX_PATH
from first_breaks.data_models.independent import ExceptionOptional
from first_breaks.desktop.graph import GraphWidget
from first_breaks.desktop.last_folder_manager import last_folder_manager
from first_breaks.desktop.nn_manager import NNManager
from first_breaks.desktop.picks_manager_widget import PicksManager
from first_breaks.desktop.settings_processing_widget import (
    PickingSettings,
    PlotseisSettings,
    SettingsProcessingWidget,
)
from first_breaks.desktop.utils import MessageBox, set_geometry
from first_breaks.picking.task import Task
from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import calc_hash, download_demo_sgy, download_model_onnx

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
    def __init__(self, use_open_gl: bool = True, show: bool = True):  # type: ignore
        super(MainWindow, self).__init__()

        if getattr(sys, "frozen", False):
            self.main_folder = Path(sys._MEIPASS)  # type: ignore
        else:
            self.main_folder = Path(__file__).parent

        set_geometry(self, width_rel=0.6, height_rel=0.6, fix_size=False, centralize=True)
        self.setWindowTitle("First breaks picking")

        self.threadpool = QThreadPool()

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

        icon_visual_settings = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        self.button_settings_processing = QAction(icon_visual_settings, "Settings and Processing", self)
        self.button_settings_processing.triggered.connect(self.show_settings_processing_window)
        self.button_settings_processing.setEnabled(False)
        self.toolbar.addAction(self.button_settings_processing)

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

        icon_picks_manager = self.style().standardIcon(QStyle.SP_FileDialogDetailedView)
        # icon_export = QIcon(str(self.main_folder / "icons" / "export.png"))
        self.button_picks_manager = QAction(icon_picks_manager, "Picks manager", self)
        self.button_picks_manager.triggered.connect(self.show_picks_manager)
        self.button_picks_manager.setEnabled(False)
        self.toolbar.addAction(self.button_picks_manager)

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
        self.plotseis_settings = PlotseisSettings()
        self.settings_processing_widget = SettingsProcessingWidget(
            hide_on_close=True,
            **self.plotseis_settings.model_dump(),
        )
        self.settings_processing_widget.hide()
        self.settings_processing_widget.export_plotseis_settings_signal.connect(self.update_plotseis_settings)
        self.settings_processing_widget.export_picking_settings_signal.connect(self.pick_fb)

        # nn manager
        self.nn_manager = NNManager(
            status_progress=self.status_progress,
            status_message=self.status_message,
            threadpool=self.threadpool,
            interrupt_on=self.settings_processing_widget.interrupt_signal,
        )
        self.nn_manager.picking_finished_signal.connect(self.on_picking_finished)
        self.nn_manager.picking_not_started_error_signal.connect(self.on_picking_not_started_error)

        # picks manager
        self.picks_manager = PicksManager()
        self.picks_manager.picks_updated_signal.connect(self.update_plot)
        self.picks_manager.hide()

        self.graph.picks_manual_edited_signal.connect(self.picks_manager.update_picks_from_external)
        self.graph.about_to_change_nn_picks_signal.connect(self.picks_manager.duplicate_active_created_by_nn_picks)

        self.is_toggled_picks_from_file = False
        # placeholders
        self.sgy: Optional[SGY] = None
        self.fn_sgy: Optional[Union[str, Path]] = None
        self.ready_to_process = ReadyToProcess()
        self.last_task: Optional[Task] = None
        self.settings: Optional[Dict[str, Any]] = None
        self.last_folder: Optional[Union[str, Path]] = None
        self.picks_from_file_in_ms: Optional[Tuple[Union[int, float], ...]] = None
        self.picker_hash = MODEL_ONNX_HASH

        if show:
            self.show()

    def pick_fb(self, settings: PickingSettings) -> None:
        self.button_get_filename.setEnabled(False)
        self.nn_manager.pick_fb(self.sgy, settings)

    def on_picking_not_started_error(self, exc: ExceptionOptional) -> None:
        self.settings_processing_widget.set_selection_mode()
        window_error = MessageBox(
            self,
            title=exc.exception.__class__.__name__,
            message=str(exc),
            detailed_message=exc.get_formatted_traceback(),
        )
        window_error.exec_()

    def on_picking_finished(self, result: Task) -> None:
        self.settings_processing_widget.set_selection_mode()
        self.last_task = result

        if result.success:
            self.picks_manager.add_nn_picks(result.picks)
            self.update_plot(refresh_view=False)
            self.run_processing_region()
        else:
            if isinstance(result.exception, InterruptedError):
                window_error = MessageBox(
                    self,
                    title="Interruption",
                    message="The picking process has been interrupted. Intermediate results will not be saved",
                )
            else:
                window_error = MessageBox(
                    self,
                    title=result.exception.__class__.__name__,
                    message=result.error_message,
                    detailed_message=result.get_formatted_traceback(),
                )
            window_error.exec_()

        self.button_get_filename.setEnabled(True)

    def processing_region_changed(self, toggle: bool) -> None:
        self.need_processing_region = toggle
        self.run_processing_region()

    def run_processing_region(self) -> None:
        if self.need_processing_region:
            self.show_processing_region()
        else:
            self.hide_processing_region()

    def show_processing_region(self) -> None:
        for picks in self.picks_manager.picks_mapping.values():
            if picks.created_by_nn and picks.active:
                tps, max_time = picks.picking_parameters.traces_per_gather, picks.picking_parameters.maximum_time
                self.graph.plot_processing_region(tps, max_time)
                break

    def hide_processing_region(self) -> None:
        if self.last_task and self.last_task.success:
            self.graph.remove_processing_region()

    def update_plotseis_settings(self, new_settings: PlotseisSettings) -> None:
        self.plotseis_settings = new_settings
        self.update_plot(False)

    def update_plot(self, refresh_view: bool = False) -> None:
        self.graph.plotseis(self.sgy, refresh_view=refresh_view, **self.plotseis_settings.model_dump())
        self.show_processing_region()

        self.graph.remove_picks()
        for picks in self.picks_manager.get_selected_picks():
            self.graph.plot_picks(picks)

    def show_settings_processing_window(self) -> None:
        self.settings_processing_widget.show()
        self.settings_processing_widget.focusWidget()

    def unlock_pickng_if_ready(self) -> None:
        if self.ready_to_process.is_ready():
            self.button_settings_processing.setEnabled(True)
            self.settings_processing_widget.enable_only_visualizations_settings()
            self.settings_processing_widget.enable_picking()
            self.status_message.setText("Click on picking to start processing")

    def load_nn(self, filename: Optional[Union[str, Path]] = None) -> None:
        if not filename:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getOpenFileName(
                self, "Select file with NN weights", directory=last_folder_manager.get_last_folder(), options=options
            )

        if filename:
            if FileState.get_file_state(filename, self.picker_hash) == FileState.valid_file:
                self.nn_manager.init_net(weights=filename)
                self.button_load_nn.setEnabled(False)
                self.ready_to_process.model_loaded = True

                status_message = "Model loaded successfully"
                if not self.ready_to_process.sgy_selected:
                    status_message += ". Open SGY file to start picking"
                self.status_message.setText(status_message)
                self.settings_processing_widget.enable_picking()
                self.unlock_pickng_if_ready()
                last_folder_manager.set_last_folder(filename)
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
                directory=last_folder_manager.get_last_folder(),
                filter="SEGY-file (*.segy *.sgy);; Any file (*)",
            )
        if filename:
            try:
                self.fn_sgy = Path(filename)
                self.last_task = None
                self.sgy = SGY(self.fn_sgy)
                self.picks_from_file_in_ms = None
                self.picks_manager.reset_manager()
                self.picks_manager.set_sgy(sgy=self.sgy)

                self.graph.full_clean()
                self.update_plot(refresh_view=True)
                self.graph.show()
                self.button_picks_manager.setEnabled(True)
                self.button_settings_processing.setEnabled(True)
                self.settings_processing_widget.enable_only_visualizations_settings()

                self.button_get_filename.setEnabled(True)
                self.ready_to_process.sgy_selected = True

                if not self.ready_to_process.model_loaded:
                    status_message = "Load model to start picking"
                    self.status_message.setText(status_message)

                self.unlock_pickng_if_ready()
                last_folder_manager.set_last_folder(filename)

            except Exception as e:
                window_err = MessageBox(self, title=e.__class__.__name__, message=str(e))
                window_err.exec_()

    def show_picks_manager(self) -> None:
        if self.picks_manager.isMinimized():
            self.picks_manager.showNormal()
        else:
            self.picks_manager.show()
        self.picks_manager.raise_()
        self.picks_manager.activateWindow()

    def closeEvent(self, e: QCloseEvent) -> None:
        self.graph.spectrum_window.close()
        self.picks_manager.close()
        e.accept()


def create_app() -> Tuple[QApplication, MainWindow]:
    return QApplication([]), MainWindow()


def run_app() -> None:
    app, _ = create_app()
    app.exec_()


def fetch_data_and_run_app() -> None:
    download_model_onnx(MODEL_ONNX_PATH)
    download_demo_sgy(DEMO_SGY_PATH)
    app, window = create_app()
    window.load_nn(MODEL_ONNX_PATH)
    window.get_filename(DEMO_SGY_PATH)
    app.exec_()


if __name__ == "__main__":
    # run_app()
    fetch_data_and_run_app()
