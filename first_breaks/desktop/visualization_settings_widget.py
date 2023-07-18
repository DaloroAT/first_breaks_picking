import warnings
from typing import Any, Dict, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
)

from first_breaks.const import HIGH_DPI
from first_breaks.data_models.dependent import TraceHeaderParams, XAxis
from first_breaks.data_models.independent import (
    Clip,
    FillBlack,
    Gain,
    Normalize,
    PicksUnit,
)
from first_breaks.desktop.byte_encode_unit_widget import QByteEncodeUnitWidget
from first_breaks.desktop.combobox_with_mapping import QComboBoxMapping
from first_breaks.desktop.radioset_widget import QRadioSetWidget
from first_breaks.desktop.slider_with_values import QSliderWithValues
from first_breaks.desktop.utils import QHSeparationLine, set_geometry

if HIGH_DPI:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

warnings.filterwarnings("ignore")


def build_x_axis_mapping() -> Dict[int, Tuple[str, str]]:
    mapping = {0: ("Sequentially", None)}
    headers = [
        "TRACENO",
        "CHAN",
        "SOURCE",
        "CDP",
        "OFFSET",
        "SOU_X",
        "SOU_Y",
        "REC_X",
        "REC_Y",
        "CDP_X",
        "CDP_Y",
        "REC_ELEV",
        "SOU_ELEV",
        "DEPTH",
        "REC_DATUM",
        "SOU_DATUM",
    ]
    mapping.update({idx: (f"Header: {h}", h) for idx, h in enumerate(headers, 1)})  # type: ignore
    return mapping  # type: ignore


X_AXIS_MAPPING = build_x_axis_mapping()


class PlotseisSettings(Gain, Clip, Normalize, XAxis, FillBlack):
    pass


class PicksFromFileSettings(TraceHeaderParams, PicksUnit):
    pass


class VisualizationSettingsWidget(QDialog):
    export_plotseis_settings_signal = pyqtSignal(PlotseisSettings)
    export_picks_from_file_settings_signal = pyqtSignal(PicksFromFileSettings)
    toggle_picks_from_file_signal = pyqtSignal(bool)

    def __init__(
        self,
        gain: float = 1.0,
        clip: float = 1.0,
        normalize: Optional[str] = "trace",
        fill_black: Optional[str] = "left",
        x_axis: Optional[str] = None,
        byte_position: int = 1,
        first_byte: int = 1,
        encoding: str = "I",
        picks_unit: str = "mcs",
        hide_on_close: bool = False,
    ):
        super().__init__()
        min_gain = -10
        max_gain = 10
        min_clip = 0.1
        max_clip = 5
        slider_space_fraction = 0.9
        self.hide_on_close = hide_on_close

        self.setWindowTitle("Visualization settings")
        self.setWindowModality(Qt.ApplicationModal)
        set_geometry(self, width_rel=0.35, height_rel=0.3, fix_size=True, centralize=True)

        self.storage_plotseis = {}

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.gain_label = QLabel("Gain")
        self.gain_widget = QSliderWithValues(
            value=gain, min_value=min_gain, max_value=max_gain, slider_space_fraction=slider_space_fraction, margins=0
        )
        self.gain_widget.slider_released_signal.connect(self.export_plotseis_settings)
        self.storage_plotseis["gain"] = self.gain_widget
        self.layout.addWidget(self.gain_label, 0, 0)
        self.layout.addWidget(self.gain_widget, 0, 1)

        self.clip_label = QLabel("Clip")
        self.clip_widget = QSliderWithValues(
            value=clip, min_value=min_clip, max_value=max_clip, slider_space_fraction=slider_space_fraction, margins=0
        )
        self.clip_widget.slider_released_signal.connect(self.export_plotseis_settings)
        self.storage_plotseis["clip"] = self.clip_widget
        self.layout.addWidget(self.clip_label, 1, 0)
        self.layout.addWidget(self.clip_widget, 1, 1)

        self.fill_black_label = QLabel("Filling wiggles with color")
        self.fill_black_widget = QRadioSetWidget(
            {0: ["Left", "left"], 1: ["Right", "right"], 2: ["No fill", None]}, current_value=fill_black
        )
        self.fill_black_widget.changed_signal.connect(self.export_plotseis_settings)
        self.storage_plotseis["fill_black"] = self.fill_black_widget
        self.layout.addWidget(self.fill_black_label, 2, 0)
        self.layout.addWidget(self.fill_black_widget, 2, 1)

        self.normalize_label = QLabel("Normalization")
        self.normalize_widget = QComboBoxMapping(
            {0: ["Individual traces", "trace"], 1: ["Gather", "gather"], 2: ["Raw", None]}, current_value=normalize
        )
        self.normalize_widget.changed_signal.connect(self.export_plotseis_settings)
        self.storage_plotseis["normalize"] = self.normalize_widget
        self.layout.addWidget(self.normalize_label, 3, 0)
        self.layout.addWidget(self.normalize_widget, 3, 1)

        self.xaxis_label = QLabel("X Axis")
        self.xaxis_widget = QComboBoxMapping(X_AXIS_MAPPING, current_value=x_axis)  # type: ignore
        self.xaxis_widget.changed_signal.connect(self.export_plotseis_settings)
        self.storage_plotseis["x_axis"] = self.xaxis_widget
        self.layout.addWidget(self.xaxis_label, 4, 0)
        self.layout.addWidget(self.xaxis_widget, 4, 1)

        self.layout.addWidget(QHSeparationLine(), 5, 0, 1, 2)

        self.picks_from_file_label = QLabel("Show picks from file")
        self.picks_from_file_toggle = QCheckBox()
        self.picks_from_file_toggle.setCheckState(False)
        self.picks_from_file_toggle.stateChanged.connect(self.export_picks_from_file_settings)
        self.picks_from_file_widget = QByteEncodeUnitWidget(
            byte_position=byte_position, first_byte=first_byte, encoding=encoding, picks_unit=picks_unit, margins=0
        )
        self.picks_from_file_widget.values_changed_signal.connect(self.update_picks_from_file_settings)
        self.picks_from_file_settings = self.picks_from_file_widget.get_values()

        self.layout.addWidget(self.picks_from_file_label, 6, 0)
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(self.picks_from_file_toggle)
        sub_layout.addWidget(self.picks_from_file_widget)
        self.layout.addLayout(sub_layout, 6, 1)

        self.show()

    def update_picks_from_file_settings(self, params: Dict[str, Any]) -> None:
        self.picks_from_file_settings = params

    def export_picks_from_file_settings(self) -> None:
        is_pressed = self.picks_from_file_toggle.checkState() == Qt.CheckState.Checked
        if is_pressed:
            self.picks_from_file_widget.setEnabled(False)
            self.export_picks_from_file_settings_signal.emit(PicksFromFileSettings(**self.picks_from_file_settings))
            self.toggle_picks_from_file_signal.emit(True)
        else:
            self.picks_from_file_widget.setEnabled(True)
            self.toggle_picks_from_file_signal.emit(False)

    def get_plotseis_values(self) -> Dict[str, Any]:
        return {k: v.value() for k, v in self.storage_plotseis.items()}

    def export_plotseis_settings(self) -> None:
        settings = self.get_plotseis_values()
        print(settings)
        self.export_plotseis_settings_signal.emit(PlotseisSettings(**settings))

    def closeEvent(self, e: QCloseEvent) -> None:
        if self.hide_on_close:
            e.ignore()
            self.hide()
        else:
            e.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = VisualizationSettingsWidget()
    app.exec_()
