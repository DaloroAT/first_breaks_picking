import warnings
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QDoubleValidator
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QWidget, QSpinBox,
)

from first_breaks.const import HIGH_DPI
from first_breaks.data_models.dependent import TraceHeaderParams, XAxis
from first_breaks.data_models.independent import (
    F1F2,
    F3F4,
    Clip,
    FillBlack,
    Gain,
    InvertX,
    InvertY,
    Normalize,
    PicksUnit,
    VSPView,
)
from first_breaks.data_models.initialised_defaults import DEFAULTS
from first_breaks.desktop.bandfilter_widget import QBandFilterWidget
from first_breaks.desktop.byte_encode_unit_widget import QByteEncodeUnitWidget
from first_breaks.desktop.combobox_with_mapping import QComboBoxMapping
from first_breaks.desktop.radioset_widget import QRadioSetWidget
from first_breaks.desktop.utils import QHSeparationLine, set_geometry
from first_breaks.utils.cuda import ONNX_CUDA_AVAILABLE

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


class PlotseisSettings(Gain, Clip, Normalize, XAxis, FillBlack, F1F2, F3F4, VSPView, InvertX, InvertY):
    pass


class PicksFromFileSettings(TraceHeaderParams, PicksUnit):
    pass


def get_value(qline: QLineEdit, minimum: Optional[float] = None, default: Optional[float] = 1.0) -> float:
    value = qline.text()
    qline.setPlaceholderText(str(default))
    if value.lstrip("-").lstrip("+").lstrip(".").lstrip(","):
        value = float(value)
        if minimum:
            value = max(minimum, value)
        return value
    else:
        return default


class _Dictable:
    @abstractmethod
    def dict(self) -> Dict[str, Any]:
        raise NotImplementedError


class GainLine(QLineEdit, _Dictable):
    changed_signal = pyqtSignal()

    def __init__(self, gain: float = DEFAULTS.gain, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.default = DEFAULTS.gain
        self.setPlaceholderText(str(self.default))
        gain_validator = QDoubleValidator()
        self.setValidator(gain_validator)
        self.setText(str(gain))
        self.textChanged.connect(self.changed_signal)

    def dict(self) -> Dict[str, Any]:
        value = self.text()
        if value.lstrip("-").lstrip("+").lstrip(".").lstrip(","):
            value = float(value)
        else:
            value = self.default
        return {"gain": value}


class ClipLine(QLineEdit, _Dictable):
    changed_signal = pyqtSignal()

    def __init__(self, clip: float = DEFAULTS.clip, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.default = DEFAULTS.clip
        self.minimum = 0.1
        self.setPlaceholderText(str(self.default))

        clip_validator = QDoubleValidator()
        clip_validator.setBottom(0.1)
        self.setValidator(clip_validator)
        self.setText(str(clip))
        self.textChanged.connect(self.changed_signal)

    def dict(self) -> Dict[str, Any]:
        value = self.text()
        if value.lstrip("-").lstrip("+").lstrip(".").lstrip(","):
            value = float(value)
            value = max(self.minimum, value)
        else:
            value = self.default

        return {"clip": value}


class NormalizationLine(QComboBoxMapping, _Dictable):
    def __init__(self, normalize: Optional[str] = DEFAULTS.normalize):
        super().__init__(
            {0: ["Individual traces", "trace"], 1: ["Gather", "gather"], 2: ["Raw", None]}, current_value=normalize
        )

    def dict(self) -> Dict[str, Any]:
        return {"normalize": self.value()}


class BandfilterLine(QWidget, _Dictable):
    changed_signal = pyqtSignal()

    def __init__(
        self,
        f1_f2: Optional[Tuple[float, float]] = DEFAULTS.f1_f2,
        f3_f4: Optional[Tuple[float, float]] = DEFAULTS.f3_f4,
    ):
        super().__init__()
        self.bandfilter_toogle = QCheckBox()
        self.bandfilter_toogle.setCheckState(False)
        self.bandfilter_toogle.clicked.connect(self.apply_bandfilter)
        if f1_f2:
            f1, f2 = f1_f2
        else:
            f1, f2 = None, None
        if f3_f4:
            f3, f4 = f3_f4
        else:
            f3, f4 = None, None
        self.bandfilter_widget = QBandFilterWidget(f1=f1, f2=f2, f3=f3, f4=f4, debug=False)
        layout = QHBoxLayout()
        layout.addWidget(self.bandfilter_toogle)
        layout.addWidget(self.bandfilter_widget)
        self.setLayout(layout)
        self.f1_f2 = f1_f2
        self.f3_f4 = f3_f4
        self.bandfilter_widget.enbale_fields()

    def apply_bandfilter(self, checked):
        if checked:
            freqs = self.bandfilter_widget.validate_and_get_values()
            if freqs is None:
                self.bandfilter_toogle.setChecked(False)
                self.f1_f2 = None
                self.f3_f4 = None
                self.bandfilter_widget.enbale_fields()
            else:
                self.bandfilter_toogle.setChecked(True)
                if freqs[1] is None and freqs[2] is None:
                    self.f1_f2 = None
                else:
                    self.f1_f2 = (freqs[1], freqs[2])
                if freqs[3] is None and freqs[4] is None:
                    self.f3_f4 = None
                else:
                    self.f3_f4 = (freqs[3], freqs[4])
                self.bandfilter_widget.disable_fields()
                self.changed_signal.emit()
        else:
            self.f1_f2 = None
            self.f3_f4 = None
            self.bandfilter_widget.enbale_fields()
            self.changed_signal.emit()

    def dict(self) -> Dict[str, Any]:
        return {"f1_f2": self.f1_f2, "f3_f4": self.f3_f4}


class WigglesLine(QRadioSetWidget, _Dictable):
    def __init__(self, fill_black: Optional[str] = DEFAULTS.fill_black):
        super().__init__(
            {0: ["Left", "left"], 1: ["Right", "right"], 2: ["No fill", None]}, current_value=fill_black, margins=0
        )

    def dict(self) -> Dict[str, Any]:
        return {"fill_black": self.value()}


class XAxisLine(QComboBoxMapping, _Dictable):
    def __init__(self, x_axis: Optional[str] = DEFAULTS.x_axis):
        super().__init__(X_AXIS_MAPPING, current_value=x_axis)

    def dict(self) -> Dict[str, Any]:
        return {"x_axis": self.value()}


class OrientationLine(QWidget, _Dictable):
    changed_signal = pyqtSignal()

    def __init__(
        self, vsp_view: bool = DEFAULTS.vsp_view, invert_x: bool = DEFAULTS.invert_x, invert_y: bool = DEFAULTS.invert_y
    ):
        super().__init__()

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.widgets = [
            ["VSP view", QCheckBox(), vsp_view, "vsp_view"],
            ["Invert X", QCheckBox(), invert_x, "invert_x"],
            ["Invert Y", QCheckBox(), invert_y, "invert_y"],
        ]

        for label, widget, init, _ in self.widgets:
            label = QLabel(label)
            widget.setChecked(init)
            widget.stateChanged.connect(self.changed_signal.emit)
            sub_layout = QHBoxLayout()
            sub_layout.addWidget(widget, alignment=Qt.AlignRight)
            sub_layout.addWidget(label, alignment=Qt.AlignLeft)
            sub_layout.setSpacing(15)  # set spacing to 0
            sub_layout.setContentsMargins(0, 0, 0, 0)
            self.layout.addLayout(sub_layout)

    def dict(self) -> Dict[str, Any]:
        return {k: w.isChecked() for _, w, _, k in self.widgets}


class PicksFromFileLine(QWidget):
    toggle_picks_from_file_signal = pyqtSignal(bool)
    export_picks_from_file_settings_signal = pyqtSignal(PicksFromFileSettings)

    def __init__(self, byte_position: int = 1, first_byte: int = 1, encoding: str = "I", picks_unit: str = "mcs"):

        super().__init__()
        self.picks_from_file_toggle = QCheckBox()
        self.picks_from_file_toggle.setCheckState(False)
        self.picks_from_file_toggle.stateChanged.connect(self.export_picks_from_file_settings)
        self.picks_from_file_widget = QByteEncodeUnitWidget(
            byte_position=byte_position, first_byte=first_byte, encoding=encoding, picks_unit=picks_unit, margins=0
        )
        self.picks_from_file_widget.values_changed_signal.connect(self.update_picks_from_file_settings)
        self.picks_from_file_settings = self.picks_from_file_widget.get_values()

        sub_layout = QHBoxLayout()
        sub_layout.addWidget(self.picks_from_file_toggle)
        sub_layout.addWidget(self.picks_from_file_widget)
        self.setLayout(sub_layout)

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


class TracesPerGatherLine(QSpinBox, _Dictable):
    changed_signal = pyqtSignal()

    def __init__(self, traces_per_gather: int = DEFAULTS.traces_per_gather, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.setRange(1, 99999999999)
        self.setValue(traces_per_gather)

    def dict(self) -> Dict[str, Any]:
        return {"traces_per_gather": int(self.text())}


class MaximumTimeLine(QLineEdit, _Dictable):
    changed_signal = pyqtSignal()

    def __init__(self, maximum_time: int = DEFAULTS.maximum_time, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        validator = QDoubleValidator()
        validator.setBottom(0.0)
        self.setValidator(validator)
        self.setText(str(maximum_time))

    def dict(self) -> Dict[str, Any]:
        return {"maximum_time": float(self.text())}


class DeviceBatchSizeLine(QWidget, _Dictable):
    changed_signal = pyqtSignal()

    def __init__(self, device: str = DEFAULTS.device, batch_size: int = DEFAULTS.batch_size):
        super().__init__()
        # super().__init__({0: ["GPU/CUDA", "cuda"], 1: ["CPU", "cpu"]}, current_value=device)


        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        if device == "cuda" and ONNX_CUDA_AVAILABLE:
            current_value = device
        else:
            current_value = "cpu"


        # self.layout = QHBoxLayout()
        # self.setLayout(self.layout)
        #
        # self.widgets = [
        #     ["VSP view", QCheckBox(), vsp_view, "vsp_view"],
        #     ["Invert X", QCheckBox(), invert_x, "invert_x"],
        #     ["Invert Y", QCheckBox(), invert_y, "invert_y"],
        # ]
        #
        # for label, widget, init, _ in self.widgets:
        #     label = QLabel(label)
        #     widget.setChecked(init)
        #     widget.stateChanged.connect(self.changed_signal.emit)
        #     sub_layout = QHBoxLayout()
        #     sub_layout.addWidget(widget, alignment=Qt.AlignRight)
        #     sub_layout.addWidget(label, alignment=Qt.AlignLeft)
        #     sub_layout.setSpacing(15)  # set spacing to 0
        #     sub_layout.setContentsMargins(0, 0, 0, 0)
        #     self.layout.addLayout(sub_layout)

        self.device_


# class DeviceLine(QComboBoxMapping, _Dictable):
#     def __init__(self, device: str = DEFAULTS.device):
#         super().__init__({0: ["GPU/CUDA", "cuda"], 1: ["CPU", "cpu"]}, current_value=device)
#         if not ONNX_CUDA_AVAILABLE:
#             self.device.setEnabled(False)
#
#
#     def dict(self) -> Dict[str, Any]:
#         return {"device": self.value()}



class VisualizationSettingsWidget(QDialog):
    export_plotseis_settings_signal = pyqtSignal(PlotseisSettings)
    export_picks_from_file_settings_signal = pyqtSignal(PicksFromFileSettings)
    toggle_picks_from_file_signal = pyqtSignal(bool)

    def __init__(
        self,
        gain: float = 1.0,
        clip: float = 1.0,
        normalize: Optional[str] = "trace",
        f1_f2: Optional[Tuple[float, float]] = None,
        f3_f4: Optional[Tuple[float, float]] = None,
        fill_black: Optional[str] = "left",
        x_axis: Optional[str] = None,
        byte_position: int = 1,
        first_byte: int = 1,
        encoding: str = "I",
        picks_unit: str = "mcs",
        hide_on_close: bool = False,
        vsp_view: bool = False,
        invert_x: bool = False,
        invert_y: bool = True,
    ):
        super().__init__()
        self.hide_on_close = hide_on_close

        self.setWindowTitle("Visualization settings")
        self.setWindowModality(Qt.ApplicationModal)
        set_geometry(self, width_rel=0.35, height_rel=0.4, fix_size=False, centralize=True)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self._plotseis_inputs = [
            ["Gain", GainLine(gain=gain), 0],
            ["Clip", ClipLine(clip=clip), 1],
            ["Normalization", NormalizationLine(normalize=normalize), 2],
            ["Band filter", BandfilterLine(f1_f2=f1_f2, f3_f4=f3_f4), 3],
            ["Filling wiggles with color", WigglesLine(fill_black=fill_black), 5],
            ["X Axis", XAxisLine(x_axis=x_axis), 6],
            ["Orientation", OrientationLine(vsp_view=vsp_view, invert_x=invert_x, invert_y=invert_y), 7],
        ]

        for label, widget, line in self._plotseis_inputs:
            label = QLabel(label)
            widget.changed_signal.connect(self.export_plotseis_settings)
            self.layout.addWidget(label, line, 0)
            self.layout.addWidget(widget, line, 1)

        self._separators = [[QHSeparationLine(), 4], [QHSeparationLine(), 8]]

        for sep, line in self._separators:
            self.layout.addWidget(sep, line, 0, 1, 2)

        picks_from_file_label = QLabel("Show picks from file")
        picks_from_file_widget = PicksFromFileLine(
            byte_position=byte_position, first_byte=first_byte, encoding=encoding, picks_unit=picks_unit
        )
        picks_from_file_widget.toggle_picks_from_file_signal.connect(self.toggle_picks_from_file_signal)
        picks_from_file_widget.export_picks_from_file_settings_signal.connect(
            self.export_picks_from_file_settings_signal
        )
        self.layout.addWidget(picks_from_file_label, 9, 0)
        self.layout.addWidget(picks_from_file_widget, 9, 1)

        self.show()

    def get_plotseis_values(self) -> Dict[str, Any]:
        output = {}
        for _, w, _ in self._plotseis_inputs:
            output.update(w.dict())
        return output

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


# class VisualizationSettingsWidget(QDialog):
#     export_plotseis_settings_signal = pyqtSignal(PlotseisSettings)
#     export_picks_from_file_settings_signal = pyqtSignal(PicksFromFileSettings)
#     toggle_picks_from_file_signal = pyqtSignal(bool)
#
#     def __init__(
#         self,
#         gain: float = 1.0,
#         clip: float = 1.0,
#         normalize: Optional[str] = "trace",
#         fill_black: Optional[str] = "left",
#         x_axis: Optional[str] = None,
#         byte_position: int = 1,
#         first_byte: int = 1,
#         encoding: str = "I",
#         picks_unit: str = "mcs",
#         hide_on_close: bool = False,
#     ):
#         super().__init__()
#         self.hide_on_close = hide_on_close
#
#         self.setWindowTitle("Visualization settings")
#         self.setWindowModality(Qt.ApplicationModal)
#         set_geometry(self, width_rel=0.35, height_rel=0.3, fix_size=True, centralize=True)
#
#         self.storage_plotseis = {}
#
#         self.layout = QGridLayout()
#         self.setLayout(self.layout)
#
#         self.gain_label = QLabel("Gain")
#         self.gain_widget = QLineEdit()
#         gain_validator = QDoubleValidator()
#         self.gain_widget.setValidator(gain_validator)
#         self.gain_widget.setText(str(gain))
#         self.gain_widget.value = lambda *args, **kwargs: get_value(self.gain_widget, default=DEFAULTS.gain)
#         self.gain_widget.textChanged.connect(self.export_plotseis_settings)
#
#         self.storage_plotseis["gain"] = self.gain_widget
#         self.layout.addWidget(self.gain_label, 0, 0)
#         self.layout.addWidget(self.gain_widget, 0, 1)
#
#         self.clip_label = QLabel("Clip")
#         self.clip_widget = QLineEdit()
#         clip_validator = QDoubleValidator()
#         clip_validator.setBottom(0.1)
#         self.clip_widget.setValidator(clip_validator)
#         self.clip_widget.setText(str(clip))
#         self.clip_widget.value = lambda *args, **kwargs: get_value(self.clip_widget, minimum=0.1, default=DEFAULTS.clip)
#         self.clip_widget.textEdited.connect(self.export_plotseis_settings)
#
#         self.storage_plotseis["clip"] = self.clip_widget
#         self.layout.addWidget(self.clip_label, 1, 0)
#         self.layout.addWidget(self.clip_widget, 1, 1)
#
#         self.normalize_label = QLabel("Normalization")
#         self.normalize_widget = QComboBoxMapping(
#             {0: ["Individual traces", "trace"], 1: ["Gather", "gather"], 2: ["Raw", None]}, current_value=normalize
#         )
#         self.normalize_widget.changed_signal.connect(self.export_plotseis_settings)
#         self.storage_plotseis["normalize"] = self.normalize_widget
#         self.layout.addWidget(self.normalize_label, 2, 0)
#         self.layout.addWidget(self.normalize_widget, 2, 1)
#
#         self.bandfilter_label = QLabel("Band filter")
#         self.bandfilter_toogle = QCheckBox()
#         self.bandfilter_toogle.setCheckState(False)
#         self.bandfilter_toogle.stateChanged.connect(self.apply_bandfilter)
#         self.bandfilter_widget = QBandFilterWidget()
#         self.layout.addWidget(self.bandfilter_label, 3, 0)
#         sub_layout = QHBoxLayout()
#         sub_layout.addWidget(self.bandfilter_toogle)
#         sub_layout.addWidget(self.bandfilter_widget)
#         self.layout.addLayout(sub_layout, 3, 1)
#
#         self.layout.addWidget(QHSeparationLine(), 4, 0, 1, 2)
#
#         self.fill_black_label = QLabel("Filling wiggles with color")
#         self.fill_black_widget = QRadioSetWidget(
#             {0: ["Left", "left"], 1: ["Right", "right"], 2: ["No fill", None]}, current_value=fill_black, margins=0
#         )
#         self.fill_black_widget.changed_signal.connect(self.export_plotseis_settings)
#         self.storage_plotseis["fill_black"] = self.fill_black_widget
#         self.layout.addWidget(self.fill_black_label, 5, 0)
#         self.layout.addWidget(self.fill_black_widget, 5, 1)
#
#         self.xaxis_label = QLabel("X Axis")
#         self.xaxis_widget = QComboBoxMapping(X_AXIS_MAPPING, current_value=x_axis)  # type: ignore
#         self.xaxis_widget.changed_signal.connect(self.export_plotseis_settings)
#         self.storage_plotseis["x_axis"] = self.xaxis_widget
#         self.layout.addWidget(self.xaxis_label, 6, 0)
#         self.layout.addWidget(self.xaxis_widget, 6, 1)
#
#         self.layout.addWidget(QHSeparationLine(), 7, 0, 1, 2)
#
#         self.picks_from_file_label = QLabel("Show picks from file")
#         self.picks_from_file_toggle = QCheckBox()
#         self.picks_from_file_toggle.setCheckState(False)
#         self.picks_from_file_toggle.stateChanged.connect(self.export_picks_from_file_settings)
#         self.picks_from_file_widget = QByteEncodeUnitWidget(
#             byte_position=byte_position, first_byte=first_byte, encoding=encoding, picks_unit=picks_unit, margins=0
#         )
#         self.picks_from_file_widget.values_changed_signal.connect(self.update_picks_from_file_settings)
#         self.picks_from_file_settings = self.picks_from_file_widget.get_values()
#
#         self.layout.addWidget(self.picks_from_file_label, 8, 0)
#         sub_layout = QHBoxLayout()
#         sub_layout.addWidget(self.picks_from_file_toggle)
#         sub_layout.addWidget(self.picks_from_file_widget)
#         self.layout.addLayout(sub_layout, 8, 1)
#
#         self.show()
#
#     def update_picks_from_file_settings(self, params: Dict[str, Any]) -> None:
#         self.picks_from_file_settings = params
#
#     def export_picks_from_file_settings(self) -> None:
#         is_pressed = self.picks_from_file_toggle.checkState() == Qt.CheckState.Checked
#         if is_pressed:
#             self.picks_from_file_widget.setEnabled(False)
#             self.export_picks_from_file_settings_signal.emit(PicksFromFileSettings(**self.picks_from_file_settings))
#             self.toggle_picks_from_file_signal.emit(True)
#         else:
#             self.picks_from_file_widget.setEnabled(True)
#             self.toggle_picks_from_file_signal.emit(False)
#
#     def get_plotseis_values(self) -> Dict[str, Any]:
#         return {k: v.value() for k, v in self.storage_plotseis.items()}
#
#     def export_plotseis_settings(self) -> None:
#         settings = self.get_plotseis_values()
#         print(settings)
#         self.export_plotseis_settings_signal.emit(PlotseisSettings(**settings))
#
#     def closeEvent(self, e: QCloseEvent) -> None:
#         if self.hide_on_close:
#             e.ignore()
#             self.hide()
#         else:
#             e.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = VisualizationSettingsWidget()
    app.exec_()
