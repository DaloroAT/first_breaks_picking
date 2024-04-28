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
    QPushButton,
    QSpinBox,
    QWidget,
)

from first_breaks.const import HIGH_DPI
from first_breaks.data_models.dependent import Device, TraceHeaderParams, XAxis
from first_breaks.data_models.independent import (
    F1F2,
    F3F4,
    Clip,
    FillBlack,
    Gain,
    InvertX,
    InvertY,
    MaximumTime,
    Normalize,
    PicksUnit,
    TNormalize,
    TracesPerGather,
    VSPView,
)
from first_breaks.data_models.initialised_defaults import DEFAULTS
from first_breaks.desktop.bandfilter_widget import QBandFilterWidget
from first_breaks.desktop.combobox_with_mapping import QComboBoxMapping
from first_breaks.desktop.radioset_widget import QRadioSetWidget
from first_breaks.desktop.utils import QHSeparationLine, set_geometry
from first_breaks.utils.cuda import ONNX_CUDA_AVAILABLE, get_recommended_device

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


class PickingSettings(Gain, Clip, Normalize, F1F2, F3F4, MaximumTime, TracesPerGather, Device):
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


def default_set_enabled(widget: QWidget, enabled: bool) -> None:
    widget.setEnabled(enabled)


class _Extras:
    @abstractmethod
    def dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def enable_fields(self) -> None:
        default_set_enabled(self, True)

    def disable_fields(self) -> None:
        default_set_enabled(self, False)


class GainLine(QLineEdit, _Extras):
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


class ClipLine(QLineEdit, _Extras):
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


class NormalizationLine(QComboBoxMapping, _Extras):
    def __init__(self, normalize: TNormalize = DEFAULTS.normalize):
        super().__init__(
            {0: ["Individual traces", "trace"], 1: ["Gather", "gather"], 2: ["Raw", None]},
            current_value=normalize,  # type: ignore
        )

    def dict(self) -> Dict[str, Any]:
        return {"normalize": self.value()}


class BandfilterLine(QWidget, _Extras):
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
        self.bandfilter_widget.enable_freqs_fields()

    def enable_fields(self) -> None:
        self.bandfilter_toogle.setEnabled(True)
        if not self.bandfilter_toogle.isChecked():
            self.bandfilter_widget.enable_freqs_fields()

    def disable_fields(self) -> None:
        self.bandfilter_toogle.setEnabled(False)
        self.bandfilter_widget.disable_freqs_fields()

    def apply_bandfilter(self, checked: bool) -> None:
        if checked:
            freqs = self.bandfilter_widget.validate_and_get_values()
            if freqs is None:
                self.bandfilter_toogle.setChecked(False)
                self.f1_f2 = None
                self.f3_f4 = None
                self.bandfilter_widget.enable_freqs_fields()
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
                self.bandfilter_widget.disable_freqs_fields()
                self.changed_signal.emit()
        else:
            self.f1_f2 = None
            self.f3_f4 = None
            self.bandfilter_widget.enable_freqs_fields()
            self.changed_signal.emit()

    def dict(self) -> Dict[str, Any]:
        return {"f1_f2": self.f1_f2, "f3_f4": self.f3_f4}


class WigglesLine(QRadioSetWidget, _Extras):
    def __init__(self, fill_black: Optional[str] = DEFAULTS.fill_black):
        super().__init__(
            {0: ["Left", "left"], 1: ["Right", "right"], 2: ["No fill", None]}, current_value=fill_black, margins=0
        )

    def dict(self) -> Dict[str, Any]:
        return {"fill_black": self.value()}


class XAxisLine(QComboBoxMapping, _Extras):
    def __init__(self, x_axis: Optional[str] = DEFAULTS.x_axis):
        super().__init__(X_AXIS_MAPPING, current_value=x_axis)  # type: ignore

    def dict(self) -> Dict[str, Any]:
        return {"x_axis": self.value()}


class OrientationLine(QWidget, _Extras):
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
            sub_layout.setSpacing(15)
            sub_layout.setContentsMargins(0, 0, 0, 0)
            self.layout.addLayout(sub_layout)

    def dict(self) -> Dict[str, Any]:
        return {k: w.isChecked() for _, w, _, k in self.widgets}


class TracesPerGatherLine(QSpinBox, _Extras):
    changed_signal = pyqtSignal()

    def __init__(self, traces_per_gather: int = DEFAULTS.traces_per_gather, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.setMinimum(1)
        self.setMaximum(1_000_000)
        self.setValue(traces_per_gather)

    def dict(self) -> Dict[str, Any]:
        return {"traces_per_gather": int(self.text())}


class MaximumTimeLine(QLineEdit, _Extras):
    changed_signal = pyqtSignal()

    def __init__(self, maximum_time: float = DEFAULTS.maximum_time, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        validator = QDoubleValidator()
        validator.setBottom(0.0)
        self.setValidator(validator)
        self.setText(str(maximum_time))

    def dict(self) -> Dict[str, Any]:
        return {"maximum_time": float(self.text())}


class DeviceLine(QComboBoxMapping, _Extras):
    def __init__(self, device: str = DEFAULTS.device):
        if device == "cuda" and ONNX_CUDA_AVAILABLE:
            current_value = device
        else:
            current_value = "cpu"
        super().__init__({0: ["GPU/CUDA", "cuda"], 1: ["CPU", "cpu"]}, current_value=current_value)
        if not ONNX_CUDA_AVAILABLE:
            self.setEnabled(False)

    def dict(self) -> Dict[str, Any]:
        return {"device": self.value()}


class SettingsProcessingWidget(QDialog):
    export_plotseis_settings_signal = pyqtSignal(PlotseisSettings)
    export_picking_settings_signal = pyqtSignal(PickingSettings)
    interrupt_signal = pyqtSignal()

    def __init__(
        self,
        gain: float = 1.0,
        clip: float = 1.0,
        normalize: Optional[str] = "trace",
        f1_f2: Optional[Tuple[float, float]] = None,
        f3_f4: Optional[Tuple[float, float]] = None,
        fill_black: Optional[str] = "left",
        x_axis: Optional[str] = None,
        hide_on_close: bool = False,
        vsp_view: bool = False,
        invert_x: bool = False,
        invert_y: bool = True,
        traces_per_gather: int = 12,
        maximum_time: float = 0.0,
        device: str = get_recommended_device(),
    ):
        super().__init__()
        self.hide_on_close = hide_on_close

        self.setWindowTitle("Settings and Processing")
        self.setWindowModality(Qt.ApplicationModal)
        set_geometry(self, width_rel=0.35, height_rel=0.4, fix_size=False, centralize=True)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self._separators = [
            [QHSeparationLine("Processing"), 0],
            [QHSeparationLine("View"), 5],
            [QHSeparationLine("NN picking"), 10],
        ]

        for sep, line in self._separators:
            self.layout.addWidget(sep, line, 0, 1, 2)

        self._inputs = [
            ("Gain", GainLine(gain=gain), 1, True),
            ("Clip", ClipLine(clip=clip), 2, True),
            ("Normalization", NormalizationLine(normalize=normalize), 3, True),
            ("Band filter", BandfilterLine(f1_f2=f1_f2, f3_f4=f3_f4), 4, True),
            ("Filling wiggles with color", WigglesLine(fill_black=fill_black), 6, True),
            ("X Axis", XAxisLine(x_axis=x_axis), 7, True),
            ("Orientation", OrientationLine(vsp_view=vsp_view, invert_x=invert_x, invert_y=invert_y), 8, True),
            ("Traces per gather", TracesPerGatherLine(traces_per_gather=traces_per_gather), 11, False),
            ("Maximum time", MaximumTimeLine(maximum_time=maximum_time), 12, False),
            ("Runtime", DeviceLine(device=device), 13, False),
        ]

        for label, widget, line, is_plotseis_param in self._inputs:
            label = QLabel(label)
            if is_plotseis_param:
                widget.changed_signal.connect(self.export_plotseis_settings)  # type: ignore
            self.layout.addWidget(label, line, 0)
            self.layout.addWidget(widget, line, 1)

        self.picking_run = False
        self.run_button = QPushButton("Run picking", self)
        self.run_button.clicked.connect(self.picking_click)
        self.layout.addWidget(self.run_button)
        self.disable_picking()

        self.set_selection_mode()

        self.show()

    def enable_only_visualizations_settings(self) -> None:
        self.setEnabled(True)
        self.disable_picking()

    def enable_picking(self) -> None:
        self.run_button.setEnabled(True)

    def disable_picking(self) -> None:
        self.run_button.setEnabled(False)

    def picking_click(self) -> None:
        self.picking_run = not self.picking_run
        if self.picking_run:
            self.set_picking_mode()
            self.export_picking_settings_signal.emit(PickingSettings(**self.get_settings()))
        else:
            self.interrupt_signal.emit()

    def set_picking_mode(self) -> None:
        for widget in self.findChildren(QWidget):
            if isinstance(widget, _Extras):
                widget.disable_fields()
        self.run_button.setEnabled(True)
        self.run_button.setText("Stop")

    def set_selection_mode(self) -> None:
        self.picking_run = False
        for widget in self.findChildren(QWidget):
            if isinstance(widget, _Extras):
                widget.enable_fields()
        self.run_button.setEnabled(True)
        self.run_button.setText("Run picking")

    def get_settings(self) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for _, w, _, _ in self._inputs:
            output.update(w.dict())
        return output

    def export_plotseis_settings(self) -> None:
        settings = self.get_settings()
        print(settings)
        self.export_plotseis_settings_signal.emit(PlotseisSettings(**settings))

    def closeEvent(self, e: QCloseEvent) -> None:
        if self.hide_on_close:
            e.ignore()
            self.hide()
        else:
            e.accept()

    def accept(self) -> None:
        pass


if __name__ == "__main__":
    app = QApplication([])
    window = SettingsProcessingWidget()
    app.exec_()
