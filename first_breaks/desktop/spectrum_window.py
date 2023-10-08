from math import ceil, floor
from typing import Any, Dict, Optional, Tuple

import pyqtgraph as pg
from PyQt5.QtGui import QCloseEvent, QFont

from first_breaks.desktop.roi_manager import RoiManager, get_rect_of_roi
from first_breaks.sgy.reader import SGY
from first_breaks.utils.fourier_transforms import (
    build_amplitude_filter,
    get_mean_amplitude_spectrum,
)
from first_breaks.utils.utils import resolve_xy2postime


class SpectrumWindow(pg.PlotWidget):
    def __init__(
        self, roi_manager: RoiManager, vsp_view: bool = False, use_open_gl: bool = True, *args: Any, **kwargs: Any
    ):
        super().__init__(useOpenGL=use_open_gl, *args, **kwargs)
        self.setWindowTitle("Amplitude spectrum")

        self.getPlotItem().disableAutoRange()
        self.setAntialiasing(True)
        self.getPlotItem().setClipToView(True)
        self.getPlotItem().setDownsampling(mode="peak")
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.7)

        text_size = 12
        self.label_style = {"font-size": f"{text_size}pt"}
        font = QFont()
        font.setPointSize(text_size)
        self.getPlotItem().getAxis("bottom").setLabel("Frequency, Hz", **self.label_style)
        self.getPlotItem().getAxis("left").setLabel("Amplitude", **self.label_style)

        self.setBackground("w")

        self.roi_manager = roi_manager
        self.roi2line: Dict[pg.ROI, pg.PlotCurveItem] = {}
        self.roi2max_spec: Dict[pg.ROI, float] = {}
        self.roi2max_freq: Dict[pg.ROI, float] = {}
        self.sgy: Optional[SGY] = None
        self.f1_f2: Optional[Tuple[float, float]] = None
        self.f3_f4: Optional[Tuple[float, float]] = None
        self.vsp_view = vsp_view

        self.roi_manager.roi_added_signal.connect(self.add_placeholder_line)
        self.roi_manager.roi_clicked_signal.connect(lambda roi: self.show())
        self.roi_manager.roi_changing_signal.connect(self.update_line)
        self.roi_manager.roi_deleted_signal.connect(self.remove_line)
        self.hide()

    def show_and_highlight_window(self) -> None:
        self.show()
        self.setFocus()
        self.raise_()
        self.activateWindow()

    def update_all(self) -> None:
        for roi in self.roi2line.keys():
            self.update_line(roi)

    def set_sgy(self, sgy: SGY) -> None:
        self.sgy = sgy

    def set_filter_params(self, f1_f2: Optional[Tuple[float, float]], f3_f4: Optional[Tuple[float, float]]) -> None:
        self.f1_f2 = f1_f2
        self.f3_f4 = f3_f4
        self.update_all()

    def set_vsp_view(self, vsp_view: bool) -> None:
        self.vsp_view = vsp_view
        self.update_all()

    def add_placeholder_line(self, roi: pg.ROI) -> None:
        line = pg.PlotCurveItem([], [], pen=roi.currentPen)
        self.roi2line[roi] = line
        self.roi2max_freq[roi] = 0
        self.roi2max_spec[roi] = 0
        self.addItem(line)

    def update_line(self, roi: pg.ROI) -> None:
        x_min, y_min, x_max, y_max = get_rect_of_roi(roi)
        x_min, y_min = resolve_xy2postime(self.vsp_view, x_min, y_min)
        x_max, y_max = resolve_xy2postime(self.vsp_view, x_max, y_max)

        x_min_idx = max(0, ceil(x_min - 1))
        x_max_idx = min(self.sgy.num_traces, floor(x_max - 1))

        y_min_idx = max(0, ceil(self.sgy.ms2index(y_min)))
        y_max_idx = min(self.sgy.num_samples, ceil(self.sgy.ms2index(y_max)))
        if x_max_idx >= x_min_idx and y_max_idx > y_min_idx:
            traces = self.sgy.read_traces_by_ids(
                list(range(x_min_idx, x_max_idx + 1)), min_sample=y_min_idx, max_sample=y_max_idx
            )

            frequencies, spectrum = get_mean_amplitude_spectrum(traces, fs=self.sgy.fs)
            amp_filter = build_amplitude_filter(frequencies, f1_f2=self.f1_f2, f3_f4=self.f3_f4)
            spectrum = amp_filter * spectrum

            self.roi2max_freq[roi] = max(frequencies)
            self.roi2max_spec[roi] = max(spectrum)
        else:
            frequencies, spectrum = [], []
            self.roi2max_freq[roi] = 0
            self.roi2max_spec[roi] = 0

        self.roi2line[roi].setData(frequencies, spectrum)
        self.update_limits()
        self.show_and_highlight_window()

    def remove_line(self, roi: pg.ROI) -> None:
        del self.roi2max_freq[roi]
        del self.roi2max_spec[roi]
        self.removeItem(self.roi2line[roi])
        del self.roi2line[roi]
        self.update_limits()
        self.show_and_highlight_window()

    def update_limits(self) -> None:
        if self.roi2max_freq:
            max_freq = 1.05 * max(list(self.roi2max_freq.values()))
        else:
            max_freq = 1

        if self.roi2max_spec:
            max_spec = 1.1 * max(list(self.roi2max_spec.values()))
        else:
            max_spec = 1

        self.getViewBox().setLimits(xMin=0, xMax=max_freq, yMin=0, yMax=max_spec)
        self.getPlotItem().setYRange(0, max_spec)
        self.getPlotItem().setXRange(0, max_freq)

    def closeEvent(self, e: QCloseEvent) -> None:
        e.ignore()
        self.hide()
