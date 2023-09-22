from math import ceil, floor
from typing import Any, Optional, Tuple

import pyqtgraph as pg
from PyQt5.QtGui import QCloseEvent, QFont

from first_breaks.desktop.roi_manager import RoiManager, get_rect_of_roi
from first_breaks.utils.fourier_transforms import get_mean_amplitude_spectrum, build_amplitude_filter


class SpectrumWindow(pg.PlotWidget):
    def __init__(self, roi_manager: RoiManager, use_open_gl: bool = True, *args: Any, **kwargs: Any):
        super().__init__(useOpenGL=use_open_gl, *args, **kwargs)
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

        self.setBackground('w')

        self.roi_manager = roi_manager
        self.roi2line = {}
        self.roi2max_spec = {}
        self.roi2max_freq = {}
        self.sgy = None
        self.f1_f2 = None
        self.f3_f4 = None

        self.roi_manager.roi_added_signal.connect(self.add_placeholder_line)
        self.roi_manager.roi_clicked_signal.connect(lambda roi: self.show())
        self.roi_manager.roi_changing_signal.connect(self.update_line)
        self.roi_manager.roi_deleted_signal.connect(self.remove_line)
        self.hide()

    def show_and_highlight_window(self):
        self.show()
        self.setFocus()
        self.raise_()
        self.activateWindow()

    def set_sgy(self, sgy):
        self.sgy = sgy

    def set_filter_params(self, f1_f2: Optional[Tuple[float, float]], f3_f4: Optional[Tuple[float, float]]):
        self.f1_f2 = f1_f2
        self.f3_f4 = f3_f4

        for roi in self.roi2line.keys():
            self.update_line(roi)

    def add_placeholder_line(self, roi: pg.ROI):
        line = pg.PlotCurveItem([], [], pen=roi.currentPen)
        self.roi2line[roi] = line
        self.roi2max_freq[roi] = 0
        self.roi2max_spec[roi] = 0
        self.addItem(line)

    def update_line(self, roi: pg.ROI):
        x_min, y_min, x_max, y_max = get_rect_of_roi(roi)
        x_min_idx = max(0, ceil(x_min - 1))
        x_max_idx = min(self.sgy.num_traces, floor(x_max - 1))

        y_min_idx = max(0, ceil(self.sgy.ms2index(y_min)))
        y_max_idx = min(self.sgy.num_samples, ceil(self.sgy.ms2index(y_max)))
        if x_max_idx >= x_min_idx and y_max_idx > y_min_idx:
            traces = self.sgy.read_traces_by_ids(list(range(x_min_idx, x_max_idx + 1)),
                                                 min_sample=y_min_idx,
                                                 max_sample=y_max_idx)

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

    def remove_line(self, roi: pg.ROI):
        del self.roi2max_freq[roi]
        del self.roi2max_spec[roi]
        self.removeItem(self.roi2line[roi])
        del self.roi2line[roi]
        self.update_limits()
        self.show_and_highlight_window()

    def update_limits(self):
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
