from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPen, QPainterPath, QColor

from first_breaks.picker.picker import Task
from first_breaks.sgy.reader import SGY


class GraphWidget(pg.PlotWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.getPlotItem().disableAutoRange()
        self.setAntialiasing(False)
        self.getPlotItem().setClipToView(True)
        self.getPlotItem().setDownsampling(mode='peak')

        self.getPlotItem().invertY(True)
        self.getPlotItem().showAxis('top', True)
        self.getPlotItem().showAxis('bottom', False)
        x_ax = self.getPlotItem().getAxis('top')
        y_ax = self.getPlotItem().getAxis('left')
        text_size = 10
        labelstyle = {'font-size': f'{text_size}pt'}
        font = QFont()
        font.setPointSize(text_size)
        x_ax.setLabel('trace', **labelstyle)
        y_ax.setLabel('t, ms', **labelstyle)
        x_ax.setTickFont(font)
        y_ax.setTickFont(font)

        self.sgy = None

    def plotseis_sgy(self,
                     fname: Path,
                     normalize: bool = True,
                     clip: float = 0.9,
                     amplification: float = 1,
                     negative_patch: bool = True):
        self.clear()
        self.sgy = SGY(fname)
        traces = self.sgy.read()

        if normalize:
            norm_factor = np.mean(np.abs(traces), axis=0)
            norm_factor[np.abs(norm_factor) < 1e-9 * np.max(np.abs(norm_factor))] = 1
            traces = traces / norm_factor

        traces = amplification * traces
        mask_clip = np.abs(traces) > clip
        traces[mask_clip] = clip * np.sign(traces[mask_clip])

        self.plotseis(traces, negative_patch)

    def plotseis(self, traces: np.ndarray, negative_patch: bool = True):
        num_sample, num_traces = np.shape(traces)
        t = np.arange(num_sample) * self.sgy.dt * 1e-3

        self.getViewBox().setLimits(xMin=0, xMax=num_traces + 1, yMin=0, yMax=t[-1])
        self.getPlotItem().setYRange(0, t[-1])
        self.getPlotItem().setXRange(0, num_traces + 1)

        for idx in range(num_traces):
            self.plot_trace_fast(traces[:, idx], t, idx + 1, negative_patch)

    def plot_trace_fast(self, trace: np.ndarray, t: np.ndarray, shift: int, negative_patch: bool):
        connect = np.ones(len(t), dtype=np.int32)
        connect[-1] = 0

        trace[0] = 0
        trace[-1] = 0

        shifted_trace = trace + shift
        path = pg.arrayToQPath(shifted_trace, t, connect)

        item = pg.QtWidgets.QGraphicsPathItem(path)
        pen = QPen(Qt.black, 1, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin)
        pen.setWidth(0.1)
        item.setPen(pen)
        item.setBrush(Qt.white)
        self.addItem(item)

        rect = QPainterPath()

        sign = -1 if negative_patch else 1
        rect.addRect(shift, t[0], sign * 1.1 * max(np.abs(trace)), t[-1])

        patch = path.intersected(rect)
        item = pg.QtWidgets.QGraphicsPathItem(patch)

        pen = QPen(QColor(255, 255, 255, 0), 1, Qt.SolidLine,
                            Qt.FlatCap, Qt.MiterJoin)

        pen.setWidth(0.1)
        item.setPen(pen)
        item.setBrush(Qt.black)
        self.addItem(item)

    def plot_picks(self, task: Task):
        num_traces = self.sgy.shape[1]
        picks = np.array(task.picks)
        ids = np.arange(num_traces) + 1

        path = pg.arrayToQPath(ids, picks, np.ones(num_traces, dtype=np.int32))
        item = pg.QtWidgets.QGraphicsPathItem(path)

        pen = pg.mkPen(color=(255, 0, 0), width=3)
        item.setPen(pen)
        self.addItem(item)
