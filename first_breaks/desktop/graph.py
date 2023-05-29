import warnings
from pathlib import Path
from typing import Union, Tuple, Sequence, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QFont, QPen, QPainterPath, QColor
from PyQt5.QtWidgets import QApplication
from pyqtgraph.exporters import ImageExporter

from first_breaks.picking.task import Task
from first_breaks.sgy.reader import SGY


TColor = Union[Tuple[int, int, int, int], Tuple[int, int, int]]


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
        text_size = 12
        labelstyle = {'font-size': f'{text_size}pt'}
        font = QFont()
        font.setPointSize(text_size)
        x_ax.setLabel('trace', **labelstyle)
        y_ax.setLabel('t, ms', **labelstyle)
        x_ax.setTickFont(font)
        y_ax.setTickFont(font)
        self.plotItem.ctrlMenu = None

        self.sgy = None
        self.picks_as_item = None
        self.processing_region_as_items = []
        self.traces_as_items = []

    def plotseis_sgy(self,
                     fname: Path,
                     normalize: bool = True,
                     clip: float = 0.9,
                     gain: float = 1,
                     negative_patch: bool = True,
                     refresh_view: bool = True):
        self.clear()
        self.sgy = SGY(fname)
        traces = self.sgy.read()

        if normalize:
            norm_factor = np.mean(np.abs(traces), axis=0)
            norm_factor[np.abs(norm_factor) < 1e-9 * np.max(np.abs(norm_factor))] = 1
            traces = traces / norm_factor

        traces = gain * traces
        mask_clip = np.abs(traces) > clip
        traces[mask_clip] = clip * np.sign(traces[mask_clip])

        self.plotseis(traces, negative_patch, refresh_view)

    def plotseis(self, traces: np.ndarray, negative_patch: bool = True, refresh_view: bool = True):
        self.remove_traces()
        num_sample, num_traces = np.shape(traces)
        t = np.arange(num_sample) * self.sgy.dt * 1e-3

        self.getViewBox().setLimits(xMin=0, xMax=num_traces + 1, yMin=0, yMax=t[-1])

        if refresh_view:
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
        self.traces_as_items.append(item)

    def remove_picks(self):
        if self.picks_as_item:
            self.removeItem(self.picks_as_item)

    def remove_processing_region(self):
        if self.processing_region_as_items:
            for item in self.processing_region_as_items:
                self.removeItem(item)

    def remove_traces(self):
        if self.traces_as_items:
            for item in self.traces_as_items:
                self.removeItem(item)

    def plot_processing_region(self,
                               traces_per_gather: int,
                               maximum_time: float,
                               contour_color: TColor = (100, 100, 100),
                               poly_color: TColor = (100, 100, 100, 50),
                               contour_width: float = 0.2
                               ):
        self.remove_processing_region()

        num_sample, num_traces = self.sgy.shape
        sgy_end_time = (num_sample + 2) * self.sgy.dt * 1e-3
        region_start_time = maximum_time if maximum_time > 0 else sgy_end_time

        contour_pen = QPen(QColor(*contour_color), contour_width, Qt.DashLine, Qt.FlatCap, Qt.MiterJoin)
        poly_brush = QColor(*poly_color)

        # Vertical lines
        line_t = np.array([0, region_start_time])
        for idx in np.arange(traces_per_gather + 0.5, num_traces - 1, traces_per_gather):
            line_x = np.array([idx, idx])
            line_path = pg.arrayToQPath(line_x, line_t, np.ones(2, dtype=np.int32))
            line_item = pg.QtWidgets.QGraphicsPathItem(line_path)
            line_item.setPen(contour_pen)
            self.processing_region_as_items.append(line_item)
            self.addItem(line_item)

        # Transparent polygon on bottom part
        poly_x = np.array([-2, num_traces + 2, num_traces + 2, -2])
        poly_t = np.array([region_start_time, region_start_time, sgy_end_time, sgy_end_time])
        poly_path = pg.arrayToQPath(poly_x, poly_t, np.ones(4, dtype=np.int32))
        poly_item = pg.QtWidgets.QGraphicsPathItem(poly_path)
        poly_item.setPen(contour_pen)
        poly_item.setBrush(poly_brush)
        self.processing_region_as_items.append(poly_item)
        self.addItem(poly_item)

    def plot_picks(self, picks: Sequence[float], color: TColor = (255, 0, 0)):
        self.remove_picks()

        num_traces = self.sgy.shape[1]
        picks = np.array(picks)
        ids = np.arange(num_traces) + 1

        path = pg.arrayToQPath(ids, picks, np.ones(num_traces, dtype=np.int32))
        self.picks_as_item = pg.QtWidgets.QGraphicsPathItem(path)

        pen = pg.mkPen(color=color, width=3)
        self.picks_as_item.setPen(pen)
        self.addItem(self.picks_as_item)


class HighMemoryConsumption(Exception):
    pass


class UnsupportedImageSize(Exception):
    pass


class GraphExporter(GraphWidget):
    @staticmethod
    def avoid_memory_bomb(height: int,
                          width_per_trace: int,
                          pixels_for_headers: int,
                          num_traces: int
                          ):
        height_image = height + pixels_for_headers
        width_image = pixels_for_headers + width_per_trace * num_traces

        max_height = max_width = 65000
        max_image_pixels = 2000 * 65000

        if height_image > max_height:
            message = f'It is not possible to render a picture of the given height = {height_image}. ' \
                      f'Max height = {max_height}. Decrease rendering parameters.'
            raise UnsupportedImageSize(message)

        if width_image > max_width:
            message = f'It is not possible to render a picture of the given width = {width_image}. ' \
                      f'Max width = {max_width}. Decrease rendering parameters.'
            raise UnsupportedImageSize(message)

        num_pixels = height_image * width_image

        if num_pixels > max_image_pixels:
            message = f'The size of the picture will turn out to be too large ({num_pixels} pixels) for ' \
                      f'this SGY file. Decrease rendering parameters.'
            raise HighMemoryConsumption(message)

    def lim_time(self, time_window: List[int]):
        self.getPlotItem().setYRange(time_window[0], time_window[1], padding=0)

    def export(self,
               export_filename: Path,
               sgy_filename: Path,
               height: int = 500,
               width_per_trace: int = 20,
               pixels_for_headers: int = 20,
               normalize: bool = True,
               clip: float = 0.9,
               amplification: float = 1,
               negative_patch: bool = True,
               task: Optional[Task] = None
               ):
        warnings.filterwarnings("ignore")

        self.sgy = SGY(sgy_filename)

        num_traces = self.sgy.shape[1]
        self.avoid_memory_bomb(height, width_per_trace, pixels_for_headers, num_traces)

        self.plotseis_sgy(sgy_filename,
                          normalize=normalize,
                          clip=clip,
                          gain=amplification,
                          negative_patch=negative_patch)

        self.plotItem.setFixedHeight(pixels_for_headers + height)
        self.plotItem.setFixedWidth(pixels_for_headers + width_per_trace * num_traces)

        exporter = ImageExporter(self.plotItem)
        Path(export_filename).parent.mkdir(exist_ok=True, parents=True)
        # exporter.export(str(export_filename))
        image = exporter.export(toBytes=True)
        image.save(str(export_filename), quality=100)
        self.sgy.close()

        warnings.resetwarnings()
        QTimer.singleShot(0, self.close_widget)

    @pyqtSlot()
    def close_widget(self):
        self.close()


def export_image_with_pyqt(export_filename: Path,
                           sgy_filename: Path,
                           height: int = 500,
                           width_per_trace: int = 20,
                           pixels_for_headers: int = 30,
                           normalize: bool = True,
                           clip: float = 0.9,
                           amplification: float = 1,
                           negative_patch: bool = True,
                           task: Optional[Task] = None):
    app = QApplication([])
    app.setQuitOnLastWindowClosed(True)
    window = GraphExporter(background='w')
    window.export(export_filename=export_filename,
                  sgy_filename=sgy_filename,
                  height=height,
                  width_per_trace=width_per_trace,
                  pixels_for_headers=pixels_for_headers,
                  normalize=normalize,
                  clip=clip,
                  amplification=amplification,
                  negative_patch=negative_patch,
                  task=task)
    app.exec()


if __name__ == '__main__':
    # task = BaseTask(Path('/home/daloro/small.sgy'), traces_per_shot=24, time_window=(0, 100), time_unit='ms')
    # task.result.picks_samples = np.random.randint(100, 200, 96)[:45]
    # # task.result.picks_samples[9: 15] = nn_config.not_presented_fb_value
    #
    # task.result.picks_samples = task.result.picks_samples.astype(int).tolist()
    #
    # task.result.processed_traces = list(range(96))[:45]
    export_image_with_pyqt(export_filename=Path(r'D:\Projects\first_breaks_picking\data\export.png'),
                           sgy_filename=Path(r'D:\Projects\first_breaks_picking\data\real_gather.sgy'),
                           height=600,
                           width_per_trace=25,
                           # task=task,
                           amplification=1,
                           clip=1)


