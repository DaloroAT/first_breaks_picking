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
from first_breaks.picking.utils import preprocess_gather
from first_breaks.sgy.reader import SGY


TColor = Union[Tuple[int, int, int, int], Tuple[int, int, int]]


class GraphDefaults:
    normalize: bool = True
    gain: float = 1.0
    clip: float = 0.9
    fill_black_left: bool = True
    picks_color: TColor = (255, 0, 0)
    region_contour_color: TColor = (100, 100, 100)
    region_contour_width: float = 0.2
    region_poly_color: TColor = (100, 100, 100, 50)



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

    def plotseis(self,
                 sgy: SGY,
                 clip: float = GraphDefaults.clip,
                 gain: float = GraphDefaults.gain,
                 normalize: bool = GraphDefaults.normalize,
                 fill_black_left: bool = GraphDefaults.fill_black_left,
                 refresh_view: bool = True):

        self.sgy = sgy
        traces = self.sgy.read()

        traces = preprocess_gather(traces, gain=gain, clip=clip, normalize=normalize, copy=True)

        self.remove_traces()
        num_sample, num_traces = self.sgy.shape
        t = np.arange(num_sample) * self.sgy.dt_ms

        self.getViewBox().setLimits(xMin=0, xMax=num_traces + 1, yMin=0, yMax=t[-1])

        if refresh_view:
            self.getPlotItem().setYRange(0, t[-1])
            self.getPlotItem().setXRange(0, num_traces + 1)

        for idx in range(num_traces):
            self._plot_trace_fast(traces[:, idx], t, idx + 1, fill_black_left)

    def _plot_trace_fast(self, trace: np.ndarray, t: np.ndarray, shift: int, fill_black_left: bool):
        connect = np.ones(len(t), dtype=np.int32)
        connect[-1] = 0

        trace[0] = 0
        trace[-1] = 0

        shifted_trace = trace + shift
        path = pg.arrayToQPath(shifted_trace, t, connect)

        item = pg.QtWidgets.QGraphicsPathItem(path)
        pen = QPen(Qt.black, 1, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin)
        pen.setWidthF(0.1)
        item.setPen(pen)
        item.setBrush(Qt.white)
        self.addItem(item)

        rect = QPainterPath()

        sign = -1 if fill_black_left else 1
        rect.addRect(shift, t[0], sign * 1.1 * max(np.abs(trace)), t[-1])

        patch = path.intersected(rect)
        item = pg.QtWidgets.QGraphicsPathItem(patch)

        pen = QPen(QColor(255, 255, 255, 0), 1, Qt.SolidLine,
                            Qt.FlatCap, Qt.MiterJoin)
        pen.setWidthF(0.1)
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
                               contour_color: TColor = GraphDefaults.region_contour_color,
                               poly_color: TColor = GraphDefaults.region_poly_color,
                               contour_width: float = GraphDefaults.region_contour_width
                               ):
        self.remove_processing_region()

        num_sample, num_traces = self.sgy.shape
        sgy_end_time = (num_sample + 2) * self.sgy.dt_ms
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

    def plot_picks(self, picks_ms: Sequence[float], color: TColor = GraphDefaults.picks_color):
        self.remove_picks()

        num_traces = self.sgy.shape[1]
        picks_ms = np.array(picks_ms)
        ids = np.arange(num_traces) + 1

        path = pg.arrayToQPath(ids, picks_ms, np.ones(num_traces, dtype=np.int32))
        self.picks_as_item = pg.QtWidgets.QGraphicsPathItem(path)

        pen = pg.mkPen(color=color, width=3)
        self.picks_as_item.setPen(pen)
        self.addItem(self.picks_as_item)


class HighMemoryConsumption(Exception):
    pass


class UnsupportedImageSize(Exception):
    pass


need_kwargs_exception = ValueError("Please, use named arguments instead of ordered for visualizations parameters. "
                                   "E.g. instead of `export(sgy, 'gather.png', 1.5)` use"
                                   " `export(sgy, 'gather.png', clip=1.5)`")


class GraphExporter(GraphWidget):
    MAX_SIDE_SIZE = 65000
    MAX_NUM_PIXELS = MAX_SIDE_SIZE * 2000

    @classmethod
    def avoid_memory_bomb(cls,
                          height: int,
                          width_per_trace: int,
                          pixels_for_headers: int,
                          num_traces: int
                          ):
        height_image = height + pixels_for_headers
        width_image = pixels_for_headers + width_per_trace * num_traces

        if height_image > cls.MAX_SIDE_SIZE:
            message = f'It is not possible to render a picture of the given height = {height_image}. ' \
                      f'Max height = {cls.MAX_SIDE_SIZE}. Decrease rendering parameters.'
            raise UnsupportedImageSize(message)

        if width_image > cls.MAX_SIDE_SIZE:
            message = f'It is not possible to render a picture of the given width = {width_image}. ' \
                      f'Max width = {cls.MAX_SIDE_SIZE}. Decrease rendering parameters.'
            raise UnsupportedImageSize(message)

        num_pixels = height_image * width_image

        if num_pixels > cls.MAX_NUM_PIXELS:
            message = f'The size of the picture will turn out to be too large ({num_pixels} pixels) for ' \
                      f'this SGY file. Decrease rendering parameters.'
            raise HighMemoryConsumption(message)

    def export(self,
               sgy: SGY,
               image_filename: Optional[Union[str, Path]],
               *args,
               clip: float = GraphDefaults.clip,
               gain: float = GraphDefaults.gain,
               normalize: bool = GraphDefaults.normalize,
               fill_black_left: bool = GraphDefaults.fill_black_left,
               time_window: Optional[Tuple[float, float]] = None,
               traces_window: Optional[Tuple[float, float]] = None,
               picks_ms: Optional[Sequence[float]] = None,
               task: Optional[Task] = None,
               show_processing_region: bool = True,
               picks_color: TColor = GraphDefaults.picks_color,
               contour_color: TColor = GraphDefaults.region_contour_color,
               poly_color: TColor = GraphDefaults.region_poly_color,
               contour_width: float = GraphDefaults.region_contour_width,
               height: int = 500,
               width_per_trace: int = 20,
               pixels_for_headers: int = 20,
               ):
        if args:
            raise need_kwargs_exception

        if picks_ms is not None and task is not None:
            raise ValueError("'picks_ms' and 'task' are mutually exclusive. Use only one of them or none")

        num_traces = sgy.num_traces
        self.avoid_memory_bomb(height, width_per_trace, pixels_for_headers, num_traces)

        self.clear()
        self.plotseis(sgy,
                      normalize=normalize,
                      clip=clip,
                      gain=gain,
                      fill_black_left=fill_black_left,
                      refresh_view=True)

        if task:
            picks_to_plot = task.picks_in_ms
        elif picks_ms:
            picks_to_plot = picks_ms
        else:
            picks_to_plot = None

        if picks_to_plot is not None:
            self.plot_picks(picks_to_plot, color=picks_color)

        if task is not None and show_processing_region:
            self.plot_processing_region(maximum_time=task.maximum_time_parsed,
                                        traces_per_gather=task.traces_per_gather_parsed,
                                        contour_color=contour_color,
                                        poly_color=poly_color,
                                        contour_width=contour_width)

        self.plotItem.setFixedHeight(pixels_for_headers + height)
        self.plotItem.setFixedWidth(pixels_for_headers + width_per_trace * num_traces)

        if time_window:
            self.getPlotItem().setYRange(time_window[0], time_window[1], padding=0)

        if traces_window:
            self.getPlotItem().setXRange(traces_window[0], traces_window[1], padding=0)

        image = ImageExporter(self.plotItem).export(toBytes=True)

        if image_filename:
            Path(image_filename).parent.mkdir(exist_ok=True, parents=True)
            image.save(str(image_filename), quality=100)

        QTimer.singleShot(0, self.close_widget)

    @pyqtSlot()
    def close_widget(self):
        self.close()


def export_image(sgy: SGY,
                 image_filename: Optional[Union[str, Path]],
                 *args,
                 clip: float = GraphDefaults.clip,
                 gain: float = GraphDefaults.gain,
                 normalize: bool = GraphDefaults.normalize,
                 fill_black_left: bool = GraphDefaults.fill_black_left,
                 time_window: Optional[Tuple[float, float]] = None,
                 traces_window: Optional[Tuple[float, float]] = None,
                 picks_ms: Optional[Sequence[float]] = None,
                 task: Optional[Task] = None,
                 show_processing_region: bool = True,
                 picks_color: TColor = GraphDefaults.picks_color,
                 contour_color: TColor = GraphDefaults.region_contour_color,
                 poly_color: TColor = GraphDefaults.region_poly_color,
                 contour_width: float = GraphDefaults.region_contour_width,
                 height: int = 500,
                 width_per_trace: int = 20,
                 pixels_for_headers: int = 20
                 ):
    if args:
        raise need_kwargs_exception

    app = QApplication([])
    app.setQuitOnLastWindowClosed(True)
    window = GraphExporter(background='w')
    window.export(sgy=sgy,
                  image_filename=image_filename,
                  clip=clip,
                  gain=gain,
                  normalize=normalize,
                  fill_black_left=fill_black_left,
                  time_window=time_window,
                  traces_window=traces_window,
                  picks_ms=picks_ms,
                  task=task,
                  show_processing_region=show_processing_region,
                  picks_color=picks_color,
                  contour_color=contour_color,
                  poly_color=poly_color,
                  contour_width=contour_width,
                  height=height,
                  width_per_trace=width_per_trace,
                  pixels_for_headers=pixels_for_headers
                  )
    app.exec()


if __name__ == '__main__':
    # task = BaseTask(Path('/home/daloro/small.sgy'), traces_per_shot=24, time_window=(0, 100), time_unit='ms')
    # task.result.picks_samples = np.random.randint(100, 200, 96)[:45]
    # # task.result.picks_samples[9: 15] = nn_config.not_presented_fb_value
    #
    # task.result.picks_samples = task.result.picks_samples.astype(int).tolist()
    #
    # task.result.processed_traces = list(range(96))[:45]
    import numpy as np
    import time

    # sgy = SGY(Path(r'D:\Projects\first_breaks_picking\data\real_gather.sgy'))
    sgy = SGY(np.random.uniform(-2, 2, (1000, 200)), dt_mcs=1e3)
    task = Task(sgy, maximum_time=100)
    task.picks_in_samples = np.random.uniform(0, 100, sgy.num_traces)

    st = time.perf_counter()
    export_image(image_filename=Path(r'D:\Projects\first_breaks_picking\data\export.jpg'),
                 sgy=sgy,
                 height=600,
                 width_per_trace=5,
                 pixels_for_headers=10,
                 task=task,
                 # time_window=(0, 100),
                 # traces_window=(10, 20),
                 show_processing_region=True,
                 gain=1,
                 clip=1)
    print(time.perf_counter() - st)


