import os
import warnings
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QPainterPath, QPen
from PyQt5.QtWidgets import QApplication
from pyqtgraph.exporters import ImageExporter
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent

from first_breaks.const import HIGH_DPI
from first_breaks.picking.task import Task
from first_breaks.picking.utils import preprocess_gather
from first_breaks.sgy.reader import SGY

if HIGH_DPI:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

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
    def __init__(self, use_open_gl: bool = True, *args: Any, **kwargs: Any):
        super().__init__(useOpenGL=use_open_gl, *args, **kwargs)
        self.getPlotItem().disableAutoRange()
        self.setAntialiasing(False)
        self.getPlotItem().setClipToView(True)
        self.getPlotItem().setDownsampling(mode="peak")

        self.getPlotItem().invertY(True)
        self.getPlotItem().showAxis("top", True)
        self.getPlotItem().showAxis("bottom", False)
        self.x_ax = self.getPlotItem().getAxis("top")
        self.y_ax = self.getPlotItem().getAxis("left")
        text_size = 12
        labelstyle = {"font-size": f"{text_size}pt"}
        font = QFont()
        font.setPointSize(text_size)
        self.x_ax.setLabel("trace", **labelstyle)
        self.y_ax.setLabel("t, ms", **labelstyle)
        self.x_ax.setTickFont(font)
        self.y_ax.setTickFont(font)
        self.plotItem.ctrlMenu = None

        self.sgy: Optional[SGY] = None
        self.picks_in_ms: Optional[np.ndarray] = None
        self.picks_as_item: Optional[pg.PlotCurveItem] = None
        self.processing_region_as_items: List[pg.QtWidgets.QGraphicsPathItem] = []
        self.traces_as_items: List[pg.QtWidgets.QGraphicsPathItem] = []
        self.is_picks_modified_manually = False

        self.mouse_click_signal = pg.SignalProxy(self.sceneObj.sigMouseClicked, rateLimit=60, slot=self.mouse_clicked)

    def mouse_clicked(self, ev: Tuple[MouseClickEvent]) -> None:
        ev = ev[0]
        if self.picks_as_item is not None and ev.button() == 1:
            mouse_point = self.getPlotItem().vb.mapSceneToView(ev.scenePos())
            x, y = mouse_point.x(), mouse_point.y()
            ids, picks = self.picks_as_item.getData()
            closest = np.argmin(np.abs(ids - x))
            picks[closest] = y
            self.picks_as_item.setData(ids, picks)
            self.picks_in_ms = picks
            self.is_picks_modified_manually = True

    def full_clean(self) -> None:
        self.remove_picks()
        self.remove_traces()
        self.remove_processing_region()
        self.picks_as_item = None
        self.picks_in_ms = None
        self.is_picks_modified_manually = False
        self.clear()

    def plotseis(
        self,
        sgy: SGY,
        clip: float = GraphDefaults.clip,
        gain: float = GraphDefaults.gain,
        normalize: bool = GraphDefaults.normalize,
        fill_black_left: bool = GraphDefaults.fill_black_left,
        refresh_view: bool = True,
    ) -> None:
        self.sgy = sgy

        traces = self.sgy.read()
        traces = preprocess_gather(traces, gain=gain, clip=clip, normalize=normalize, copy=True)

        self.clear()

        num_sample, num_traces = self.sgy.shape
        t = np.arange(num_sample) * self.sgy.dt_ms

        self.getViewBox().setLimits(xMin=0, xMax=num_traces + 1, yMin=0, yMax=t[-1])

        if refresh_view:
            self.getPlotItem().setYRange(0, t[-1])
            self.getPlotItem().setXRange(0, num_traces + 1)

        for idx in range(num_traces):
            self._plot_trace_fast(traces[:, idx], t, idx + 1, fill_black_left)

    def _plot_trace_fast(self, trace: np.ndarray, t: np.ndarray, shift: int, fill_black_left: bool) -> None:
        connect = np.ones(len(t), dtype=np.int32)
        connect[-1] = 0

        trace[0] = 0
        trace[-1] = 0

        shifted_trace = trace + shift
        path = pg.arrayToQPath(shifted_trace, t, connect)

        item = pg.QtWidgets.QGraphicsPathItem(path)
        pen = QPen(Qt.black, 1, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin)
        # pen.setWidthF(0.01)
        pen.setWidth(0.1)
        item.setPen(pen)
        item.setBrush(Qt.white)
        self.addItem(item)

        rect = QPainterPath()

        sign = -1 if fill_black_left else 1
        rect.addRect(shift, t[0], sign * 1.1 * max(np.abs(trace)), t[-1])

        patch = path.intersected(rect)
        item = pg.QtWidgets.QGraphicsPathItem(patch)

        pen = QPen(QColor(255, 255, 255, 0), 1, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin)
        # pen.setWidthF(0.01)
        pen.setWidth(0.1)
        item.setPen(pen)
        item.setBrush(Qt.black)
        self.addItem(item)
        self.traces_as_items.append(item)

    def remove_picks(self) -> None:
        self.is_picks_modified_manually = False
        if self.picks_as_item:
            self.removeItem(self.picks_as_item)
            self.picks_as_item = None
            self.picks_in_ms = None

    def remove_processing_region(self) -> None:
        if self.processing_region_as_items:
            for item in self.processing_region_as_items:
                self.removeItem(item)

    def remove_traces(self) -> None:
        if self.traces_as_items:
            for item in self.traces_as_items:
                self.removeItem(item)

    def plot_processing_region(
        self,
        traces_per_gather: int,
        maximum_time: float,
        contour_color: TColor = GraphDefaults.region_contour_color,
        poly_color: TColor = GraphDefaults.region_poly_color,
        contour_width: float = GraphDefaults.region_contour_width,
    ) -> None:
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

    def plot_picks(self, picks_ms: Sequence[float], color: TColor = GraphDefaults.picks_color) -> None:
        self.remove_picks()
        self.is_picks_modified_manually = False

        picks_ms = np.array(picks_ms)
        ids = np.arange(self.sgy.num_traces) + 1

        self.picks_in_ms = picks_ms
        self.picks_as_item = pg.PlotCurveItem()
        self.picks_as_item.setData(ids, picks_ms)

        pen = pg.mkPen(color=color, width=3)
        self.picks_as_item.setPen(pen)
        self.addItem(self.picks_as_item)


class HighMemoryConsumption(Exception):
    pass


class UnsupportedImageSize(Exception):
    pass


need_kwargs_exception = ValueError(
    "Please, use named arguments instead of ordered for visualizations parameters. "
    "E.g. instead of `export(sgy, 'gather.png', 1.5)` use"
    " `export(sgy, 'gather.png', clip=1.5)`"
)


class GraphExporter(GraphWidget):
    MAX_SIDE_SIZE = 65000
    MAX_NUM_PIXELS = MAX_SIDE_SIZE * 2000

    def __init__(self, *args: Any, **kwargs: Any):
        os.environ["DEBIAN_FRONTEND"] = "noninteractive"
        os.environ["LIBGL_ALWAYS_INDIRECT"] = "1"
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        super().__init__(*args, **kwargs)
        self.setAntialiasing(True)

    @classmethod
    def avoid_memory_bomb(
        cls,
        height_image: int,
        width_image: int,
    ) -> None:

        if height_image > cls.MAX_SIDE_SIZE:
            message = (
                f"It is not possible to render a picture of the given height = {height_image}. "
                f"Max height = {cls.MAX_SIDE_SIZE}. Decrease rendering parameters."
            )
            raise UnsupportedImageSize(message)

        if width_image > cls.MAX_SIDE_SIZE:
            message = (
                f"It is not possible to render a picture of the given width = {width_image}. "
                f"Max width = {cls.MAX_SIDE_SIZE}. Decrease rendering parameters."
            )
            raise UnsupportedImageSize(message)

        num_pixels = height_image * width_image

        if num_pixels > cls.MAX_NUM_PIXELS:
            message = (
                f"The size of the picture will turn out to be too large ({num_pixels} pixels) for "
                f"this SGY file. Decrease rendering parameters."
            )
            raise HighMemoryConsumption(message)

    def export(
        self,
        sgy: SGY,
        image_filename: Optional[Union[str, Path]],
        *args: Any,
        # content parameters
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
        # rendering parameters
        height: int = 500,
        width: Optional[int] = None,
        width_per_trace: int = 20,
        headers_total_pixels: int = 50,
        headers_font_pixels: Optional[int] = None,
        time_spacing: Optional[int] = None,
        traces_spacing: Optional[int] = None,
        hide_traces_axis: bool = False,
    ) -> None:
        if args:
            raise need_kwargs_exception

        if picks_ms is not None and task is not None:
            raise ValueError("'picks_ms' and 'task' are mutually exclusive. Use only one of them or none")

        if width is None:
            if traces_window is None:
                num_traces = sgy.num_traces
                width = int(width_per_trace * num_traces) + headers_total_pixels
            else:
                width = int(width_per_trace * (traces_window[1] - traces_window[0])) + headers_total_pixels

        self.avoid_memory_bomb(height, width)

        self.clear()
        self.plotseis(
            sgy, normalize=normalize, clip=clip, gain=gain, fill_black_left=fill_black_left, refresh_view=True
        )

        if task:
            picks_to_plot = task.picks_in_ms  # type: ignore
        elif picks_ms is not None:
            picks_to_plot = picks_ms  # type: ignore
        else:
            picks_to_plot = None  # type: ignore

        if picks_to_plot is not None:
            self.plot_picks(picks_to_plot, color=picks_color)

        if task is not None and show_processing_region:
            self.plot_processing_region(
                maximum_time=task.maximum_time_parsed,
                traces_per_gather=task.traces_per_gather_parsed,
                contour_color=contour_color,
                poly_color=poly_color,
                contour_width=contour_width,
            )

        if time_window:
            self.getPlotItem().setYRange(time_window[0], time_window[1], padding=0)
        if traces_window:
            self.getPlotItem().setXRange(traces_window[0], traces_window[1], padding=0)

        headers_font_pixels = headers_font_pixels or int(0.35 * headers_total_pixels)
        labelstyle = {"font-size": f"{headers_font_pixels}px"}
        tickfont = QFont()
        tickfont.setPixelSize(max(int(0.9 * headers_font_pixels), 1))
        self.x_ax.setLabel("trace", **labelstyle)
        self.y_ax.setLabel("t, ms", **labelstyle)
        self.x_ax.setTickFont(tickfont)
        self.y_ax.setTickFont(tickfont)

        self.x_ax.setHeight(headers_total_pixels)

        if time_spacing:
            self.y_ax.setTickSpacing(time_spacing, time_spacing)
        if traces_spacing:
            self.x_ax.setTickSpacing(traces_spacing, time_spacing)

        self.y_ax.setWidth(headers_total_pixels)
        if hide_traces_axis:
            self.x_ax.showLabel(False)
            self.x_ax.setTicks([])
            self.x_ax.setHeight(0)
        else:
            self.x_ax.setHeight(headers_total_pixels)

        self.plotItem.setFixedHeight(height)
        self.plotItem.setFixedWidth(width)

        image = ImageExporter(self.plotItem).export(toBytes=True)

        if image_filename:
            Path(image_filename).parent.mkdir(exist_ok=True, parents=True)
            image.save(str(image_filename), quality=100)

        QTimer.singleShot(0, self.close_widget)

    @pyqtSlot()
    def close_widget(self) -> None:
        self.close()


def export_image(
    source: Union[str, Path, bytes, np.ndarray, SGY, Task],
    image_filename: Optional[Union[str, Path]],
    *args: Any,
    dt_mcs: float = 1e3,
    clip: float = GraphDefaults.clip,
    gain: float = GraphDefaults.gain,
    normalize: bool = GraphDefaults.normalize,
    fill_black_left: bool = GraphDefaults.fill_black_left,
    time_window: Optional[Tuple[float, float]] = None,
    traces_window: Optional[Tuple[float, float]] = None,
    picks_ms: Optional[Sequence[float]] = None,
    show_processing_region: bool = True,
    picks_color: TColor = GraphDefaults.picks_color,
    contour_color: TColor = GraphDefaults.region_contour_color,
    poly_color: TColor = GraphDefaults.region_poly_color,
    contour_width: float = GraphDefaults.region_contour_width,
    # rendering parameters
    height: int = 500,
    width: Optional[int] = None,
    width_per_trace: int = 20,
    headers_total_pixels: int = 50,
    headers_font_pixels: Optional[int] = None,
    time_spacing: Optional[int] = None,
    traces_spacing: Optional[int] = None,
    hide_traces_axis: bool = False,
) -> None:
    if args:
        raise need_kwargs_exception

    if isinstance(source, (str, Path, bytes)):
        sgy = SGY(source)
        task = None
    elif isinstance(source, np.ndarray):
        sgy = SGY(source, dt_mcs=dt_mcs)
        task = None
    elif isinstance(source, SGY):
        sgy = source
        task = None
    elif isinstance(source, Task):
        sgy = source.sgy
        task = source
    else:
        raise TypeError("Unsupported type for 'source'")

    warnings.filterwarnings("ignore")
    app = QApplication([])
    app.setQuitOnLastWindowClosed(True)
    window = GraphExporter(background="w")
    window.hide()
    window.setAntialiasing(True)
    window.export(
        sgy=sgy,
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
        width=width,
        width_per_trace=width_per_trace,
        headers_total_pixels=headers_total_pixels,
        headers_font_pixels=headers_font_pixels,
        time_spacing=time_spacing,
        traces_spacing=traces_spacing,
        hide_traces_axis=hide_traces_axis,
    )
    app.exec()
    warnings.resetwarnings()


if __name__ == "__main__":
    from first_breaks.utils.utils import download_demo_sgy

    demo_sgy = download_demo_sgy()
    export_image(demo_sgy, "demo_sgy.png")
