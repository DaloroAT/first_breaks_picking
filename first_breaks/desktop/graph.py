import ast
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QCloseEvent, QColor, QFont, QPainterPath, QPen
from PyQt5.QtWidgets import QApplication
from pyqtgraph import AxisItem
from pyqtgraph.exporters import ImageExporter
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent

from first_breaks.const import HIGH_DPI
from first_breaks.data_models.independent import TColor, TNormalize
from first_breaks.data_models.initialised_defaults import DEFAULTS
from first_breaks.desktop.roi_manager import RoiManager
from first_breaks.desktop.spectrum_window import SpectrumWindow
from first_breaks.picking.picks import Picks
from first_breaks.picking.task import Task
from first_breaks.picking.utils import preprocess_gather
from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import resolve_postime2xy as postime2xy
from first_breaks.utils.utils import resolve_xy2postime as xy2postime

if HIGH_DPI:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


class GraphWidget(pg.PlotWidget):
    picks_manual_edited_signal = pyqtSignal(Picks)
    about_to_change_nn_picks_signal = pyqtSignal()

    def __init__(self, use_open_gl: bool = True, *args: Any, **kwargs: Any):
        super().__init__(useOpenGL=use_open_gl, *args, **kwargs)
        self.plotItem.disableAutoRange()
        self.setAntialiasing(False)
        self.plotItem.setClipToView(True)
        self.plotItem.setDownsampling(mode="peak")
        self.plotItem.ctrlMenu = None

        self.invert_x = DEFAULTS.invert_x
        self.invert_y = DEFAULTS.invert_y
        self.vsp_view = DEFAULTS.vsp_view
        self.sgy: Optional[SGY] = None
        # self.picks_as_items: List[pg.PlotCurveItem] = []
        self.picks2items: Dict[Picks, pg.PlotCurveItem] = {}
        self.processing_region_as_items: List[pg.QtWidgets.QGraphicsPathItem] = []
        self.traces_as_items: List[pg.QtWidgets.QGraphicsPathItem] = []
        self.pos_ax_header: Optional[str] = None

        self.x_ax: Optional[AxisItem] = None
        self.y_ax: Optional[AxisItem] = None
        self.pos_ax: Optional[AxisItem] = None
        self.time_ax: Optional[AxisItem] = None
        self.setup_axes()

        self.spectrum_roi_manager = RoiManager(viewbox=self.getViewBox())
        self.spectrum_window = SpectrumWindow(use_open_gl=use_open_gl, roi_manager=self.spectrum_roi_manager)
        self.mouse_click_signal = pg.SignalProxy(self.sceneObj.sigMouseClicked, rateLimit=60, slot=self.mouse_clicked)

    def resolve_postime2xy(self, position: Any, time: Any) -> Tuple[Any, Any]:
        return postime2xy(vsp_view=self.vsp_view, position=position, time=time)

    def resolve_xy2postime(self, x: Any, y: Any) -> Tuple[Any, Any]:
        return xy2postime(vsp_view=self.vsp_view, x=x, y=y)

    def setup_axes(self) -> None:
        self.getPlotItem().invertX(self.invert_x)
        self.getPlotItem().invertY(self.invert_y)
        self.getPlotItem().showAxis("top", True)
        self.getPlotItem().showAxis("bottom", False)

        self.x_ax = self.getPlotItem().getAxis("top")
        self.y_ax = self.getPlotItem().getAxis("left")

        self.pos_ax, self.time_ax = self.resolve_xy2postime(self.x_ax, self.y_ax)

        self.pos_ax.tickStrings = self.replace_tick_labels

        text_size = 12
        self.label_style = {"font-size": f"{text_size}pt"}
        font = QFont()
        font.setPointSize(text_size)
        self.pos_ax.setLabel(None, **self.label_style)
        self.time_ax.setLabel("t, ms", **self.label_style)
        self.pos_ax.setTickFont(font)
        self.time_ax.setTickFont(font)

    def full_clean(self) -> None:
        self.remove_picks()
        self.remove_traces()
        self.remove_processing_region()
        self.spectrum_roi_manager.delete_all_rois()
        self.clear()

    def plotseis(
        self,
        sgy: SGY,
        clip: float = DEFAULTS.clip,
        gain: float = DEFAULTS.gain,
        normalize: TNormalize = DEFAULTS.normalize,
        f1_f2: Optional[Tuple[float, float]] = None,
        f3_f4: Optional[Tuple[float, float]] = None,
        x_axis: Optional[str] = DEFAULTS.x_axis,
        fill_black: Optional[str] = DEFAULTS.fill_black,
        vsp_view: bool = DEFAULTS.vsp_view,
        refresh_view: bool = True,
        invert_x: bool = DEFAULTS.invert_x,
        invert_y: bool = DEFAULTS.invert_y,
    ) -> None:
        self.pos_ax_header = x_axis

        self.sgy = sgy
        self.spectrum_window.set_sgy(sgy)
        self.spectrum_window.set_filter_params(f1_f2, f3_f4)

        traces = self.sgy.read()

        traces = preprocess_gather(
            traces, gain=gain, clip=clip, normalize=normalize, f1_f2=f1_f2, f3_f4=f3_f4, fs=self.sgy.fs, copy=True
        )

        # we put clearing after preprocessing to reduce time when user see nothing
        self.clear()
        # axes related checks
        if self.vsp_view != vsp_view:
            self.vsp_view = vsp_view
            self.spectrum_roi_manager.change_rois_view()
            self.spectrum_window.set_vsp_view(self.vsp_view)
            need_to_refresh = True
        else:
            need_to_refresh = False
        if self.invert_x != invert_x:
            self.invert_x = invert_x
            need_to_refresh |= True
        else:
            need_to_refresh |= False
        if self.invert_y != invert_y:
            self.invert_y = invert_y
            need_to_refresh |= True
        else:
            need_to_refresh |= False

        if need_to_refresh:
            self.setup_axes()

        num_sample, num_traces = self.sgy.shape
        t = np.arange(num_sample) * self.sgy.dt_ms

        pos_max = num_traces + 1
        time_max = t[-1]
        x_max, y_max = self.resolve_postime2xy(pos_max, time_max)

        self.getViewBox().setLimits(xMin=0, xMax=x_max, yMin=0, yMax=y_max)

        if refresh_view or need_to_refresh:
            self.getPlotItem().setYRange(min=0, max=y_max)
            self.getPlotItem().setXRange(min=0, max=x_max)

        for idx in range(num_traces):
            self._plot_trace_fast(trace=traces[:, idx], time=t, shift=idx + 1, fill_black=fill_black)

        self.pos_ax.showLabel()

    def _plot_trace_fast(self, trace: np.ndarray, time: np.ndarray, shift: int, fill_black: Optional[str]) -> None:
        connect = np.ones(len(time), dtype=np.int32)
        connect[-1] = 0

        if fill_black == "right":
            trace = np.sign(trace) * trace**3
            trace = np.gradient(np.gradient(trace)) * 10

        trace[0] = 0
        trace[-1] = 0

        shifted_trace = trace + shift

        path_x, path_y = self.resolve_postime2xy(shifted_trace, time)
        path = pg.arrayToQPath(x=path_x, y=path_y, connect=connect)

        item = pg.QtWidgets.QGraphicsPathItem(path)
        pen = QPen(Qt.black, 1, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin)
        pen.setWidth(0.1)
        item.setPen(pen)
        item.setBrush(Qt.white)
        self.addItem(item)
        self.traces_as_items.append(item)

        if fill_black is None:
            return
        else:
            sign = -1 if fill_black == "left" else 1

            x, y = self.resolve_postime2xy(shift, time[0])
            w, h = self.resolve_postime2xy(sign * 1.1 * max(np.abs(shifted_trace)), time[-1])

            rect = QPainterPath()
            rect.addRect(x, y, w, h)

            patch = path.intersected(rect)
            item = pg.QtWidgets.QGraphicsPathItem(patch)

            pen = QPen(QColor(255, 255, 255, 0), 1, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin)
            pen.setWidth(0.1)
            item.setPen(pen)
            item.setBrush(Qt.black)
            self.addItem(item)
            self.traces_as_items.append(item)

    def replace_tick_labels(self, *args: Any, **kwargs: Any) -> List[str]:
        self.pos_ax.setLabel(self.pos_ax_header, **self.label_style)
        previous_labels = AxisItem.tickStrings(self.pos_ax, *args, **kwargs)

        if self.pos_ax_header is not None:
            labels_from_headers = []
            for v in previous_labels:
                v = ast.literal_eval(v)
                if v % 1 == 0:
                    v = int(v) - 1
                    if 0 <= v < self.sgy.num_traces:
                        labels_from_headers.append(str(self.sgy.traces_headers[self.pos_ax_header].iloc[v]))
                    else:
                        labels_from_headers.append("")
                else:
                    labels_from_headers.append("")
            return labels_from_headers
        else:
            return previous_labels

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
        region_contour_color: TColor = DEFAULTS.region_contour_color,
        region_poly_color: TColor = DEFAULTS.region_poly_color,
        region_contour_width: float = DEFAULTS.region_contour_width,
    ) -> None:
        self.remove_processing_region()

        num_sample, num_traces = self.sgy.shape
        sgy_end_time = (num_sample + 2) * self.sgy.dt_ms
        region_start_time = maximum_time if maximum_time > 0 else sgy_end_time

        contour_pen = QPen(QColor(*region_contour_color), region_contour_width, Qt.DashLine, Qt.FlatCap, Qt.MiterJoin)
        poly_brush = QColor(*region_poly_color)

        # Vertical lines
        line_time = np.array([0, region_start_time])
        for idx in np.arange(traces_per_gather + 0.5, num_traces - 1, traces_per_gather):
            line_pos = np.array([idx, idx])
            line_x, line_y = self.resolve_postime2xy(line_pos, line_time)
            line_path = pg.arrayToQPath(line_x, line_y, np.ones(2, dtype=np.int32))
            line_item = pg.QtWidgets.QGraphicsPathItem(line_path)
            line_item.setPen(contour_pen)
            self.processing_region_as_items.append(line_item)
            self.addItem(line_item)

        # Transparent polygon on ignored part
        poly_pos = np.array([-2, num_traces + 2, num_traces + 2, -2])
        poly_time = np.array([region_start_time, region_start_time, sgy_end_time, sgy_end_time])
        poly_x, poly_y = self.resolve_postime2xy(poly_pos, poly_time)
        poly_path = pg.arrayToQPath(poly_x, poly_y, np.ones(4, dtype=np.int32))
        poly_item = pg.QtWidgets.QGraphicsPathItem(poly_path)
        poly_item.setPen(contour_pen)
        poly_item.setBrush(poly_brush)
        self.processing_region_as_items.append(poly_item)
        self.addItem(poly_item)

    def get_picks_as_item(self, picks: Picks) -> pg.PlotCurveItem:
        x, y = self.resolve_postime2xy(np.arange(self.sgy.num_traces) + 1, np.array(picks.picks_in_ms))

        line = pg.PlotCurveItem()
        line.setData(x, y)
        pen = pg.mkPen(color=picks.color, width=picks.width)
        line.setPen(pen)

        return line

    def plot_picks(self, picks: Picks) -> None:
        picks_item = self.get_picks_as_item(picks)
        self.addItem(picks_item)
        self.picks2items[picks] = picks_item

    def remove_picks(self) -> None:
        for picks in list(self.picks2items.keys()):
            self.removeItem(self.picks2items[picks])
            del self.picks2items[picks]

    def mouse_clicked(self, ev: Tuple[MouseClickEvent]) -> None:
        ev = ev[0]
        active_picks_list = [k for k in self.picks2items.keys() if k.active]

        if active_picks_list:
            active_picks = active_picks_list[0]
        else:
            return

        if active_picks.created_by_nn and not active_picks.modified_manually:
            self.about_to_change_nn_picks_signal.emit()
            self.mouse_clicked((ev,))
            return

        if active_picks and ev.button() == 1:
            mouse_xy = self.getPlotItem().vb.mapSceneToView(ev.scenePos())
            mouse_pos, mouse_time = self.resolve_xy2postime(mouse_xy.x(), mouse_xy.y())

            picks_x, picks_y = self.picks2items[active_picks].getData()
            picks_pos, picks_time = self.resolve_xy2postime(picks_x, picks_y)

            closest = np.argmin(np.abs(picks_pos - mouse_pos))
            mouse_time = np.clip(mouse_time, 0, self.sgy.max_time_ms)
            picks_time[closest] = mouse_time

            picks_x, picks_y = self.resolve_postime2xy(picks_pos, picks_time)

            self.picks2items[active_picks].setData(picks_x, picks_y)
            active_picks.from_ms(picks_time)

            self.picks_manual_edited_signal.emit(active_picks)

    def closeEvent(self, e: QCloseEvent) -> None:
        self.spectrum_window.close()
        e.accept()


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
        clip: float = DEFAULTS.clip,
        gain: float = DEFAULTS.gain,
        normalize: TNormalize = DEFAULTS.normalize,
        fill_black: Optional[str] = DEFAULTS.fill_black,
        time_window: Optional[Tuple[float, float]] = None,
        traces_window: Optional[Tuple[float, float]] = None,
        picks: Optional[Picks] = None,
        task: Optional[Task] = None,
        show_processing_region: bool = True,
        contour_color: TColor = DEFAULTS.region_contour_color,
        poly_color: TColor = DEFAULTS.region_poly_color,
        contour_width: float = DEFAULTS.region_contour_width,
        x_axis: Optional[str] = DEFAULTS.x_axis,
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

        if picks is not None and task is not None:
            raise ValueError("'picks' and 'task' are mutually exclusive. Use only one of them or none")

        if width is None:
            if traces_window is None:
                num_traces = sgy.num_traces
                width = int(width_per_trace * num_traces) + headers_total_pixels
            else:
                width = int(width_per_trace * (traces_window[1] - traces_window[0])) + headers_total_pixels

        self.avoid_memory_bomb(height, width)

        self.x_ax_header = x_axis

        self.clear()
        self.plotseis(sgy, normalize=normalize, clip=clip, gain=gain, fill_black=fill_black, refresh_view=True)

        if task:
            self.plot_picks(task.picks)
        elif picks is not None:
            self.plot_picks(picks)

        if task is not None and show_processing_region:
            self.plot_processing_region(
                maximum_time=task.maximum_time,
                traces_per_gather=task.traces_per_gather,
                region_contour_color=contour_color,
                region_poly_color=poly_color,
                region_contour_width=contour_width,
            )

        if time_window:
            self.getPlotItem().setYRange(time_window[0], time_window[1], padding=0)
        if traces_window:
            self.getPlotItem().setXRange(traces_window[0], traces_window[1], padding=0)

        headers_font_pixels = headers_font_pixels or int(0.35 * headers_total_pixels)
        labelstyle = {"font-size": f"{headers_font_pixels}px"}
        tickfont = QFont()
        tickfont.setPixelSize(max(int(0.9 * headers_font_pixels), 1))
        self.x_ax.setLabel(self.x_ax_header, **labelstyle)
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
    clip: float = DEFAULTS.clip,
    gain: float = DEFAULTS.gain,
    normalize: TNormalize = DEFAULTS.normalize,
    fill_black: Optional[str] = DEFAULTS.fill_black,
    time_window: Optional[Tuple[float, float]] = None,
    traces_window: Optional[Tuple[float, float]] = None,
    picks: Optional[Picks] = None,
    show_processing_region: bool = True,
    contour_color: TColor = DEFAULTS.region_contour_color,
    poly_color: TColor = DEFAULTS.region_poly_color,
    contour_width: float = DEFAULTS.region_contour_width,
    x_axis: Optional[str] = DEFAULTS.x_axis,
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
        fill_black=fill_black,
        time_window=time_window,
        traces_window=traces_window,
        picks=picks,
        task=task,
        show_processing_region=show_processing_region,
        contour_color=contour_color,
        poly_color=poly_color,
        contour_width=contour_width,
        x_axis=x_axis,
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
    from first_breaks.sgy.reader import SGY
    from first_breaks.utils.utils import download_demo_sgy

    demo_sgy = download_demo_sgy()
    export_image(demo_sgy, "demo_sgy.png")
    # app = QApplication([])
    # window = GraphWidget(background="w")
    # window.show()
    # window.plotseis(SGY(demo_sgy))
    # app.exec_()
