from typing import Any, List, Tuple

import pyqtgraph as pg
from PyQt5.QtCore import QEvent, QObject, QPointF, Qt, pyqtSignal

from first_breaks.desktop.utils import get_mouse_position_in_scene_coords
from first_breaks.utils.utils import generate_color


def get_rect_of_roi(roi: pg.ROI) -> Tuple[float, float, float, float]:
    x0, y0 = roi.pos().x(), roi.pos().y()
    x1, y1 = roi.pos().x() + roi.size().x(), roi.pos().y() + roi.size().y()
    x_min, x_max = sorted([x0, x1])
    y_min, y_max = sorted([y0, y1])
    return x_min, y_min, x_max, y_max


def invert_roi_shape(roi: pg.ROI) -> pg.ROI:
    position, size = roi.pos(), roi.size()
    roi.setPos((position.y(), position.x()))
    roi.setSize((size.y(), size.x()))
    return roi


class RoiManager(QObject):
    roi_added_signal = pyqtSignal(object)
    roi_change_started_signal = pyqtSignal(object)
    roi_change_finished_signal = pyqtSignal(object)
    roi_changing_signal = pyqtSignal(object)
    roi_hover_signal = pyqtSignal(object)
    roi_clicked_signal = pyqtSignal(object)
    roi_deleted_signal = pyqtSignal(object)

    def __init__(self, viewbox: pg.ViewBox):
        super().__init__()
        self.viewbox = viewbox
        self.rois: List[pg.ROI] = []
        self.selected_roi = None
        self.creating_roi = False

        self.viewbox.installEventFilter(self)
        self.viewbox.setFocusPolicy(Qt.ClickFocus)  # need to receive keyboard events

        self.mouse_move_signal = pg.SignalProxy(self.viewbox.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

    def change_rois_view(self) -> None:
        for roi in self.rois:
            invert_roi_shape(roi)

    def mouse_moved(self, pos: Tuple[QPointF]) -> None:
        pos = pos[0]
        if self.creating_roi:
            self.update_roi_size(get_mouse_position_in_scene_coords(pos, self.viewbox))

    def eventFilter(self, obj: Any, event: QEvent) -> Any:
        # event_dict = {value: name for name, value in vars(QEvent).items() if isinstance(value, int)}
        # print(event.type(), event_dict.get(event.type(), "UNK"))
        if obj == self.viewbox:
            if event.type() == QEvent.GraphicsSceneMousePress:
                if event.button() == Qt.LeftButton and event.modifiers() == Qt.ShiftModifier:
                    self.creating_roi = True
                    self.start_roi_creation(get_mouse_position_in_scene_coords(event, self.viewbox))
                    return True
                self.select_roi()  # we want to accept other action by regular clicking, so we don't return True
            elif event.type() == QEvent.GraphicsSceneMouseRelease:
                if event.button() == Qt.LeftButton and self.creating_roi:
                    self.creating_roi = False
                    return True

            elif event.type() == QEvent.KeyPress:
                if event.key() == Qt.Key_Delete:
                    self.delete_selected_roi()
                    return True

        # If the event wasn't handled, propagate it further
        return super(RoiManager, self).eventFilter(obj, event)

    def start_roi_creation(self, position: QPointF) -> None:
        color = generate_color()
        pen = pg.mkPen(color=color, width=3)
        hover_pen = pg.mkPen(color=color, width=5)
        roi = pg.ROI([position.x(), position.y()], size=[0, 0], movable=True, pen=pen, hoverPen=hover_pen)

        roi.sigClicked.connect(self.on_roi_clicked)
        roi.sigRegionChangeStarted.connect(self.roi_change_started_signal)
        roi.sigRegionChanged.connect(self.roi_changing_signal)
        roi.sigRegionChangeFinished.connect(self.roi_change_finished_signal)
        roi.sigClicked.connect(self.roi_clicked_signal)
        self.viewbox.addItem(roi)
        self.rois.append(roi)
        self.roi_added_signal.emit(roi)
        self.selected_roi = roi
        self.creating_roi = True

    def update_roi_size(self, position: QPointF) -> None:
        if self.rois:
            self.rois[-1].setSize([position.x() - self.rois[-1].pos().x(), position.y() - self.rois[-1].pos().y()])

    def select_roi(self) -> None:
        self.selected_roi = None
        for roi in self.rois:
            if roi.isUnderMouse():
                self.selected_roi = roi
                break

    def delete_roi(self, roi: pg.ROI) -> None:
        roi.sigClicked.disconnect(self.on_roi_clicked)
        roi.sigRegionChangeStarted.disconnect(self.roi_change_started_signal)
        roi.sigRegionChanged.disconnect(self.roi_changing_signal)
        roi.sigRegionChangeFinished.disconnect(self.roi_change_finished_signal)
        roi.sigClicked.disconnect(self.roi_clicked_signal)

        self.roi_deleted_signal.emit(roi)

        self.viewbox.removeItem(roi)
        self.rois.remove(roi)

    def delete_all_rois(self) -> None:
        for roi in list(self.rois):
            self.delete_roi(roi)

    def delete_selected_roi(self) -> None:
        if self.selected_roi:
            self.delete_roi(self.selected_roi)
            self.selected_roi = None

    def on_roi_clicked(self, roi: pg.ROI) -> None:
        self.selected_roi = roi
