import sys
import uuid
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QByteArray, QEvent, QMimeData, Qt, pyqtSignal
from PyQt5.QtGui import QDrag, QMouseEvent
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QPushButton,
    QWidget,
)

from first_breaks.desktop.utils import QVSeparationLine

DRAG_ID_KEY = "application/x-draggable-id"


def set_normal_style(widget: QWidget) -> None:
    widget.setStyleSheet(
        """
                background-color: rgba(211, 211, 211, 255);
                border: 1px solid gray;
                color: rgb(0, 0, 0);
                padding: 5px;
                border-radius: 5px;
            """
    )


def set_transparent_style(widget: QWidget) -> None:
    widget.setStyleSheet(
        """
                background-color: rgba(211, 211, 211, 0);
                border: rgba(211, 211, 211, 0);
                color: rgba(0, 0, 0, 0);
            """
    )


class Tag(QLineEdit):
    def __init__(self, text: str, parent: Optional[QWidget] = None, fixed_height_policy: bool = False) -> None:
        super(Tag, self).__init__(text, parent)
        self.id = uuid.uuid4().bytes
        self.setMinimumWidth(10)
        h_policy = QtWidgets.QSizePolicy.Fixed if fixed_height_policy else QtWidgets.QSizePolicy.Expanding
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, h_policy)
        self.setReadOnly(True)
        self.setToolTip(text)
        self.setAlignment(Qt.AlignCenter)

        self.drag_start_position = 0

        self.set_normal_style()

    def set_normal_style(self) -> None:
        set_normal_style(self)

    def set_transparent_style(self) -> None:
        set_transparent_style(self)

    def dragLeaveEvent(self, event: QMouseEvent) -> None:
        self.set_normal_style()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not (event.buttons() & Qt.LeftButton):
            return
        if (event.pos() - self.drag_start_position).manhattanLength() < QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setData(DRAG_ID_KEY, QByteArray(self.id))
        drag.setMimeData(mime_data)
        pixmap = self.grab()
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.pos())
        drag.exec_(Qt.MoveAction)


class BaseMultiSelectWidget(QWidget):
    list_changed_signal = pyqtSignal(list)

    def __init__(self, fixed_height_policy: bool = False) -> None:
        super(BaseMultiSelectWidget, self).__init__()
        self._main_layout = QHBoxLayout(self)
        self.setLayout(self._main_layout)

        self.layout = QHBoxLayout(self)
        self._main_layout.addLayout(self.layout)

        self.setAcceptDrops(True)
        self.id2tag: Dict[bytes, Tag] = {}
        self.ids: List[bytes] = []
        self.currently_dragged_id: Optional[bytes] = None
        self.fixed_height_policy = fixed_height_policy

        self.vline = QVSeparationLine()
        self.vline.setVisible(False)
        self._main_layout.addWidget(self.vline)

        self.button_add_remove = QPushButton("")
        self.set_button_text_add_tag()
        h_policy = QtWidgets.QSizePolicy.Fixed if fixed_height_policy else QtWidgets.QSizePolicy.Expanding
        self.button_add_remove.setSizePolicy(QtWidgets.QSizePolicy.Expanding, h_policy)
        self.button_add_remove.setFixedWidth(60)
        set_normal_style(self.button_add_remove)
        self.button_add_remove.clicked.connect(self.add_tag_prompt)
        self._main_layout.addWidget(self.button_add_remove)

    def set_button_text_add_tag(self) -> None:
        self.button_add_remove.setText("Add")

    def set_button_text_remove_tag(self) -> None:
        self.button_add_remove.setText("Remove")

    def add_tag(self, text: str) -> None:
        tag = Tag(text, self, self.fixed_height_policy)
        self.id2tag[tag.id] = tag
        self.ids.append(tag.id)
        self.layout.addWidget(tag)
        self.vline.setVisible(True)
        self.list_changed_signal.emit(self.get_values())

    def add_tag_prompt(self) -> None:
        text, ok = QInputDialog.getText(self, "Add Tag", "Enter tag text:")
        if ok and text:
            self.add_tag(text)

    def dragEnterEvent(self, event: QEvent) -> None:
        if event.mimeData().hasFormat(DRAG_ID_KEY):
            self.set_button_text_remove_tag()
            tag_id = event.mimeData().data(DRAG_ID_KEY).data()
            tag = self.id2tag[tag_id]
            tag.set_transparent_style()

            self.currently_dragged_id = tag_id
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QEvent) -> None:
        if event.mimeData().hasFormat(DRAG_ID_KEY):
            self._update_while_dragging(event)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QEvent) -> None:
        if event.mimeData().hasFormat(DRAG_ID_KEY):
            self._update_while_dragging(event)

            tag_id = event.mimeData().data(DRAG_ID_KEY).data()
            tag = self.id2tag[tag_id]
            tag.set_normal_style()

            self.currently_dragged_id = None
            event.setDropAction(Qt.MoveAction)
            event.accept()
            self.set_button_text_add_tag()

            if self.button_add_remove.geometry().contains(event.pos()):
                self.ids.remove(tag_id)
                self.id2tag.pop(tag_id)
                self.layout.removeWidget(tag)

                if not self.ids:
                    self.vline.setVisible(False)
            self.list_changed_signal.emit(self.get_values())
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QEvent) -> None:
        self.set_button_text_add_tag()
        if self.currently_dragged_id is not None:
            self.id2tag[self.currently_dragged_id].set_normal_style()
            self.currently_dragged_id = None
        event.accept()
        self.list_changed_signal.emit(self.get_values())

    def _update_while_dragging(self, event: QEvent) -> None:
        tag_id = event.mimeData().data(DRAG_ID_KEY).data()
        tag = self.id2tag[tag_id]

        pos = event.pos()

        if self.layout.geometry().contains(pos):
            tag_index = self.ids.index(tag_id)

            left_borders = np.array([self.id2tag[tag_id].pos().x() for tag_id in self.ids])
            insert_index = np.argmin(np.abs(left_borders - pos.x()))

            if insert_index is not None and insert_index != tag_index:

                self.layout.removeWidget(tag)
                self.layout.insertWidget(insert_index, tag)

                self.ids.remove(tag_id)
                if insert_index > len(self.ids):
                    self.ids.append(tag_id)
                else:
                    self.ids.insert(insert_index, tag_id)

    def get_values(self) -> List[str]:
        return [self.id2tag[id_].text() for id_ in self.ids]


class MultiSelectWidget(BaseMultiSelectWidget):
    def __init__(
        self,
        values: Sequence[str],
        selected_values: Optional[Sequence[str]] = None,
        unique_selection: bool = True,
        max_visible_items: int = 10,
        fit_all_items: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        assert isinstance(values, (list, tuple))
        assert len(values) == len(set(values))
        assert all(isinstance(v, str) for v in values)
        if selected_values is not None:
            assert isinstance(selected_values, (list, tuple))
            assert len(selected_values) == len(set(selected_values))
            assert all(isinstance(v, str) for v in selected_values)
            assert all(v in values for v in selected_values)

        self.all_values = list(values)
        self.unique_selection = unique_selection

        self.tag_selector = QtWidgets.QComboBox(self)
        self.tag_selector.setMaxVisibleItems(max_visible_items)

        self.tag_selector.setStyleSheet(
            """
                    QComboBox {
                        combobox-popup: 0;
                    }
                    QComboBox::item {
                        min-height: 30px;
                    }
                    QComboBox QAbstractItemView {
                        border: 2px solid darkgray;
                        selection-background-color: lightgray;
                    }
                    QScrollBar:vertical {
                        border: 1px solid #999999;
                        background:white;
                        width:10px;
                        margin: 0px 0px 0px 0px;
                    }
                    QScrollBar::handle:vertical {
                        min-height: 0px;
                        border: 2px solid grey;
                        border-radius: 4px;
                        background-color: lightgrey;
                    }
                    QScrollBar::add-line:vertical {
                        height: 0px;
                        subcontrol-position: bottom;
                        subcontrol-origin: margin;
                    }
                    QScrollBar::sub-line:vertical {
                        height: 0 px;
                        subcontrol-position: top;
                        subcontrol-origin: margin;
                    }
                """
        )

        self.tag_selector.addItems(self.all_values)
        self.tag_selector.hide()
        self.button_add_remove.clicked.disconnect()
        self.button_add_remove.clicked.connect(self.toggle_tag_selector)

        self.tag_selector.activated.connect(self.add_tag_from_selection)

        self.adjust_dropdown_width(fit_all_items)

        if selected_values:
            for value in selected_values:
                self.add_tag(value)

    def adjust_dropdown_width(self, fit_all_items: bool) -> None:
        withs_elems = []
        width_widget = self.tag_selector.width()  # Start with the current width of the ComboBox
        font_metrics = QtGui.QFontMetrics(self.tag_selector.font())
        withs_elems.append(width_widget)
        for i in range(self.tag_selector.count()):
            withs_elems.append(font_metrics.width(self.tag_selector.itemText(i)))
        padding = 40
        aggregate = np.max if fit_all_items else np.median
        width = round(aggregate(withs_elems)) + padding
        self.tag_selector.setMinimumWidth(width)

    def toggle_tag_selector(self) -> None:
        global_pos = QtGui.QCursor.pos()
        local_pos = self.mapFromGlobal(global_pos)

        self._update_available()

        self.tag_selector.move(local_pos)
        self.tag_selector.setFocus()
        self.tag_selector.showPopup()

    def add_tag_from_selection(self, index: int) -> None:
        if index >= 0:
            tag_text = self.tag_selector.itemText(index)
            self.add_tag(tag_text)
            self.tag_selector.setCurrentIndex(-1)

            self._update_available()

    def _update_available(self) -> None:
        if self.unique_selection:
            selected = self.get_values()
            available = [v for v in self.all_values if v not in selected]
            self.tag_selector.clear()
            self.tag_selector.addItems(available)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_widget = MultiSelectWidget(values=[str(v) for v in range(20)], selected_values=["1", "3"])
    main_widget.show()
    sys.exit(app.exec_())
