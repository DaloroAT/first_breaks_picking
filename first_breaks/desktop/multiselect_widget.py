import sys
import uuid
from typing import List, Dict, Optional

import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QByteArray, QMimeData, pyqtSignal
from PyQt5.QtGui import QDrag, QMouseEvent
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QPushButton, QInputDialog, QLineEdit

from first_breaks.desktop.utils import QVSeparationLine

DRAG_ID_KEY = "application/x-draggable-id"


def set_normal_style(widget: QWidget):
    widget.setStyleSheet("""
                background-color: rgba(211, 211, 211, 255);
                border: 1px solid gray;
                color: rgb(0, 0, 0);
                padding: 5px;
                border-radius: 5px;
            """)


def set_transparent_style(widget: QWidget):
    widget.setStyleSheet("""
                background-color: rgba(211, 211, 211, 0);
                border: rgba(211, 211, 211, 0);
                color: rgba(0, 0, 0, 0);
            """)


class Tag(QLineEdit):
    def __init__(self, text, parent=None):
        super(Tag, self).__init__(text, parent)
        self.id = uuid.uuid4().bytes
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setReadOnly(True)
        self.setToolTip(text)
        self.setAlignment(Qt.AlignCenter)

        self.set_normal_style()

    def set_normal_style(self):
        set_normal_style(self)

    def set_transparent_style(self):
        set_transparent_style(self)

    def dragLeaveEvent(self, event):
        self.set_normal_style()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
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

    def __init__(self):
        super(BaseMultiSelectWidget, self).__init__()
        self._main_layout = QHBoxLayout(self)
        self.setLayout(self._main_layout)

        self.layout = QHBoxLayout(self)
        self._main_layout.addLayout(self.layout)

        self.setAcceptDrops(True)
        self.id2tag: Dict[bytes, Tag] = {}
        self.ids: List[bytes] = []
        self.currently_dragged_id: Optional[bytes] = None

        self.vline = QVSeparationLine()
        self.vline.setVisible(False)
        self._main_layout.addWidget(self.vline)

        self.button_add_remove = QPushButton("")
        self.set_button_text_add_tag()
        self.button_add_remove.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.button_add_remove.setFixedWidth(100)
        set_normal_style(self.button_add_remove)
        self.button_add_remove.clicked.connect(self.add_tag_prompt)
        self._main_layout.addWidget(self.button_add_remove)

        # self.add_tag("1" * 10)
        # self.add_tag("2" * 10)
        # self.add_tag("3" * 10)
        # self.add_tag("4" * 10)
        # self.add_tag("1")

    def set_button_text_add_tag(self):
        self.button_add_remove.setText("Add")

    def set_button_text_remove_tag(self):
        self.button_add_remove.setText("Remove")

    def add_tag(self, text):
        tag = Tag(text, self)
        self.id2tag[tag.id] = tag
        self.ids.append(tag.id)
        self.layout.addWidget(tag)
        self.vline.setVisible(True)
        self.list_changed_signal.emit(self.get_values())

    def add_tag_prompt(self):
        text, ok = QInputDialog.getText(self, 'Add Tag', 'Enter tag text:')
        if ok and text:
            self.add_tag(text)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(DRAG_ID_KEY):
            self.set_button_text_remove_tag()
            tag_id = event.mimeData().data(DRAG_ID_KEY).data()
            tag = self.id2tag[tag_id]
            tag.set_transparent_style()

            self.currently_dragged_id = tag_id
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(DRAG_ID_KEY):
            self._update_while_dragging(event)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
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

    def dragLeaveEvent(self, event):
        self.set_button_text_add_tag()
        if self.currently_dragged_id is not None:
            self.id2tag[self.currently_dragged_id].set_normal_style()
            self.currently_dragged_id = None
        event.accept()
        self.list_changed_signal.emit(self.get_values())

    def _update_while_dragging(self, event):
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
    def __init__(self, predefined_values=None):
        super().__init__()
        self.predefined_values = predefined_values if predefined_values is not None else ["Tag 1", "Tag 2", "Tag 3"]

        self.tag_selector = QtWidgets.QComboBox(self)
        self.tag_selector.addItems(self.predefined_values)
        self.tag_selector.hide()
        self.button_add_remove.clicked.disconnect()
        self.button_add_remove.clicked.connect(self.toggle_tag_selector)

        # Connect the 'activated' signal to actually add tags
        self.tag_selector.activated.connect(self.add_tag_from_selection)

    def toggle_tag_selector(self):
        if self.tag_selector.isVisible():
            self.tag_selector.hide()
        else:
            self.tag_selector.move(QtGui.QCursor.pos())  # Positioning at mouse cursor
            self.tag_selector.show()
            self.tag_selector.showPopup()  # Automatically expand the dropdown

    def add_tag_from_selection(self, index):
        # Add the tag selected from the combo box
        if index >= 0:  # Check valid index
            tag_text = self.tag_selector.itemText(index)
            self.add_tag(tag_text)
            self.tag_selector.setCurrentIndex(-1)  # Reset the combo box
            self.tag_selector.hide()  # Hide the selector after adding the tag


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_widget = MultiSelectWidget()
    main_widget.show()
    sys.exit(app.exec_())
