from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtWidgets import (
    QDesktopWidget,
    QDialog,
    QDialogButtonBox,
    QGraphicsSceneMouseEvent,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph import ViewBox


class MessageBox(QDialog):
    def __init__(
        self,
        parent: QWidget,
        title: str = "Error",
        message: str = "Error",
        detailed_message: str = None,
        add_cancel_option: bool = False,
        label_ok: str = "Ok",
        label_cancel: str = "Cancel",
    ):
        super().__init__(parent=parent)

        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        set_geometry(self, width_rel=0.2, height_rel=0.2, centralize=True, fix_size=False)

        error_label = QLabel(message)
        error_label.setWordWrap(True)
        error_label.setTextInteractionFlags(Qt.LinksAccessibleByMouse | Qt.TextSelectableByMouse)

        self.button_box = QDialogButtonBox()
        self.button_box.addButton(label_ok, QDialogButtonBox.AcceptRole)
        self.button_box.accepted.connect(self.accept)
        if add_cancel_option:
            self.button_box.addButton(label_cancel, QDialogButtonBox.RejectRole)
            self.button_box.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.layout.addWidget(error_label)

        if detailed_message:
            self.details_button = QPushButton("Show Details")
            self.details_text = QTextEdit()
            self.details_text.setPlainText(detailed_message)
            self.details_text.setVisible(False)
            self.details_text.setReadOnly(True)

            self.details_button.clicked.connect(self.toggle_details)

            self.layout.addWidget(self.details_button)
            self.layout.addWidget(self.details_text)

        self.layout.addWidget(self.button_box)
        self.setLayout(self.layout)

    def toggle_details(self) -> None:
        if self.details_text.isVisible():
            self.details_text.setVisible(False)
            self.details_button.setText("Show Details")
        else:
            self.details_text.setVisible(True)
            self.details_button.setText("Hide Details")


def set_geometry(
    widget: QWidget,
    width_rel: float,
    height_rel: float,
    centralize: bool = True,
    fix_size: bool = False,
) -> None:
    assert 0 < width_rel <= 1
    assert 0 < height_rel <= 1
    h, w = widget.screen().size().height(), widget.screen().size().width()
    width = int(width_rel * w)
    height = int(height_rel * h)
    widget.setGeometry(0, 0, width, height)

    if centralize:
        qt_rectangle = widget.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        qt_rectangle.moveCenter(center_point)
        widget.move(qt_rectangle.topLeft())

    if fix_size:
        widget.setFixedSize(width, height)

    # monitor = QDesktopWidget().screenGeometry(1)
    # widget.move(monitor.left(), monitor.top())


class QSeparationLine(QtWidgets.QWidget):
    def __init__(self, text: str = "", type_line: Literal["h", "v"] = "h"):
        super().__init__()

        assert type_line in ["h", "v"]

        if type_line == "h":
            layout = QtWidgets.QHBoxLayout(self)
            size_policy = (QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            frame_shape = QtWidgets.QFrame.HLine
        else:
            layout = QtWidgets.QVBoxLayout(self)
            size_policy = (QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
            frame_shape = QtWidgets.QFrame.VLine

        self.line = QtWidgets.QFrame(self)
        self.line.setFrameShape(frame_shape)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)

        self.line.setSizePolicy(*size_policy)

        if text:
            self.label = QtWidgets.QLabel(text, self)
            self.label.setStyleSheet("background-color: transparent; color: grey;")
            self.label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.label)
            layout.setSpacing(10)

        layout.addWidget(self.line)
        layout.setContentsMargins(1, 1, 1, 1)

        self.setLayout(layout)


class QHSeparationLine(QSeparationLine):
    def __init__(self, text: str = ""):
        super().__init__(text=text, type_line="h")


class QVSeparationLine(QSeparationLine):
    def __init__(self, text: str = ""):
        super().__init__(text=text, type_line="v")


TMappingSetup = Dict[int, Union[Tuple[str, Any], List[Any]]]


def validate_mapping_setup_and_get_current_index(
    mapping: TMappingSetup,
    current_index: Optional[int] = None,
    current_label: Optional[str] = None,
    current_value: Optional[Any] = None,
) -> int:
    assert all(isinstance(k, int) for k in mapping.keys())
    assert all(isinstance(v, (list, tuple)) and len(v) == 2 for v in mapping.values())
    assert all(isinstance(label, str) for label, _ in mapping.values())
    assert set(range(len(mapping.keys()))) == set(mapping.keys())

    current_is_not_none = [v is not None for v in [current_index, current_label, current_value]]
    assert (
        0 <= sum(current_is_not_none) <= 1
    ), "Only one of 'current_index' or 'current_label' or 'current_value' might be specified."

    if current_label or current_value:
        if current_label:
            assert isinstance(current_label, str)
            idx_to_check = 0
            type_to_check = "Label"
            got = current_label
        else:
            idx_to_check = 1
            type_to_check = "Value"
            got = current_value

        corresponding_ids = [k for k, v in mapping.items() if v[idx_to_check] == got]

        if not corresponding_ids:
            raise KeyError(
                f"{type_to_check} '{type_to_check}' not found in mapping. "
                f"Available {[v[idx_to_check] for v in mapping.values()]}, got '{got}'"
            )
        current_index = corresponding_ids[0]
    elif current_index:
        assert isinstance(current_index, int)
        assert 0 <= current_index < len(mapping)
    else:
        current_index = 0

    return current_index


def get_mouse_position_in_scene_coords(
    event_or_point: Union[QGraphicsSceneMouseEvent, QEvent, QPointF, QPoint], viewbox: ViewBox
) -> QPointF:
    if isinstance(event_or_point, QGraphicsSceneMouseEvent):
        point = event_or_point.scenePos()
    elif isinstance(event_or_point, QEvent):
        point = event_or_point.localPos()
    elif isinstance(event_or_point, (QPointF, QPoint)):
        point = event_or_point
    else:
        raise TypeError("Only events and points are available")
    return viewbox.mapSceneToView(point)


class LabelWithHelp(QWidget):
    def __init__(self, text: str, help_string: str, move_help_close_to_label: bool = True) -> None:
        super().__init__()
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel(text)
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self.help_button = QPushButton()
        self.help_button.setFixedWidth(20)
        icon = self.style().standardIcon(QStyle.SP_MessageBoxQuestion)
        self.help_button.setIcon(icon)
        self.help_button.setToolTip(help_string)
        self.help_button.setStyleSheet("QPushButton { border: none; background-color: transparent; }")
        self.help_button.setCursor(QtGui.QCursor(QtCore.Qt.WhatsThisCursor))

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.help_button)
        if move_help_close_to_label:
            self.layout.addStretch(1)
        self.layout.setContentsMargins(0, 0, 0, 0)
