from typing import Any, Dict, List, Optional, Tuple, Union

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDesktopWidget,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class MessageBox(QDialog):
    def __init__(
        self,
        parent: QWidget,
        title: str = "Error",
        message: str = "Error",
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
        self.layout.addWidget(self.button_box)
        self.setLayout(self.layout)


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

    monitor = QDesktopWidget().screenGeometry(1)
    widget.move(monitor.left(), monitor.top())


class QHSeparationLine(QtWidgets.QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumWidth(1)
        self.setFixedHeight(20)
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)


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
