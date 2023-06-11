from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDesktopWidget,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout, QWidget,
)


class WarnBox(QDialog):
    def __init__(self, parent, title="Error", message="Error"):  # type: ignore
        super().__init__(parent=parent)

        self.setWindowTitle(title)
        self.setGeometry(100, 100, 350, 100)

        qt_rectangle = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        qt_rectangle.moveCenter(center_point)
        self.move(qt_rectangle.topLeft())

        QBtn = QDialogButtonBox.Ok
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        error_label = QLabel(message)
        error_label.setTextInteractionFlags(Qt.LinksAccessibleByMouse | Qt.TextSelectableByMouse)

        self.layout = QVBoxLayout()
        self.layout.addWidget(error_label)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


def set_geometry(widget: QWidget,
                 width_rel: float,
                 height_rel: float,
                 centralize: bool = True,
                 fix_size: bool = False) -> None:
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
