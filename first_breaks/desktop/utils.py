from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDesktopWidget,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout, QWidget,
)


class MessageBox(QDialog):
    def __init__(self,
                 parent: QWidget,
                 title: str = "Error",
                 message: str = "Error",
                 add_cancel_option: bool = False,
                 label_ok: str = "Ok",
                 label_cancel: str = "Cancel"):
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


def set_geometry(widget: QWidget,
                 width_rel: float,
                 height_rel: float,
                 centralize: bool = True,
                 fix_size: bool = False,) -> None:
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

    monitor = QDesktopWidget().screenGeometry(1)
    widget.move(monitor.left(), monitor.top())

    if fix_size:
        widget.setFixedSize(width, height)


class QHSeparationLine(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(1)
        self.setFixedHeight(20)
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
