from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDesktopWidget,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
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
