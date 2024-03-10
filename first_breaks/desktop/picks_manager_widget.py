import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QPushButton,
                             QListWidget, QListWidgetItem, QCheckBox, QRadioButton, QLabel,
                             QDialog, QWidget, QHBoxLayout, QButtonGroup)


class PropertiesDialog(QDialog):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QPushButton("Close", self, clicked=self.close))
        self.setLayout(layout)


class CustomItemWidget(QWidget):
    def __init__(self, text=""):
        super().__init__()

        self.checkbox = QCheckBox(self)
        self.radiobutton = QRadioButton(self)
        self.label = QLabel(text, self)

        # Set fixed widths for the checkbox and radiobutton
        checkbox_width = 30
        radiobutton_width = 30
        self.checkbox.setFixedWidth(checkbox_width)
        self.radiobutton.setFixedWidth(radiobutton_width)

        layout = QHBoxLayout()
        layout.addWidget(self.checkbox)
        layout.addWidget(self.radiobutton)
        layout.addWidget(self.label, 1)

        self.setLayout(layout)


class ListManager(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('List Manager')
        self.setGeometry(100, 100, 300, 400)

        layout = QVBoxLayout()

        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.list_widget.itemDoubleClicked.connect(self.open_properties)
        layout.addWidget(self.list_widget)

        self.add_button = QPushButton("+", self)
        self.add_button.clicked.connect(self.add_item)
        self.remove_button = QPushButton("-", self)
        self.remove_button.clicked.connect(self.remove_item)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        layout.addLayout(button_layout)

        self.radio_group = QButtonGroup(self)  # Group for radio buttons

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def add_item(self):
        item = QListWidgetItem()
        self.list_widget.addItem(item)

        custom_widget = CustomItemWidget(f"Item {self.list_widget.count()}")
        self.list_widget.setItemWidget(item, custom_widget)
        item.setSizeHint(custom_widget.sizeHint())

        # Add the radiobutton to the group
        self.radio_group.addButton(custom_widget.radiobutton)

    def remove_item(self):
        current_row = self.list_widget.currentRow()
        if current_row >= 0:
            item = self.list_widget.item(current_row)
            self.radio_group.removeButton(self.list_widget.itemWidget(item).radiobutton)
            self.list_widget.takeItem(current_row)

    def open_properties(self, item):
        # This method now should open a properties dialog specific to the item
        # For simplicity, this example just opens a generic dialog; customize as needed
        dialog = PropertiesDialog()
        dialog.exec_()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ListManager()
    window.show()
    sys.exit(app.exec_())
