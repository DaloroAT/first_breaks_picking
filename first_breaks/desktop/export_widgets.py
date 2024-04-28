import sys
from pathlib import Path
from typing import Optional, Dict, Sequence, Union
from uuid import uuid4

from PyQt5.QtCore import QPoint, pyqtSignal, Qt
from PyQt5.QtGui import QColor, QCloseEvent, QDoubleValidator
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QPushButton,
                             QListWidget, QListWidgetItem, QCheckBox, QRadioButton, QLabel,
                             QDialog, QWidget, QHBoxLayout, QButtonGroup, QMenu, QAction, QColorDialog, QLineEdit,
                             QSpinBox, QGridLayout, QDialogButtonBox, QComboBox, QTabWidget, QFileDialog, QMessageBox,
                             QAbstractItemView)
import numpy as np

from first_breaks.const import FIRST_BYTE
from first_breaks.data_models.dependent import TraceHeaderParams
from first_breaks.data_models.independent import PicksWidth
from first_breaks.desktop.byte_encode_unit_widget import QDialogByteEncodeUnit, QByteEncodeUnitWidget
from first_breaks.desktop.combobox_with_mapping import QComboBoxMapping
from first_breaks.desktop.last_folder_manager import last_folder_manager
from first_breaks.desktop.utils import set_geometry
from first_breaks.picking.picks import Picks
from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import generate_color, download_demo_sgy


class _ExporterWidget(QWidget):
    def __init__(self, picks: Picks, sgy: SGY, formats: Sequence[str]) -> None:
        super().__init__()
        self.picks = picks
        self.sgy = sgy
        self.formats = formats

        self._main_layout = QVBoxLayout()
        self.setLayout(self._main_layout)

        self.layout = QVBoxLayout()
        self._main_layout.addLayout(self.layout)

        self.export_button = QPushButton("Export to file", self)
        self.export_button.clicked.connect(self.export)
        self._main_layout.addWidget(self.export_button)

    def export(self):
        formats = ";; ".join(self.formats)
        filename, _ = QFileDialog.getSaveFileName(self, "Save result", directory=last_folder_manager.get_last_folder(), filter=formats)

        if filename:
            filename = Path(filename).resolve()
            filename.parent.mkdir(parents=True, exist_ok=True)

            try:
                self.export_to_file(filename)
                QMessageBox.information(self, "Export Successful", f"The data was successfully exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Failed to export data: {str(e)}")

    def export_to_file(self, filename: Union[str, Path]) -> None:
        raise NotImplementedError


class ExporterSGY(_ExporterWidget):
    def __init__(self, picks: Picks, sgy: SGY) -> None:
        formats = ["SEGY-file (*.segy *.sgy)"]
        super().__init__(picks=picks, sgy=sgy, formats=formats)

        self.export_params = QByteEncodeUnitWidget(first_byte=FIRST_BYTE, byte_position=237, encoding="I", picks_unit="mcs")
        self.layout.addWidget(self.export_params)

    def export_to_file(self, filename: Union[str, Path]) -> None:
        export_params = self.export_params.get_values()

        self.sgy.export_sgy_with_picks(
            output_fname=filename,
            picks_in_mcs=self.picks.picks_in_mcs,
            **export_params
        )


class ExporterTXT(_ExporterWidget):
    def __init__(self, picks: Picks, sgy: SGY) -> None:
        formats = ["TXT-file (*.txt)", "CSV-file (*.csv)"]
        super().__init__(picks=picks, sgy=sgy, formats=formats)

        self.separator = ','  # Default to comma
        self.include_column_names = False

        # Setup UI components directly in __init__
        self.column_list = self.setup_draggable_list(['Pick Time', 'Confidence Level', 'Other Info'])
        self.separator_selector = self.setup_separator_selector()
        self.include_names_checkbox = self.setup_include_names_checkbox()
        self.export_button = QPushButton("Export to TXT", self)
        self.export_button.clicked.connect(self.export)

        # Add widgets to layout
        self.layout.addWidget(self.column_list)
        self.layout.addWidget(self.separator_selector)
        self.layout.addWidget(self.include_names_checkbox)
        self.layout.addWidget(self.export_button)

    def setup_draggable_list(self, columns):
        column_list = QListWidget(self)
        column_list.setDragDropMode(QAbstractItemView.InternalMove)
        column_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        column_list.setAcceptDrops(True)
        column_list.setDragEnabled(True)
        column_list.setDropIndicatorShown(True)
        for column in columns:
            column_list.addItem(column)
        return column_list

    def setup_separator_selector(self):
        separator_selector = QComboBox(self)
        separator_selector.addItems(['Tab', 'Space', 'Comma'])
        separator_selector.currentTextChanged.connect(self.on_separator_changed)
        return separator_selector

    def on_separator_changed(self, text):
        if text == "Tab":
            self.separator = '\t'
        elif text == "Space":
            self.separator = ' '
        elif text == "Comma":
            self.separator = ','

    def setup_include_names_checkbox(self):
        checkbox = QCheckBox("Include Column Names", self)
        checkbox.stateChanged.connect(self.on_checkbox_state_changed)
        return checkbox

    def on_checkbox_state_changed(self, state):
        self.include_column_names = state == Qt.Checked



if __name__ == "__main__":
    app = QApplication([])

    sgy_ = SGY(download_demo_sgy())
    picks_ = Picks(values=[1000] * sgy_.num_traces, dt_mcs=sgy_.dt_mcs, unit='mcs')

    # window = _ExporterWidget(picks=picks_, sgy=sgy_, formats=[])
    # window = ExporterSGY(picks=picks_, sgy=sgy_)
    window = ExporterTXT(picks=picks_, sgy=sgy_)


    window.show()
    app.exec_()
