import json
from pathlib import Path
from typing import Sequence, Union, List, Optional, Any, Dict

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QPushButton,
                             QLabel,
                             QWidget, QGridLayout, QFileDialog, QMessageBox, QCheckBox, QSpinBox, QTabWidget)

from first_breaks.const import FIRST_BYTE
from first_breaks.desktop.byte_encode_unit_widget import QByteEncodeUnitWidget
from first_breaks.desktop.combobox_with_mapping import QComboBoxMapping
from first_breaks.desktop.last_folder_manager import last_folder_manager
from first_breaks.desktop.multiselect_widget import MultiSelectWidget
from first_breaks.desktop.utils import LabelWithHelp
from first_breaks.picking.picks import Picks, PickingParameters
from first_breaks.sgy.headers import TraceHeaders
from first_breaks.sgy.reader import SGY
import numpy as np


class _ExporterWidget(QWidget):
    def __init__(self, picks: Picks, sgy: SGY, formats: Sequence[str]) -> None:
        super().__init__()
        self.picks = picks
        self.sgy = sgy
        self.formats = formats

        self._main_layout = QVBoxLayout()
        self.setLayout(self._main_layout)

        self.layout: QGridLayout = QGridLayout()
        self._main_layout.addLayout(self.layout)

        self._main_layout.addStretch(1)

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

    def get_values(self) -> Dict[str, Any]:
        raise NotImplementedError


class ExporterSGY(_ExporterWidget):
    def __init__(self, picks: Picks, sgy: SGY, byte_position: int = 237, encoding: str = "I",
                 picks_unit: str = "mcs") -> None:
        formats = ["SEGY-file (*.segy *.sgy)"]
        super().__init__(picks=picks, sgy=sgy, formats=formats)

        self.export_params = QByteEncodeUnitWidget(first_byte=FIRST_BYTE, byte_position=byte_position,
                                                   encoding=encoding, picks_unit=picks_unit)
        self.layout.addWidget(self.export_params)

    def export_to_file(self, filename: Union[str, Path]) -> None:
        export_params = self.export_params.get_values()

        self.sgy.export_sgy_with_picks(
            output_fname=filename,
            picks_in_mcs=self.picks.picks_in_mcs,
            **export_params
        )

    def get_values(self) -> Dict[str, Any]:
        return self.export_params.get_values()


columns_selector_tip = """
- You can select which columns to add for export by clicking on the "Add" button.\n
- Columns are exported in the order they are presented below.\n
- Drag the tag if you want to change the position of the column and drop it in the new location.\n
- When moving a tag, the "Add" button is replaced with "Remove". Throw a tag there if you don't need this column.
"""


class PicksCols:
    COL_PICKS_IN_MS = "* Picks, ms"
    COL_PICKS_IN_MCS = "* Picks, mcs"
    COL_PICKS_IN_SAMPLES = "* Picks, samples"
    COL_CONFIDENCE = "* Confidence"


class _ColumnExporter(_ExporterWidget):
    def __init__(
            self,
            picks: Picks,
            sgy: SGY,
            formats: Sequence[str],
            selected_columns: Optional[Sequence[str]] = (PicksCols.COL_PICKS_IN_MCS,),
    ):
        super().__init__(picks=picks, sgy=sgy, formats=formats)

        self.columns_selector_label = LabelWithHelp("Select columns", columns_selector_tip)
        self.layout.addWidget(self.columns_selector_label, 0, 0, Qt.AlignmentFlag.AlignTop)

        self.columns, self.column2vgetter, self.column2name = self._prepare_columns()

        self.columns_selector = MultiSelectWidget(
            self.columns,
            selected_values=selected_columns,
            unique_selection=True,
            fixed_height_policy=True,
        )
        self.layout.addWidget(self.columns_selector, 1, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)

        self.columns_selector.list_changed_signal.connect(self.set_enable_export)
        self.export_button.setEnabled(bool(self.columns_selector.get_values()))

    def set_enable_export(self, selected_columns: List[str]):
        self.export_button.setEnabled(bool(selected_columns))

    def _prepare_columns(self):
        columns = []
        column2vgetter = {}
        column2name = {}

        col_getter_specials = [(PicksCols.COL_PICKS_IN_MS, lambda: self.picks.picks_in_ms),
                               (PicksCols.COL_PICKS_IN_MCS, lambda: self.picks.picks_in_mcs),
                               (PicksCols.COL_PICKS_IN_SAMPLES, lambda: self.picks.picks_in_samples)]

        if self.picks.confidence is not None:
            col_getter_specials.append([(PicksCols.COL_CONFIDENCE, lambda: self.picks.confidence)])

        for col, getter in col_getter_specials:
            columns.append(col)
            column2vgetter[col] = getter
            column2name[col] = col.replace("*", "").strip()

        for pos, name, _ in TraceHeaders().headers_schema:
            label = f"{pos + FIRST_BYTE} - {name}"
            columns.append(label)
            column2vgetter[label] = lambda: self.sgy.traces_headers[name]
            column2name[label] = name

        return columns, column2vgetter, column2name

    def _process_column_values(self, values: np.ndarray) -> np.ndarray:
        return values

    def _prepare_column_values_to_export(self):
        columns = self.columns_selector.get_values()

        to_export = {}

        for col in columns:
            values = self.column2vgetter[col]()
            values = self._process_column_values(np.array(values))
            assert isinstance(values, np.ndarray)
            if col in [PicksCols.COL_PICKS_IN_MCS, PicksCols.COL_PICKS_IN_SAMPLES]:
                values = values.astype(int)
            values = values.tolist()

            to_export[self.column2name[col]] = values

        return to_export


class ExporterTXT(_ColumnExporter):
    def __init__(
        self,
        picks: Picks,
        sgy: SGY,
        selected_columns: Optional[Sequence[str]] = (PicksCols.COL_PICKS_IN_MCS,),
        separator: str = "\t",
        include_column_names: bool = True,
        precision: int = 3
    ) -> None:
        formats = ["TXT-file (*.txt)", "CSV-file (*.csv)"]
        super().__init__(picks=picks, sgy=sgy, selected_columns=selected_columns, formats=formats)

        self.separator_label = QLabel("Separator")
        self.layout.addWidget(self.separator_label, 2, 0)

        separator_mapping = {0: ["Tab", "\t"], 1: ["Space", " "], 2: ["Comma", ","]}
        self.separator_selector = QComboBoxMapping(separator_mapping, current_value=separator)
        self.layout.addWidget(self.separator_selector, 2, 1, Qt.AlignmentFlag.AlignRight)

        self.column_names_label = QLabel("Include column names")
        self.layout.addWidget(self.column_names_label, 3, 0)

        self.column_names_checkbox = QCheckBox()
        self.column_names_checkbox.setChecked(include_column_names)
        self.layout.addWidget(self.column_names_checkbox, 3, 1, Qt.AlignmentFlag.AlignRight)

        self.precision_label = QLabel("Decimal places")
        self.layout.addWidget(self.precision_label, 4, 0)

        self.precision_selector = QSpinBox()
        self.precision_selector.setMinimum(0)
        self.precision_selector.setMaximum(10)
        self.precision_selector.setValue(precision)
        self.precision = int(self.precision_selector.text())
        self.layout.addWidget(self.precision_selector, 4, 1, Qt.AlignmentFlag.AlignRight)

    def _process_column_values(self, values: np.ndarray) -> np.ndarray:
        return np.round(values, self.precision)

    def export_to_file(self, filename: Union[str, Path]) -> None:
        # self.precision = int(self.precision_selector.text())
        # include_column_names = self.column_names_checkbox.isChecked()
        # separator = self.separator_selector.value()
        values = self.get_values()
        self.precision = values["precision"]
        include_column_names = values["include_column_names"]
        separator = values["separator"]

        to_export = self._prepare_column_values_to_export()

        pd.DataFrame(to_export).to_csv(filename, index=False, sep=separator, header=include_column_names)

    def get_values(self) -> Dict[str, Any]:
        return {"precision": int(self.precision_selector.text()),
                "include_column_names": self.column_names_checkbox.isChecked(),
                "separator": self.separator_selector.value(),
                "selected_columns": self.columns_selector.get_values()}


class ExporterJSON(_ColumnExporter):
    def __init__(
        self,
        picks: Picks,
        sgy: SGY,
        selected_columns: Optional[Sequence[str]] = (PicksCols.COL_PICKS_IN_MCS,),
        include_picking_parameters: bool = True,
    ) -> None:
        formats = ["JSON-file (*.json)",]
        super().__init__(picks=picks, sgy=sgy, selected_columns=selected_columns, formats=formats)

        with_picking_parameters = picks.picking_parameters is not None
        message_no_parameters = "Avaialable only for picks from NN"

        self.picking_parameters_label = QLabel("Include picking paramteres")
        self.picking_parameters_label.setEnabled(with_picking_parameters)
        if not with_picking_parameters:
            self.picking_parameters_label.setToolTip(message_no_parameters)
        self.layout.addWidget(self.picking_parameters_label, 2, 0)

        self.picking_parameters_checkbox = QCheckBox("")
        self.layout.addWidget(self.picking_parameters_checkbox, 2, 1)
        self.picking_parameters_checkbox.setChecked(include_picking_parameters and with_picking_parameters)
        if not with_picking_parameters:
            self.picking_parameters_checkbox.setToolTip(message_no_parameters)
        self.picking_parameters_checkbox.setEnabled(include_picking_parameters and with_picking_parameters)

    def export_to_file(self, filename: Union[str, Path]) -> None:
        # include_picking_parameters = self.picking_parameters_checkbox.isChecked()

        include_picking_parameters = self.get_values()["include_picking_parameters"]

        to_export = self._prepare_column_values_to_export()

        if include_picking_parameters and self.picks.picking_parameters is not None:
            to_export["picking_parameters"] = self.picks.picking_parameters.model_dump()

        with open(filename, "w") as f:
            json.dump(to_export, f)

    def get_values(self) -> Dict[str, Any]:
        return {"include_picking_parameters": self.picking_parameters_checkbox.isChecked(),
                "selected_columns": self.columns_selector.get_values()}


class TabExporter(QWidget):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.tab = QTabWidget()
        self.layout.addWidget(self.tab)


if __name__ == "__main__":
    from first_breaks.utils.utils import download_demo_sgy

    app = QApplication([])
    sgy_ = SGY(download_demo_sgy())
    picks_ = Picks(values=[float(f"{v}.{v}") for v in range(sgy_.num_traces)], dt_mcs=sgy_.dt_mcs, unit='mcs', picking_parameters=PickingParameters())

    # window = _ExporterWidget(picks=picks_, sgy=sgy_, formats=[])
    # window = ExporterSGY(picks=picks_, sgy=sgy_)
    # window = ExporterTXT(picks=picks_, sgy=sgy_)
    window = ExporterJSON(picks=picks_, sgy=sgy_)

    window.show()
    app.exec_()
