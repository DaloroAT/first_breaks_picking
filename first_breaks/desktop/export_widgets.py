from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from first_breaks.const import FIRST_BYTE
from first_breaks.desktop.byte_encode_unit_widget import QByteEncodeUnitWidget
from first_breaks.desktop.combobox_with_mapping import QComboBoxMapping
from first_breaks.desktop.last_folder_manager import last_folder_manager
from first_breaks.desktop.multiselect_widget import MultiSelectWidget
from first_breaks.desktop.utils import LabelWithHelp
from first_breaks.exports.export_picks import (
    COL_PICKS_IN_MCS,
    PICKS_COLUMNS,
    export_to_json,
    export_to_sgy,
    export_to_txt,
)
from first_breaks.picking.picks import PickingParameters, Picks
from first_breaks.sgy.headers import TraceHeaders
from first_breaks.sgy.reader import SGY


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

    def export(self) -> None:
        formats = ";; ".join(self.formats)
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save result", directory=last_folder_manager.get_last_folder(), filter=formats
        )

        if filename:
            filename = Path(filename).resolve()
            filename.parent.mkdir(parents=True, exist_ok=True)

            last_folder_manager.set_last_folder(filename)

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
    def __init__(
        self, picks: Picks, sgy: SGY, byte_position: int = 237, encoding: str = "I", picks_unit: str = "mcs"
    ) -> None:
        formats = ["SEGY-file (*.segy *.sgy)"]
        super().__init__(picks=picks, sgy=sgy, formats=formats)

        self.export_params = QByteEncodeUnitWidget(
            first_byte=FIRST_BYTE, byte_position=byte_position, encoding=encoding, picks_unit=picks_unit
        )
        self.layout.addWidget(self.export_params)

    def export_to_file(self, filename: Union[str, Path]) -> None:
        export_params = self.export_params.get_values()
        export_to_sgy(sgy=self.sgy, filename=filename, picks=self.picks, **export_params)

    def get_values(self) -> Dict[str, Any]:
        return self.export_params.get_values()


columns_selector_tip = """
- You can select which columns to add for export by clicking on the "Add" button.\n
- Columns are exported in the order they are presented below.\n
- Drag the tag if you want to change the position of the column and drop it in the new location.\n
- When moving a tag, the "Add" button is replaced with "Remove". Throw a tag there if you don't need this column.
"""


def _add_picks_tag_prefix(name: str) -> str:
    return f"* {name}"


class _ColumnExporter(_ExporterWidget):
    def __init__(
        self,
        picks: Picks,
        sgy: SGY,
        formats: Sequence[str],
        selected_tags: Optional[Sequence[str]] = (_add_picks_tag_prefix(COL_PICKS_IN_MCS),),
    ):
        super().__init__(picks=picks, sgy=sgy, formats=formats)

        self.tags_selector_label = LabelWithHelp("Select columns", columns_selector_tip)
        self.layout.addWidget(self.tags_selector_label, 0, 0, Qt.AlignmentFlag.AlignTop)

        self.tags, self.tag2column = self._prepare_tags()

        self.tags_selector = MultiSelectWidget(
            self.tags,
            selected_values=selected_tags,
            unique_selection=True,
            fixed_height_policy=True,
        )
        self.layout.addWidget(self.tags_selector, 1, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)

        self.tags_selector.list_changed_signal.connect(self.set_enable_export)
        self.export_button.setEnabled(bool(self.tags_selector.get_values()))

    def set_enable_export(self, selected_tags: List[str]) -> None:
        self.export_button.setEnabled(bool(selected_tags))

    @staticmethod
    def _prepare_tags() -> Tuple[List[str], Dict[str, str]]:
        tags = []
        tag2column = {}

        for col in PICKS_COLUMNS:
            tag = f"* {col}"
            tags.append(tag)
            tag2column[tag] = col

        for pos, name, _ in TraceHeaders().headers_schema:
            tag = f"{pos + FIRST_BYTE} - {name}"
            tags.append(tag)
            tag2column[tag] = name

        return tags, tag2column


class ExporterTXT(_ColumnExporter):
    def __init__(
        self,
        picks: Picks,
        sgy: SGY,
        selected_tags: Optional[Sequence[str]] = (_add_picks_tag_prefix(COL_PICKS_IN_MCS),),
        separator: str = "\t",
        include_column_names: bool = True,
        precision: int = 3,
    ) -> None:
        formats = ["TXT-file (*.txt)", "CSV-file (*.csv)"]
        super().__init__(picks=picks, sgy=sgy, selected_tags=selected_tags, formats=formats)

        self.separator_label = QLabel("Separator")
        self.layout.addWidget(self.separator_label, 2, 0)

        separator_mapping = {0: ["Tab", "\t"], 1: ["Space", " "], 2: ["Comma", ","]}
        self.separator_selector = QComboBoxMapping(separator_mapping, current_value=separator)  # type: ignore
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
        self.layout.addWidget(self.precision_selector, 4, 1, Qt.AlignmentFlag.AlignRight)

    def export_to_file(self, filename: Union[str, Path]) -> None:
        values = self.get_values()
        precision = values["precision"]
        include_column_names = values["include_column_names"]
        separator = values["separator"]

        columns = [self.tag2column[tag] for tag in values["selected_tags"]]

        export_to_txt(
            sgy=self.sgy,
            filename=filename,
            picks=self.picks,
            columns=columns,
            separator=separator,
            include_column_names=include_column_names,
            precision=precision,
        )

    def get_values(self) -> Dict[str, Any]:
        return {
            "precision": int(self.precision_selector.text()),
            "include_column_names": self.column_names_checkbox.isChecked(),
            "separator": self.separator_selector.value(),
            "selected_tags": self.tags_selector.get_values(),
        }


class ExporterJSON(_ColumnExporter):
    def __init__(
        self,
        picks: Picks,
        sgy: SGY,
        selected_tags: Optional[Sequence[str]] = (_add_picks_tag_prefix(COL_PICKS_IN_MCS),),
        include_picking_parameters: bool = True,
    ) -> None:
        formats = [
            "JSON-file (*.json)",
        ]
        super().__init__(picks=picks, sgy=sgy, selected_tags=selected_tags, formats=formats)

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
        values = self.get_values()
        include_picking_parameters = values["include_picking_parameters"]
        columns = [self.tag2column[tag] for tag in values["selected_tags"]]

        export_to_json(
            sgy=self.sgy,
            filename=filename,
            picks=self.picks,
            columns=columns,
            include_picking_parameters=include_picking_parameters,
        )

    def get_values(self) -> Dict[str, Any]:
        return {
            "include_picking_parameters": self.picking_parameters_checkbox.isChecked(),
            "selected_tags": self.tags_selector.get_values(),
        }


if __name__ == "__main__":
    from first_breaks.utils.utils import download_demo_sgy

    app = QApplication([])
    sgy_ = SGY(download_demo_sgy())
    picks_ = Picks(
        values=[float(f"{v}.{v}") for v in range(sgy_.num_traces)],
        dt_mcs=sgy_.dt_mcs,
        unit="mcs",
        picking_parameters=PickingParameters(),
    )

    # window = ExporterSGY(picks=picks_, sgy=sgy_)
    # window = ExporterTXT(picks=picks_, sgy=sgy_)
    window = ExporterJSON(picks=picks_, sgy=sgy_)

    window.show()
    app.exec_()
