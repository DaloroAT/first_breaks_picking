import sys
from typing import Optional, Dict, Any

import numpy as np
from PyQt5.QtCore import QPoint, pyqtSignal
from PyQt5.QtGui import QColor, QCloseEvent, QDoubleValidator
from PyQt5.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
    QRadioButton,
    QLabel,
    QDialog,
    QWidget,
    QHBoxLayout,
    QButtonGroup,
    QMenu,
    QAction,
    QColorDialog,
    QLineEdit,
    QDialogButtonBox,
    QTabWidget,
)

from first_breaks.const import FIRST_BYTE
from first_breaks.desktop.byte_encode_unit_widget import QDialogByteEncodeUnit
from first_breaks.desktop.combobox_with_mapping import QComboBoxMapping
from first_breaks.desktop.export_widgets import ExporterSGY, ExporterTXT, ExporterJSON
from first_breaks.desktop.utils import set_geometry
from first_breaks.picking.picks import Picks, DEFAULT_PICKS_WIDTH
from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import generate_color


class ConstantValuesInputDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enter value in microseconds")

        layout = QVBoxLayout(self)
        input_layout = QHBoxLayout()

        self.line_edit = QLineEdit(self)
        validator = QDoubleValidator()
        validator.setBottom(0)
        validator.setDecimals(0)
        self.line_edit.setValidator(validator)
        self.line_edit.setText(str(0))
        input_layout.addWidget(self.line_edit)

        self.unit_label = QLabel("mcs", self)
        input_layout.addWidget(self.unit_label)

        layout.addLayout(input_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout.addWidget(self.button_box)

    def get_value(self):
        return self.line_edit.text()


class AggregationDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Aggregation Method")

        layout = QVBoxLayout(self)

        mapping = {
            0: ("Mean", lambda x: np.mean(x, axis=0)),
            1: ("Median", lambda x: np.median(x, axis=0)),
            2: ("RMS", lambda x: np.sqrt(np.mean(np.square(x), axis=0))),
        }
        self.combo_box = QComboBoxMapping(mapping, current_label="Mean")

        layout.addWidget(self.combo_box)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout.addWidget(self.button_box)

    def get_aggregation_function(self):
        return self.combo_box.value()


class PicksItemWidget(QWidget):
    color_changed_signal = pyqtSignal(QColor)  # Signal to emit when color changes

    def __init__(self, text="", color=QColor(255, 255, 255)):
        super().__init__()

        self.checkbox = QCheckBox(self)
        self.radio_button = QRadioButton(self)
        self.label = QLabel(text, self)

        # Set fixed widths for the checkbox and radiobutton
        checkbox_width = 30
        radiobutton_width = 30
        self.checkbox.setFixedWidth(checkbox_width)
        self.radio_button.setFixedWidth(radiobutton_width)

        self.color_display = QLabel(self)  # Using QLabel to display color
        self.color_display.setFixedSize(20, 20)  # Fixed size for color display
        self.color_display.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
        self.currentColor = color  # Store the current color

        layout = QHBoxLayout()
        layout.addWidget(self.checkbox)
        layout.addWidget(self.radio_button)
        layout.addWidget(self.color_display)
        layout.addWidget(self.label, 1)
        self.setLayout(layout)

        self.radio_button.clicked.connect(self.on_radiobutton_clicked)
        self.color_display.mousePressEvent = self.edit_color

    def change_name(self, name: str) -> None:
        self.label.setText(str(name))

    def get_name(self) -> str:
        return self.label.text()

    def on_radiobutton_clicked(self, checked):
        if checked:  # If the radiobutton is checked, also check the checkbox
            self.checkbox.setChecked(True)

    def edit_color(self, event):
        new_color = QColorDialog.getColor(self.currentColor, self)
        if new_color.isValid():
            self.color_display.setStyleSheet(f"background-color: {new_color.name()}; border: 1px solid black;")
            self.currentColor = new_color
            self.color_changed_signal.emit(new_color)  # Emit signal with the new color


class PropertiesDialog(QDialog):
    def __init__(
        self,
        picks_item_widget: PicksItemWidget,
        picks_mapping: Dict[PicksItemWidget, Picks],
        sgy: SGY,
        sgy_exporter_kwargs: Optional[Dict[str, Any]] = None,
        txt_exporter_kwargs: Optional[Dict[str, Any]] = None,
        json_exporter_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        sgy_exporter_kwargs = sgy_exporter_kwargs or {}
        txt_exporter_kwargs = txt_exporter_kwargs or {}
        json_exporter_kwargs = json_exporter_kwargs or {}

        set_geometry(self, width_rel=0.3, height_rel=0.3, centralize=True, fix_size=False)

        self.picks_item_widget = picks_item_widget
        self.picks_mapping = picks_mapping
        self.sgy = sgy

        self.setWindowTitle("Picks")

        layout = QVBoxLayout(self)

        input_layout = QHBoxLayout()

        self.unit_label = QLabel("Label", self)
        input_layout.addWidget(self.unit_label)

        self.line_edit = QLineEdit(self)
        self.line_edit.setText(picks_item_widget.get_name())
        self.line_edit.textChanged.connect(picks_item_widget.change_name)
        input_layout.addWidget(self.line_edit)

        layout.addLayout(input_layout)

        self.tab_all = QTabWidget(self)
        self.tab_export = QTabWidget()

        picks = self.picks_mapping[self.picks_item_widget]

        self.exports_widgets = {
            "SGY": ExporterSGY(picks=picks, sgy=sgy, **sgy_exporter_kwargs),
            "TXT": ExporterTXT(picks=picks, sgy=sgy, **txt_exporter_kwargs),
            "JSON": ExporterJSON(picks=picks, sgy=sgy, **json_exporter_kwargs),
        }

        for exporter_name, exporter in self.exports_widgets.items():
            self.tab_export.addTab(exporter, exporter_name)

        self.tab_all.addTab(self.tab_export, "Export")
        layout.addWidget(self.tab_all)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout.addWidget(self.button_box)

    def get_sgy_exporter_settings(self):
        return self.exports_widgets["SGY"].get_values()

    def get_txt_exporter_settings(self):
        return self.exports_widgets["TXT"].get_values()

    def get_json_exporter_settings(self):
        return self.exports_widgets["JSON"].get_values()


class ItemsCounter:
    def __init__(self):
        self.constant_values = 0
        self.nn = 0
        self.duplicated = 0
        self.aggregated = 0
        self.loaded = 0


class PicksManager(QWidget):
    picks_updated_signal = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Picks Manager")
        self.setGeometry(100, 100, 300, 400)

        layout = QVBoxLayout()

        self.sgy_exporter_settings = {}
        self.txt_exporter_settings = {}
        self.json_exporter_settings = {}

        self.list_widget = QListWidget(self)
        self.list_widget.itemDoubleClicked.connect(self.open_properties)
        self.list_widget.itemSelectionChanged.connect(self.update_properties_button_state)

        # Enable multi-selection mode
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        self.add_button = QPushButton("+", self)
        # self.add_button.clicked.connect(self.add_item)
        self.add_button.clicked.connect(self.show_add_menu)
        self.remove_button = QPushButton("-", self)
        self.remove_button.clicked.connect(self.remove_items)  # renamed for clarity
        # self.properties_button = QPushButton("\u2699", self)  # Unicode character for gear
        self.properties_button = QPushButton("\U0001F4BE", self)  # "\U0001F4BE" or "\U0001F5AB"
        self.properties_button.setFont(self.font())  # to increase the size of the button a bit
        self.properties_button.clicked.connect(self.open_properties)
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.properties_button)
        layout.addLayout(button_layout)

        self.radio_group = QButtonGroup(self)  # Group for radio buttons
        self.radio_group.buttonToggled.connect(self.update_active_picks)

        self.setLayout(layout)

        self.items_counter = ItemsCounter()
        self.picks_mapping: Dict[PicksItemWidget, Picks] = {}
        self.sgy = None

        self.update_properties_button_state()
        self.hide()

    def duplicate_active_created_by_nn_picks(self):
        for item, picks in self.picks_mapping.items():
            if picks.active and picks.created_by_nn and not picks.modified_manually:
                new_item = self.add_duplicate_pick(item)
                new_item.radio_button.toggle()
                break

    def update_picks_from_external(self, external_picks: Picks) -> None:
        for item, picks in self.picks_mapping.items():
            if picks.id == external_picks.id:
                self.picks_mapping[item] = external_picks

    def set_sgy(self, sgy: SGY):
        self.sgy = sgy

    def reset_manager(self):
        for item in list(self.picks_mapping.keys()):
            self.radio_group.removeButton(item.radio_button)
            self.picks_mapping.pop(item)
            self.items_counter = ItemsCounter()
            self.sgy: Optional[SGY] = None

        self.list_widget.clear()

        self.picks_updated_signal.emit()

    def show_add_menu(self):
        menu = QMenu(self)

        constant_values_action = QAction("Constant Values", self)
        duplicate_action = QAction("Duplicate", self)
        aggregate_action = QAction("Aggregate", self)
        from_headers_action = QAction("Load from Headers", self)

        menu.addAction(constant_values_action)
        menu.addAction(duplicate_action)
        menu.addAction(aggregate_action)
        menu.addAction(from_headers_action)

        constant_values_action.triggered.connect(self.add_constant_values_pick)
        duplicate_action.triggered.connect(self.add_duplicate_pick)
        aggregate_action.triggered.connect(self.add_aggregate_pick)
        from_headers_action.triggered.connect(self.add_from_headers_pick)

        num_selected_items = len(self.list_widget.selectedItems())
        duplicate_action.setEnabled(num_selected_items == 1)
        aggregate_action.setEnabled(num_selected_items >= 2)

        button_pos = self.add_button.mapToGlobal(QPoint(0, 0))
        menu_pos = button_pos + QPoint(0, self.add_button.height())
        menu.exec_(menu_pos)

    def add_constant_values_pick(self):
        assert self.sgy is not None, "Setup SGY"

        dialog = ConstantValuesInputDialog()

        if dialog.exec_() == QDialog.Accepted:
            value = dialog.get_value()
            if value:
                value = int(value)
                picks = Picks(
                    values=[value] * self.sgy.num_traces,
                    unit="mcs",
                    dt_mcs=self.sgy.dt_mcs,
                    created_by_nn=False,
                    picks_color=generate_color(),
                )
                self.items_counter.constant_values += 1
                return self.add_picks(picks, f"Manual {self.items_counter.constant_values}")

    def add_duplicate_pick(self, selected_picks_item: Optional[QListWidgetItem] = None):
        selected_picks_item = selected_picks_item or self.list_widget.itemWidget(self.list_widget.selectedItems()[0])

        picks = self.picks_mapping[selected_picks_item]
        duplicated_picks = picks.create_duplicate(keep_color=False)

        self.items_counter.duplicated += 1
        return self.add_picks(duplicated_picks, f"Duplicated from '{selected_picks_item.get_name()}'")

    def add_aggregate_pick(self):
        self.items_counter.aggregated += 1
        selected_picks_items = [self.list_widget.itemWidget(item) for item in self.list_widget.selectedItems()]

        dialog = AggregationDialog()

        if dialog.exec_() == QDialog.Accepted:
            func = dialog.get_aggregation_function()

            selected_picks = [self.picks_mapping[item] for item in selected_picks_items]
            selected_values = [picks.values for picks in selected_picks]
            aggregated_values = func(selected_values)

            picks = Picks(
                values=aggregated_values,
                unit="mcs",
                dt_mcs=self.sgy.dt_mcs,
                created_by_nn=False,
                picks_color=generate_color(),
            )

            self.items_counter.aggregated += 1
            return self.add_picks(picks, f"Aggregated from {[item.get_name() for item in selected_picks_items]}")

    def add_from_headers_pick(self):
        dialog = QDialogByteEncodeUnit(byte_position=1, first_byte=FIRST_BYTE, encoding="I", picks_unit="mcs")

        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_values()
            values = self.sgy.read_custom_trace_header(
                byte_position=params["byte_position"],
                encoding=params["encoding"],
            )
            picks = Picks(
                values=values,
                unit=params["picks_unit"],
                dt_mcs=self.sgy.dt_mcs,
                created_by_nn=False,
                picks_color=generate_color(),
            )
            self.items_counter.loaded += 1
            return self.add_picks(picks, f"Loaded {self.items_counter.loaded}")

    def add_nn_picks(self, picks: Picks):
        self.items_counter.nn += 1
        return self.add_picks(picks, f"NN {self.items_counter.nn}")

    def add_picks(self, picks: Picks, name: str):
        item = QListWidgetItem()
        self.list_widget.addItem(item)

        picks_item_widget = PicksItemWidget(
            text=name,
            color=QColor(*picks.color),
        )
        self.list_widget.setItemWidget(item, picks_item_widget)
        item.setSizeHint(picks_item_widget.sizeHint())

        picks_item_widget.color_changed_signal.connect(
            lambda color, widget=picks_item_widget: self.update_picks_color(widget, color)
        )
        picks_item_widget.checkbox.clicked.connect(self.picks_updated_signal)
        picks_item_widget.checkbox.setChecked(True)

        self.radio_group.addButton(picks_item_widget.radio_button)

        self.picks_mapping[picks_item_widget] = picks

        self.picks_updated_signal.emit()

        return picks_item_widget

    def get_selected_picks(self):
        seleted_picks = []
        for widget, picks in self.picks_mapping.items():
            if widget.checkbox.isChecked():
                seleted_picks.append(picks)
        return seleted_picks

    def get_active_picks(self):
        for widget, picks in self.picks_mapping.items():
            if widget.radio_button.isChecked():
                return picks
        return None

    def update_active_picks(self, radio_button):
        for widget, picks in self.picks_mapping.items():
            if widget.radio_button is radio_button:
                width = int(DEFAULT_PICKS_WIDTH * 1.7)
                active = True
            else:
                width = DEFAULT_PICKS_WIDTH
                active = False
            picks.width = width
            picks.active = active
        self.picks_updated_signal.emit()

    def remove_items(self):
        for item in self.list_widget.selectedItems():
            picks_item_widget = self.list_widget.itemWidget(item)
            self.radio_group.removeButton(picks_item_widget.radio_button)
            row = self.list_widget.row(item)
            self.list_widget.takeItem(row)

            self.picks_mapping.pop(picks_item_widget)

        self.picks_updated_signal.emit()

    def open_properties(self):
        item = self.list_widget.selectedItems()[0]
        picks_item_widget = self.list_widget.itemWidget(item)
        dialog = PropertiesDialog(
            picks_item_widget,
            self.picks_mapping,
            self.sgy,
            sgy_exporter_kwargs=self.sgy_exporter_settings,
            txt_exporter_kwargs=self.txt_exporter_settings,
            json_exporter_kwargs=self.json_exporter_settings,
        )
        dialog.exec_()
        self.sgy_exporter_settings = dialog.get_sgy_exporter_settings()
        self.txt_exporter_settings = dialog.get_txt_exporter_settings()
        self.json_exporter_settings = dialog.get_json_exporter_settings()

    def update_properties_button_state(self):
        selected_items = self.list_widget.selectedItems()
        self.properties_button.setEnabled(len(selected_items) == 1)

    def update_picks_color(self, picks_item_widget, color):
        pick = self.picks_mapping.get(picks_item_widget)
        if pick:
            pick.color = (color.red(), color.green(), color.blue())
            self.picks_updated_signal.emit()

    def closeEvent(self, e: QCloseEvent) -> None:
        e.ignore()
        self.hide()


if __name__ == "__main__":
    from first_breaks.utils.utils import download_demo_sgy

    sgy_ = SGY(download_demo_sgy())
    picks_ = Picks(values=list(range(sgy_.num_traces)), dt_mcs=sgy_.dt_mcs, unit="mcs")

    app = QApplication(sys.argv)
    window = PicksManager()
    window.set_sgy(sgy_)
    window.add_picks(picks_, "range")
    window.show()
    sys.exit(app.exec_())
