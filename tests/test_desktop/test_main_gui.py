from pathlib import Path
from typing import Optional, Type

from PyQt5.QtCore import Qt, QObject, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QMenu, QWidget, QAction, QDialog, QPushButton, QDialogButtonBox, QApplication

from first_breaks.desktop.main_gui import MainWindow
from first_breaks.desktop.picks_manager_widget import PicksManager, ConstantValuesInputDialog
from first_breaks.sgy.reader import SGY


def find_global(widget_class: Type[QWidget]) -> QWidget:
    found_widgets = []
    app = QApplication.instance()
    for widget in app.topLevelWidgets():
        if isinstance(widget, widget_class):
            found_widgets.append(widget)

    if len(found_widgets) == 0:
        raise RuntimeError("Widget not found")
    elif len(found_widgets) > 1:
        raise RuntimeError("Widget is not unique")
    else:
        return found_widgets[0]


def click_and_get_response(qtbot, qobject: QObject, widget_response_class: Type[QWidget], find_in_parent=True):
    if hasattr(qobject, "clicked"):
        signal = qobject.clicked
        action = lambda: qtbot.mouseClick(qobject, Qt.LeftButton)
    else:
        signal = qobject.triggered
        action = qobject.trigger

    with qtbot.waitSignal(signal, timeout=1000):
        action()

    where_to_find = qobject.parent() if find_in_parent else qobject
    response_widget = where_to_find.findChild(widget_response_class)
    assert isinstance(response_widget, widget_response_class)
    return response_widget


def assert_add_picks_options_enabled(widget: QMenu, aggregate: bool, duplicate: bool):
    for action in widget.actions():
        if action.text() == PicksManager.ADD_PICKS_NAME_AGGREGATE:
            assert action.isEnabled() == aggregate
        elif action.text() == PicksManager.ADD_PICKS_NAME_DUPLICATE:
            assert action.isEnabled() == duplicate
        else:
            assert action.isEnabled()  # others are lways enabled


def enter_const_picks_mcs(qtbot, mcs: int):
    enter_dialog: ConstantValuesInputDialog = find_global(ConstantValuesInputDialog)
    line_edit = enter_dialog.line_edit
    qtbot.keyClicks(line_edit, str(mcs))
    button_box = enter_dialog.findChild(QDialogButtonBox)
    assert button_box is not None
    ok_button = button_box.button(QDialogButtonBox.Ok)
    assert ok_button is not None
    qtbot.mouseClick(ok_button, Qt.LeftButton)


def test_main_gui(qtbot, demo_sgy: Path, model_onnx: Path):
    main_window = MainWindow(show=False)
    qtbot.addWidget(main_window)
    sgy = SGY(demo_sgy)

    # init state of app
    assert main_window.button_load_nn.isEnabled()
    assert main_window.button_get_filename.isEnabled()
    assert not main_window.button_settings_processing.isEnabled()
    assert not main_window.button_picks_manager.isEnabled()

    # load sgy programmatically
    main_window.get_filename(demo_sgy)

    assert main_window.button_load_nn.isEnabled()
    assert main_window.button_get_filename.isEnabled()
    assert main_window.button_settings_processing.isEnabled()
    assert main_window.button_picks_manager.isEnabled()
    assert not main_window.settings_processing_widget.run_button.isEnabled()

    # load nn programmatically
    main_window.load_nn(model_onnx)

    assert not main_window.button_load_nn.isEnabled()
    assert main_window.button_get_filename.isEnabled()
    assert main_window.button_settings_processing.isEnabled()
    assert main_window.button_picks_manager.isEnabled()
    assert main_window.settings_processing_widget.run_button.isEnabled()

    # check that traces are plotted
    assert main_window.graph.traces_as_items
    x_lim_graph = main_window.graph.getViewBox().state["limits"]["xLimits"]
    y_lim_graph = main_window.graph.getViewBox().state["limits"]["yLimits"]
    assert x_lim_graph[0] == 0
    assert y_lim_graph[0] == 0
    assert x_lim_graph[1] == sgy.num_traces + 1  # extra space after last trace
    assert y_lim_graph[1] == (sgy.num_samples - 1) * sgy.dt_ms

    # add const picks
    assert not main_window.picks_manager.picks_mapping

    add_menu = click_and_get_response(qtbot, main_window.picks_manager.add_button, QMenu)
    assert_add_picks_options_enabled(add_menu, aggregate=False, duplicate=False)

    mcs_const = 100_000

    for action in add_menu.actions():
        if action.text() == PicksManager.ADD_PICKS_NAME_CONSTANT_VALUES:
            QTimer.singleShot(200, lambda: enter_const_picks_mcs(qtbot, mcs_const))
            action.trigger()

    assert len(main_window.picks_manager.picks_mapping) == 1

    const_picks_item_widget = list(main_window.picks_manager.picks_mapping.keys())[0]
    const_picks = main_window.picks_manager.picks_mapping[const_picks_item_widget]
    assert all(v == mcs_const for v in const_picks.values)
    assert const_picks.unit == "mcs"
    assert len(const_picks) == sgy.num_traces
    assert const_picks.dt_mcs == sgy.dt_mcs
    assert len(main_window.graph.picks2items) == 1
    assert not const_picks.active
    assert not const_picks_item_widget.radio_button.isChecked()

    const_picks_item_widget_rgb = (
        const_picks_item_widget.currentColor.red(),
        const_picks_item_widget.currentColor.green(),
        const_picks_item_widget.currentColor.blue(),
    )
    assert const_picks_item_widget_rgb == tuple(const_picks.color)
