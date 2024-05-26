import time
from pathlib import Path
from typing import Optional, Type, Tuple, Literal, Union

from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt, QObject, QTimer
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QMenu, QWidget, QAction, QDialog, QPushButton, QDialogButtonBox, QApplication, QColorDialog
import numpy as np
from pytestqt.qtbot import QtBot

from first_breaks.desktop.main_gui import MainWindow
from first_breaks.desktop.picks_manager_widget import (
    PicksManager,
    ConstantValuesInputDialog,
    ACTIVE_PICKS_WIDTH,
    PicksItemWidget,
)
from first_breaks.desktop.settings_processing_widget import GainLine, MaximumTimeLine, DeviceLine
from first_breaks.picking.picks import DEFAULT_PICKS_WIDTH, Picks
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


def click_and_get_response(qtbot: QtBot, qobject: QObject, widget_response_class: Type[QWidget], find_in_parent=True):
    if hasattr(qobject, "clicked"):
        signal = qobject.clicked
        action = lambda: qtbot.mouseClick(qobject, Qt.LeftButton)
    else:
        signal = qobject.triggered
        action = qobject.trigger

    print("before with")
    with qtbot.waitSignal(signal, timeout=1000):
        action()
    print("after with")

    where_to_find = qobject.parent() if find_in_parent else qobject
    response_widget = where_to_find.findChild(widget_response_class)
    print(where_to_find.findChildren(widget_response_class), "all children")
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


# def enter_const_picks_mcs(qtbot: QtBot, mcs: int):
#     enter_dialog: ConstantValuesInputDialog = find_global(ConstantValuesInputDialog)
#     print(enter_dialog, "enter dialog")
#     line_edit = enter_dialog.line_edit
#     qtbot.keyClicks(line_edit, str(mcs))
#     press_ok(qtbot, enter_dialog)
#     enter_dialog.close()
#     qtbot.wait(100)
#     del enter_dialog
#     # enter_dialog.deleteLater()
#     # qtbot.wait(100)


# def enter_const_picks_mcs(qtbot: QtBot, mcs: int, picks_manager: PicksManager):
#     enter_dialog = picks_manager._const_values_widget
#     line_edit = enter_dialog.line_edit
#     qtbot.keyClicks(line_edit, str(mcs))
#     click_ok(qtbot, enter_dialog)


def change_color(qtbot: QtBot, new_color):
    color_dialog: QColorDialog = find_global(QColorDialog)
    color_dialog.setCurrentColor(new_color)
    click_ok(qtbot, color_dialog)


def qcolor2rgb(qcolor: QColor) -> Tuple[int, int, int]:
    return qcolor.red(), qcolor.green(), qcolor.blue()


def rgb2qcolor(rgb: Tuple[int, int, int]) -> QColor:
    return QColor(*rgb)


def click_ok(qtbot: QtBot, widget: QWidget):
    button_box = widget.findChild(QDialogButtonBox)
    assert button_box is not None
    ok_button = button_box.button(QDialogButtonBox.Ok)
    assert ok_button is not None
    click(qtbot, ok_button)


def click(qtbot: QtBot, widget: QWidget):
    with qtbot.waitSignal(widget.clicked, timeout=1000):
        qtbot.mouseClick(widget, Qt.LeftButton)


def enter_const_picks_mcs(qtbot: QtBot, mcs: int, picks_manager: PicksManager):
    enter_dialog = picks_manager._const_values_widget
    line_edit = enter_dialog.line_edit
    qtbot.keyClicks(line_edit, str(mcs))
    click_ok(qtbot, enter_dialog)


def __select_from_menu_and_enter_const_picks_mcs(qtbot: QtBot, mcs: int, main_window: MainWindow):
    add_menu = main_window.picks_manager._add_menu_widget
    assert_add_picks_options_enabled(add_menu, aggregate=False, duplicate=False)
    clicked = False
    for action in add_menu.actions():
        if action.text() == PicksManager.ADD_PICKS_NAME_CONSTANT_VALUES:
            QTimer.singleShot(200, lambda: enter_const_picks_mcs(qtbot, mcs, main_window.picks_manager))
            action.trigger()
            clicked = True
            break
    assert clicked


def select_from_menu_and_enter_const_picks_mcs(qtbot: QtBot, mcs: int, main_window: MainWindow):
    QTimer.singleShot(200, lambda: __select_from_menu_and_enter_const_picks_mcs(qtbot, mcs, main_window))
    click(qtbot, main_window.picks_manager.add_button)


def aggregate_mean(qtbot, picks_manager: PicksManager):
    aggregate_dialog = picks_manager._aggregation_widget
    aggregate_dialog.combo_box.setCurrentIndex(aggregate_dialog.MEAN_INDEX)
    click_ok(qtbot, aggregate_dialog)


def __select_from_menu_and_aggregate_mean(qtbot: QtBot, main_window: MainWindow):
    add_menu = main_window.picks_manager._add_menu_widget
    assert_add_picks_options_enabled(add_menu, aggregate=True, duplicate=False)
    clicked = False
    for action in add_menu.actions():
        if action.text() == PicksManager.ADD_PICKS_NAME_AGGREGATE:
            QTimer.singleShot(200, lambda: aggregate_mean(qtbot, main_window.picks_manager))
            action.trigger()
            clicked = True
            break
    assert clicked


def select_from_menu_and_aggregate_mean(qtbot: QtBot, main_window: MainWindow):
    QTimer.singleShot(200, lambda: __select_from_menu_and_aggregate_mean(qtbot, main_window))
    click(qtbot, main_window.picks_manager.add_button)


def __select_from_menu_and_duplicate(main_window: MainWindow):
    add_menu = main_window.picks_manager._add_menu_widget
    assert_add_picks_options_enabled(add_menu, aggregate=False, duplicate=True)
    clicked = False
    for action in add_menu.actions():
        if action.text() == PicksManager.ADD_PICKS_NAME_DUPLICATE:
            action.trigger()
            clicked = True
            break
    assert clicked


def select_from_menu_and_duplicate(qtbot: QtBot, main_window: MainWindow):
    QTimer.singleShot(200, lambda: __select_from_menu_and_duplicate(main_window))
    click(qtbot, main_window.picks_manager.add_button)


def __set_params(qtbot: QtBot, main_window: MainWindow, gain: float, maximum_time: float):
    settings_widget = main_window.settings_processing_widget

    gain_widget = None
    maximum_time_widget = None
    device_widget = None

    for label, widget, line, is_for_plotseis in settings_widget._inputs:
        if isinstance(widget, GainLine):
            gain_widget = widget
        elif isinstance(widget, MaximumTimeLine):
            maximum_time_widget = widget
        elif isinstance(widget, DeviceLine):
            device_widget = widget

    assert gain_widget is not None
    assert maximum_time_widget is not None
    assert device_widget is not None

    device_widget.setFocus()
    device_widget.select_cpu()

    maximum_time_widget.setFocus()
    qtbot.keyClick(maximum_time_widget, "a", modifier=Qt.ControlModifier)
    qtbot.keyClick(maximum_time_widget, Qt.Key_Delete)
    qtbot.keyClicks(maximum_time_widget, str(maximum_time))

    gain_widget.setFocus()
    qtbot.keyClick(gain_widget, "a", modifier=Qt.ControlModifier)
    qtbot.keyClick(gain_widget, Qt.Key_Delete)
    qtbot.keyClicks(gain_widget, str(gain))


def set_params_and_run_processing(qtbot: QtBot, main_window: MainWindow, gain: float, maximum_time: float):
    clicked = False
    for action in main_window.toolbar.actions():
        if action.text() == main_window.TOOLBAR_SETTINGS_AND_PROCESSINGS:
            QTimer.singleShot(200, lambda: __set_params(qtbot, main_window, gain, maximum_time))
            action.trigger()
            clicked = True
            break
    assert clicked

    qtbot.waitUntil(lambda: main_window.plotseis_settings.gain == gain, timeout=1000)

    with qtbot.waitSignal(main_window.processing_finished_signal, timeout=10000):
        QTimer.singleShot(50, main_window.settings_processing_widget.close)
        click(qtbot, main_window.settings_processing_widget.run_button)


def get_picks_widget_from_picks_manager(picks_manager: PicksManager, line_idx: int) -> PicksItemWidget:
    item_in_list = picks_manager.list_widget.item(line_idx)
    picks_item_widget = picks_manager.list_widget.itemWidget(item_in_list)
    assert list(picks_manager.picks_mapping.keys())[line_idx] == picks_item_widget
    return picks_item_widget


def get_picks_from_picks_manager(picks_manager: PicksManager, line_idx: int) -> Picks:
    picks_item_widget = get_picks_widget_from_picks_manager(picks_manager, line_idx)
    return picks_manager.picks_mapping[picks_item_widget]


def asserts_set_for_line_in_picks_manager(
    main_window: MainWindow,
    sgy: SGY,
    total_picks: int,
    line_idx: int,
    units: Optional[str],
    is_active_picks: bool,
    selected_to_show: bool,
) -> None:
    assert main_window.picks_manager.list_widget.count() == total_picks

    picks_item_widget = get_picks_widget_from_picks_manager(main_window.picks_manager, line_idx)
    picks = main_window.picks_manager.picks_mapping[picks_item_widget]

    if units is not None:
        assert picks.unit == units
    assert len(picks) == sgy.num_traces
    assert picks.dt_mcs == sgy.dt_mcs
    assert len(main_window.graph.picks2items) == len(main_window.picks_manager.get_selected_picks())

    assert bool(picks.active) == is_active_picks
    assert picks_item_widget.radio_button.isChecked() == is_active_picks
    width = ACTIVE_PICKS_WIDTH if is_active_picks else DEFAULT_PICKS_WIDTH
    assert picks.width == width

    picks_item_widget_rgb = qcolor2rgb(picks_item_widget.currentColor)
    assert picks_item_widget_rgb == tuple(picks.color)

    if selected_to_show:
        assert picks in main_window.graph.picks2items
        const_picks_graph_rgb = qcolor2rgb(main_window.graph.picks2items[picks].opts["pen"].color())
        assert const_picks_graph_rgb == tuple(picks.color)
    else:
        assert picks not in main_window.graph.picks2items


def test_main_gui(qtbot: QtBot, demo_sgy: Path, model_onnx: Path):
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
    qtbot.waitUntil(lambda: main_window.sgy is not None, timeout=1000)

    assert main_window.button_load_nn.isEnabled()
    assert main_window.button_get_filename.isEnabled()
    assert main_window.button_settings_processing.isEnabled()
    assert main_window.button_picks_manager.isEnabled()
    assert not main_window.settings_processing_widget.run_button.isEnabled()

    # load nn programmatically
    main_window.load_nn(model_onnx)
    qtbot.waitUntil(lambda: main_window.nn_manager.picker is not None, timeout=10000)

    assert not main_window.button_load_nn.isEnabled()
    assert main_window.button_get_filename.isEnabled()
    assert main_window.button_settings_processing.isEnabled()
    assert main_window.button_picks_manager.isEnabled()
    assert main_window.settings_processing_widget.run_button.isEnabled()

    main_window.get_filename(demo_sgy)
    qtbot.waitUntil(lambda: main_window.sgy is not None, timeout=1000)

    # check that traces are plotted
    assert main_window.graph.traces_as_items
    x_lim_graph = main_window.graph.getViewBox().state["limits"]["xLimits"]
    y_lim_graph = main_window.graph.getViewBox().state["limits"]["yLimits"]
    assert x_lim_graph[0] == 0
    assert y_lim_graph[0] == 0
    assert x_lim_graph[1] == sgy.num_traces + 1  # extra space after last trace
    assert y_lim_graph[1] == (sgy.num_samples - 1) * sgy.dt_ms

    # add picks
    nn_picks_ids = []
    const_picks_ids = []
    total_picks = 0
    assert not main_window.picks_manager.picks_mapping

    # NN picks
    # for gain, max_time in [
    #     (2, 50),
    #     # (1.1, 100),
    # ]:
    #     nn_picks_ids.append(total_picks)
    #     total_picks += 1
    #     set_params_and_run_processing(qtbot, main_window, gain, max_time)
    #
    #     qtbot.waitUntil(lambda: len(main_window.picks_manager.picks_mapping) == total_picks, timeout=2000)
    #
    #     idx_last_added_picks = main_window.picks_manager.list_widget.count() - 1
    #     asserts_set_for_line_in_picks_manager(
    #         main_window=main_window,
    #         sgy=sgy,
    #         total_picks=total_picks,
    #         line_idx=idx_last_added_picks,
    #         units="sample",
    #         is_active_picks=False,
    #         selected_to_show=True,
    #     )
    #
    #     picks = get_picks_from_picks_manager(main_window.picks_manager, idx_last_added_picks)
    #     assert picks.created_by_nn
    #     assert picks.picking_parameters.gain == gain
    #     assert picks.picking_parameters.maximum_time == max_time

    # const picks
    for idx in range(3):
        const_picks_ids.append(total_picks)
        total_picks += 1
        mcs_const = idx * 10000

        select_from_menu_and_enter_const_picks_mcs(qtbot, mcs_const, main_window)

        qtbot.waitUntil(lambda: len(main_window.picks_manager.picks_mapping) == total_picks, timeout=2000)

        idx_last_added_picks = main_window.picks_manager.list_widget.count() - 1
        asserts_set_for_line_in_picks_manager(
            main_window=main_window,
            sgy=sgy,
            total_picks=total_picks,
            line_idx=idx_last_added_picks,
            units="mcs",
            is_active_picks=False,
            selected_to_show=True,
        )

        picks = get_picks_from_picks_manager(main_window.picks_manager, idx_last_added_picks)
        assert not picks.created_by_nn
        assert all(v == mcs_const for v in picks.values)

    # duplicate picks
    main_window.picks_manager.list_widget.clearSelection()
    for idx in [
        # nn_picks_ids[0],
        const_picks_ids[0],
    ]:
        main_window.picks_manager.list_widget.item(idx).setSelected(True)
        selected_picks = get_picks_from_picks_manager(main_window.picks_manager, idx)

        select_from_menu_and_duplicate(qtbot, main_window)
        total_picks += 1
        qtbot.waitUntil(lambda: len(main_window.picks_manager.picks_mapping) == total_picks, timeout=2000)

        asserts_set_for_line_in_picks_manager(
            main_window=main_window,
            sgy=sgy,
            total_picks=total_picks,
            line_idx=total_picks - 1,
            units=selected_picks.unit,
            is_active_picks=False,
            selected_to_show=True,
        )
        duplicated_picks = get_picks_from_picks_manager(main_window.picks_manager, total_picks - 1)

        assert np.all(selected_picks.picks_in_mcs == duplicated_picks.picks_in_mcs)
        assert duplicated_picks.created_by_nn == selected_picks.created_by_nn

    # aggregate picks
    main_window.picks_manager.list_widget.clearSelection()
    picks_to_aggregate = []
    for idx in nn_picks_ids + const_picks_ids:
        main_window.picks_manager.list_widget.item(idx).setSelected(True)
        picks_to_aggregate.append(get_picks_from_picks_manager(main_window.picks_manager, idx))

    select_from_menu_and_aggregate_mean(qtbot, main_window)
    total_picks += 1
    qtbot.waitUntil(lambda: len(main_window.picks_manager.picks_mapping) == total_picks, timeout=2000)

    asserts_set_for_line_in_picks_manager(
        main_window=main_window,
        sgy=sgy,
        total_picks=total_picks,
        line_idx=total_picks - 1,
        units="mcs",
        is_active_picks=False,
        selected_to_show=True,
    )

    picks_aggregated = get_picks_from_picks_manager(main_window.picks_manager, total_picks - 1)

    picks_aggregated_values = picks_aggregated.picks_in_mcs
    picks_expected_values = np.mean([p.values for p in picks_to_aggregate], axis=0).astype(int)
    assert np.all(picks_expected_values == picks_aggregated_values)

    assert not picks_aggregated.created_by_nn

    # # manupulate with picks
    # for idx in range(total_picks):
    #     picks_item_widget = get_picks_widget_from_picks_manager(main_window.picks_manager, idx)
    #     picks = get_picks_from_picks_manager(main_window.picks_manager, idx)
    #
    #     # change colors
    #     new_rgb = tuple(int(np.random.randint(0, 255)) for _ in range(3))
    #     new_color = QColor(*new_rgb)
    #     QTimer.singleShot(200, lambda: change_color(qtbot, new_color))
    #     qtbot.mouseClick(picks_item_widget.color_display, Qt.LeftButton)
    #
    #     const_picks_item_widget_rgb = qcolor2rgb(picks_item_widget.currentColor)
    #     assert const_picks_item_widget_rgb == tuple(picks.color) == new_rgb
    #
    #     const_picks_graph_rgb = qcolor2rgb(main_window.graph.picks2items[picks].opts["pen"].color())
    #     assert const_picks_graph_rgb == tuple(picks.color) == new_rgb
    #
    #     asserts_set_for_line_in_picks_manager(
    #         main_window=main_window,
    #         sgy=sgy,
    #         total_picks=total_picks,
    #         line_idx=idx,
    #         units=None,
    #         is_active_picks=False,
    #         selected_to_show=True,
    #     )
    #
    #     # make active
    #     with qtbot.waitSignal(picks_item_widget.radio_button.toggled, timeout=1000):
    #         picks_item_widget.radio_button.toggle()
    #
    #     asserts_set_for_line_in_picks_manager(
    #         main_window=main_window,
    #         sgy=sgy,
    #         total_picks=total_picks,
    #         line_idx=idx,
    #         units=None,
    #         is_active_picks=True,
    #         selected_to_show=True,
    #     )
    #
    #     # hide from graph
    #     with qtbot.waitSignal(main_window.picks_manager.picks_updated_signal, timeout=1000):
    #         picks_item_widget.checkbox.click()
    #
    #     asserts_set_for_line_in_picks_manager(
    #         main_window=main_window,
    #         sgy=sgy,
    #         total_picks=total_picks,
    #         line_idx=idx,
    #         units=None,
    #         is_active_picks=True,
    #         selected_to_show=False,
    #     )
    #
    #     # show again
    #     with qtbot.waitSignal(main_window.picks_manager.picks_updated_signal, timeout=1000):
    #         picks_item_widget.checkbox.click()
    #
    #     asserts_set_for_line_in_picks_manager(
    #         main_window=main_window,
    #         sgy=sgy,
    #         total_picks=total_picks,
    #         line_idx=idx,
    #         units=None,
    #         is_active_picks=True,
    #         selected_to_show=True,
    #     )
