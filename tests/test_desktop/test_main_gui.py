from pathlib import Path

from PyQt5.QtCore import Qt, QObject, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QMenu, QWidget, QAction, QDialog, QPushButton, QDialogButtonBox

from first_breaks.desktop.main_gui import MainWindow
from first_breaks.desktop.picks_manager_widget import PicksManager, ConstantValuesInputDialog
from first_breaks.sgy.reader import SGY


def click_and_get_response(qtbot, qobject: QObject, widget_response_class, find_in_parent=True):
    if hasattr(qobject, "clicked"):
        signal = qobject.clicked
        action = lambda: qtbot.mouseClick(qobject, Qt.LeftButton)
    else:
        signal = qobject.triggered
        action = qobject.trigger

    with qtbot.waitSignal(signal, timeout=1000):
        action()

    # if isinstance(widget, QWidget):
    #     with qtbot.waitSignal(widget.clicked, timeout=1000):
    #         qtbot.mouseClick(widget, Qt.LeftButton)
    # elif isinstance(widget, QAction):
    #     with qtbot.waitSignal(widget.triggered, timeout=1000):
    #         widget.trigger()
    # else:
    #     raise TypeError("")

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

    picks_manager = main_window.picks_manager

    # Find the QMenu
    add_menu: QMenu = click_and_get_response(qtbot, main_window.picks_manager.add_button, QMenu)

    # Find the specific QAction
    constant_values_action = None
    for action in add_menu.actions():
        if action.text() == picks_manager.ADD_PICKS_NAME_CONSTANT_VALUES:
            constant_values_action = action
            break

    assert constant_values_action is not None

    def get_child():
        ww = add_menu.findChildren(QObject)
        assert False, [w.text() for w in ww]
        return w

    QTimer.singleShot(1000, get_child)

    constant_values_action.trigger()



    # add_menu: QMenu = click_and_get_response(qtbot, main_window.picks_manager.add_button, QMenu)
    # assert_add_picks_options_enabled(add_menu, aggregate=False, duplicate=False)
    # for action in add_menu.actions():
    #     if action.text() == PicksManager.ADD_PICKS_NAME_CONSTANT_VALUES:
    #         action.trigger()
    #         input_const_picks_menu = main_window.picks_manager.findChild(ConstantValuesInputDialog)
    #         # assert isinstance(input_const_picks_menu, ConstantValuesInputDialog)
    #
    #         QTest.keyClicks(widget=input_const_picks_menu, sequence="42")
    #         # QTest.mouseClick(input_const_picks_menu.buttonBox.button(QDialog.Accepted), Qt.LeftButton)
    #         # input_const_picks_menu: ConstantValuesInputDialog = click_and_get_response(
    #         #     qtbot, action, QDialog, find_in_parent=True
    #         # )
    #         # print(input_const_picks_menu)
    #         # qtbot.keyClicks(input_const_picks_menu, "100_000")
    #         # qtbot.mouseClick(input_const_picks_menu.okButton, Qt.LeftButton)
