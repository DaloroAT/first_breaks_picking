import sys
from typing import Union, Sequence, Optional, Callable, Any, Tuple

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QLabel, QFrame

from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt, QTimer, QPoint, QEasingCurve, QPropertyAnimation, pyqtSignal, QObject, pyqtBoundSignal


# class GeneralToolTip(QLabel):
#     def __init__(self, widget, text=None, highlight=False, shake=False, duration=1000):
#         super().__init__(None)  # No parent to be a top-level window
#
#         # ToolTip Settings
#         self.setFrameShape(QFrame.StyledPanel)
#         self.setStyleSheet("""
#             background-color: yellow;
#             border: 1px solid black;
#             padding: 2px;
#         """)
#         self.setAutoFillBackground(True)
#         self.setWindowFlags(Qt.ToolTip | Qt.WindowStaysOnTopHint)
#
#         # If there's a message, show it.
#         if text:
#             self.setText(text)
#             global_pos = widget.mapToGlobal(QPoint(0, 0))
#             self.move(global_pos.x() + widget.width(), global_pos.y())
#             self.show()
#
#         # Highlighting the target widget
#         if highlight:
#             self.original_palette = widget.palette()
#             palette = widget.palette()
#             palette.setColor(QPalette.Base, QColor(255, 220, 220))  # Light red
#             widget.setPalette(palette)
#
#             QTimer.singleShot(duration, lambda: widget.setPalette(self.original_palette))
#
#         # Shaking the target widget
#         if shake:
#             self.animation = QPropertyAnimation(widget, b"pos")
#             original_pos = widget.pos()
#             self.animation.setStartValue(original_pos)
#             self.animation.setEndValue(original_pos)
#             num_shakes = 20
#             for i in range(num_shakes):
#                 self.animation.setKeyValueAt((i + 1) / num_shakes, original_pos + (-1) ** i * QPoint(1, 0))
#             self.animation.setDuration(duration)
#
#             self.animation.setEasingCurve(QEasingCurve.InOutQuad)
#             self.animation.start()
#
#         QTimer.singleShot(duration, self.close)


class QErrorToolTip(QLabel):
    index_changed_signal = pyqtSignal()

    def __init__(self,
                 widget,
                 text=None,
                 text_position="top",
                 highlight_widgets=None,
                 shake_widgets=None,
                 text_duration=1000,
                 shake_duration=1000,
                 highlight_duration=1000,
                 show_until_event=None):

        super().__init__(widget)
        text_duration = max(1, int(text_duration))
        shake_duration = max(1, int(shake_duration))
        highlight_duration = max(1, int(highlight_duration))

        self._original_palettes = {}
        self._shake_animations = []

        if text:
            self.show_text(widget, text, text_position, text_duration, show_until_event)

        if highlight_widgets:
            for w in highlight_widgets:
                self.highlight_widget(w, highlight_duration)

        if shake_widgets:
            for w in shake_widgets:
                self.shake_widget(w, shake_duration)

        if show_until_event:
            if not isinstance(show_until_event, (list, tuple)):
                show_until_event = [show_until_event]
            for ev in show_until_event:
                ev.connect(self.close)
        else:
            max_duration = int(1.2 * max(text_duration, shake_duration, highlight_duration))
            QTimer.singleShot(max_duration, self.close)

    def show_text(self, widget, text, text_position, text_duration, show_until_event):
        self.setWordWrap(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            background-color: yellow;
            border: 1px solid black;
            padding: 2px;
        """)
        self.setAutoFillBackground(True)
        self.setWindowFlags(Qt.ToolTip | Qt.WindowStaysOnTopHint)

        self.setText(text)
        global_pos = widget.mapToGlobal(QPoint(0, 0))
        if text_position == "top":
            self.move(global_pos.x(), global_pos.y() - self.height())
        elif text_position == "bottom":
            self.move(global_pos.x(), global_pos.y() + widget.height() + 10)
        elif text_position == "left":
            self.move(global_pos.x() - self.width() - widget.width(), global_pos.y())
        elif text_position == "right":
            self.move(global_pos.x() + widget.width() + 10, global_pos.y())
        else:
            raise ValueError("Wrong position value")
        self.show()

        if show_until_event:
            pass
        else:
            QTimer.singleShot(text_duration, self.hide)

    def shake_widget(self, widget, shake_duration):
        shakes_per_second = 20
        animation = QPropertyAnimation(widget, b"pos")
        self._shake_animations.append(animation)
        original_pos = widget.pos()
        animation.setStartValue(original_pos)
        animation.setEndValue(original_pos)
        num_shakes = max(1, int(shakes_per_second * shake_duration / 1000))
        for i in range(num_shakes):
            animation.setKeyValueAt((i + 1) / num_shakes, original_pos + (-1) ** i * QPoint(1, 0))
        animation.setDuration(shake_duration)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        animation.start()

    def highlight_widget(self, widget, highlight_duration):
        self._original_palettes[widget] = widget.palette()
        palette = widget.palette()
        palette.setColor(QPalette.Base, QColor(255, 220, 220))  # Light red
        widget.setPalette(palette)

        QTimer.singleShot(highlight_duration, lambda: widget.setPalette(self._original_palettes[widget]))


# class QErrorToolTip(QLabel):
#     index_changed_signal = pyqtSignal()
#
#     def __init__(self,
#                  widget: QWidget,
#                  text: Optional[str] = None,
#                  text_position: str = "top",
#                  text_until: Union[float, pyqtSignal, Sequence[pyqtSignal]] = 1000,
#                  highlight_widgets: Optional[Union[QWidget, Sequence[QWidget]]] = None,
#                  highlight_until: Union[float, pyqtSignal, Sequence[pyqtSignal]] = 1000,
#                  shake_widgets: Optional[Union[QWidget, Sequence[QWidget]]] = None,
#                  shake_duration: Union[float, pyqtSignal, Sequence[pyqtSignal]] = 1000,
#                  ):
#
#         super().__init__(widget)
#         text_duration = max(1, int(text_duration))
#         shake_duration = max(1, int(shake_duration))
#         highlight_duration = max(1, int(highlight_duration))
#
#         self._original_palettes = {}
#         self._shake_animations = []
#
#         if text:
#             self.show_text(widget, text, text_position, text_duration, show_until_event)
#
#         if highlight_widgets:
#             for w in highlight_widgets:
#                 self.highlight_widget(w, highlight_duration)
#
#         if shake_widgets:
#             for w in shake_widgets:
#                 self.shake_widget(w, shake_duration)
#
#         if show_until_event:
#             if not isinstance(show_until_event, (list, tuple)):
#                 show_until_event = [show_until_event]
#             for ev in show_until_event:
#                 ev.connect(self.close)
#         else:
#             max_duration = int(1.2 * max(text_duration, shake_duration, highlight_duration))
#             QTimer.singleShot(max_duration, self.close)
#
#     def show_text(self, widget, text, text_position, text_duration, show_until_event):
#         self.setWordWrap(True)
#         self.setFrameShape(QFrame.StyledPanel)
#         self.setStyleSheet("""
#             background-color: yellow;
#             border: 1px solid black;
#             padding: 2px;
#         """)
#         self.setAutoFillBackground(True)
#         self.setWindowFlags(Qt.ToolTip | Qt.WindowStaysOnTopHint)
#
#         self.setText(text)
#         global_pos = widget.mapToGlobal(QPoint(0, 0))
#         if text_position == "top":
#             self.move(global_pos.x(), global_pos.y() - self.height())
#         elif text_position == "bottom":
#             self.move(global_pos.x(), global_pos.y() + widget.height() + 10)
#         elif text_position == "left":
#             self.move(global_pos.x() - self.width() - widget.width(), global_pos.y())
#         elif text_position == "right":
#             self.move(global_pos.x() + widget.width() + 10, global_pos.y())
#         else:
#             raise ValueError("Wrong position value")
#         self.show()
#
#         if show_until_event:
#             pass
#         else:
#             QTimer.singleShot(text_duration, self.hide)
#
#     def shake_widget(self, widget, shake_duration):
#         shakes_per_second = 20
#         animation = QPropertyAnimation(widget, b"pos")
#         self._shake_animations.append(animation)
#         original_pos = widget.pos()
#         animation.setStartValue(original_pos)
#         animation.setEndValue(original_pos)
#         num_shakes = max(1, int(shakes_per_second * shake_duration / 1000))
#         for i in range(num_shakes):
#             animation.setKeyValueAt((i + 1) / num_shakes, original_pos + (-1) ** i * QPoint(1, 0))
#         animation.setDuration(shake_duration)
#         animation.setEasingCurve(QEasingCurve.InOutQuad)
#         animation.start()
#
#     def highlight_widget(self, widget, highlight_duration):
#         self._original_palettes[widget] = widget.palette()
#         palette = widget.palette()
#         palette.setColor(QPalette.Base, QColor(255, 220, 220))  # Light red
#         widget.setPalette(palette)
#
#         QTimer.singleShot(highlight_duration, lambda: widget.setPalette(self._original_palettes[widget]))


TDoUntil = Union[int, float, pyqtSignal, Sequence[pyqtSignal]]


def run_do_until(do_until: TDoUntil, func: Callable[[Any], Any]) -> None:
    if isinstance(do_until, (int, float)):
        timeout = max(1, int(do_until))
        QTimer.singleShot(timeout, func)
        return
    elif isinstance(do_until, (pyqtSignal, pyqtBoundSignal)):
        do_until = [do_until]
    for signal in do_until:
        signal.connect(func)


class TextToolTip(QLabel):
    def __init__(self,
                 widget: QWidget,
                 text: str,
                 text_position="right",
                 color: Union[Tuple[int, int, int], str] = "#faf3be",
                 do_until: TDoUntil = 5000):
        super().__init__(widget)

        self.setWordWrap(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
                    background-color: {color};
                    border: 1px solid black;
                    padding: 2px;
                """)
        self.setAutoFillBackground(True)
        self.setWindowFlags(Qt.ToolTip | Qt.WindowStaysOnTopHint)

        self.setText(text)
        self.adjustSize()

        global_pos = widget.mapToGlobal(QPoint(0, 0))
        if text_position == "top":
            self.move(global_pos.x(), global_pos.y() - self.height() - 5)
        elif text_position == "bottom":
            self.move(global_pos.x(), global_pos.y() + widget.height() + 5)
        elif text_position == "left":
            self.move(global_pos.x() - self.width() - 5, global_pos.y())
        elif text_position == "right":
            self.move(global_pos.x() + widget.width() + 5, global_pos.y())
        else:
            raise ValueError("Wrong position value")
        self.show()

        run_do_until(do_until, self.close)


class HighlightToolTip(QObject):
    _all_highlight_tips = []

    def __init__(self,
                 widgets: Optional[Union[QWidget, Sequence[QWidget]]] = None,
                 do_until: TDoUntil = 5000):
        super().__init__()

        widgets = widgets if isinstance(widgets, (list, tuple)) else [widgets]
        self._original_palettes = {}
        for widget in widgets:
            self._original_palettes[widget] = widget.palette()
            palette = widget.palette()
            palette.setColor(QPalette.Base, QColor(255, 220, 220))  # Light red
            widget.setPalette(palette)
            # widget._highlight_tooltip_auto_attribute = self

        HighlightToolTip._all_highlight_tips.append(self)

        run_do_until(do_until, self.remove_highlight)

        print(HighlightToolTip._all_highlight_tips)

    def remove_highlight(self):
        for widget, original_color in self._original_palettes.items():
            widget.setPalette(self._original_palettes[widget])

        HighlightToolTip._all_highlight_tips.remove(self)


if __name__ == "__main__":
    class ValidatorWidget(QWidget):
        def __init__(self):
            super().__init__()

            layout = QVBoxLayout(self)
            self.line_edit = QLineEdit(self)
            layout.addWidget(self.line_edit)

            # Signal connection
            self.line_edit.textChanged.connect(self.validate_input)

        def validate_input(self, text):
            # print(getattr(self.line_edit, "_highlight_tooltip_auto_attribute", None))
            if "@" not in text:
                # QErrorToolTip(self.line_edit, text="Password must contain '@' symbol!",
                #               text_position="bottom",
                #               highlight_widgets=[self.line_edit],
                #               shake_widgets=[self.line_edit],
                #               text_duration=10,
                #               shake_duration=1000,
                #               highlight_duration=3000,
                #               show_until_event=self.line_edit.textChanged
                #               )
                # TextToolTip(self.line_edit, text="Password must contain '@' symbol!" * 1, do_until=self.line_edit.textChanged)
                HighlightToolTip(self.line_edit, do_until=self.line_edit.textChanged)


    app = QApplication(sys.argv)
    window = ValidatorWidget()
    window.show()
    sys.exit(app.exec_())
