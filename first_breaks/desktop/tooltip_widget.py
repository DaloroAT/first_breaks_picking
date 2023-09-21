import sys
from typing import Union, Sequence, Optional, Callable, Any, Tuple
from weakref import WeakKeyDictionary

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QLabel, QFrame

from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt, QTimer, QPoint, QEasingCurve, QPropertyAnimation, pyqtSignal, QObject, pyqtBoundSignal


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


class _HighlightManager:
    _original_colors = WeakKeyDictionary()
    _highlight_counts = WeakKeyDictionary()

    @classmethod
    def highlight_widget(cls, widget: QWidget, color: Tuple[int, int, int]):
        if widget not in cls._original_colors:
            cls._original_colors[widget] = widget.palette()

        if widget not in cls._highlight_counts:
            cls._highlight_counts[widget] = 0

        cls._highlight_counts[widget] += 1
        palette = widget.palette()
        palette.setColor(QPalette.Base, QColor(*color))
        widget.setPalette(palette)

    @classmethod
    def remove_highlight(cls, widget: QWidget):
        if widget in cls._highlight_counts:
            cls._highlight_counts[widget] -= 1
            if cls._highlight_counts[widget] <= 0:
                widget.setPalette(cls._original_colors[widget])
                del cls._original_colors[widget]
                del cls._highlight_counts[widget]


class HighlightToolTip(QObject):
    _manager = _HighlightManager()

    def __init__(
            self,
            widgets: Union[QWidget, Sequence[QWidget]],
            parent: Optional[QWidget] = None,
            do_until: TDoUntil = 5000,
            color=(255, 220, 220),):
        if not parent and widgets:
            if isinstance(widgets, (list, tuple)):
                parent = widgets[0]
            else:
                parent = widgets
        super().__init__(parent)

        self.widgets = widgets if isinstance(widgets, (list, tuple)) else [widgets]

        for widget in self.widgets:
            self._manager.highlight_widget(widget, color)

        run_do_until(do_until, self.remove_highlights)

    def remove_highlights(self) -> None:
        for widget in self.widgets:
            self._manager.remove_highlight(widget)
        self.deleteLater()


class _Shaker:
    def __init__(self, widget: QWidget):
        self._widget = widget
        self._start_pos = widget.pos()
        self._animation = QPropertyAnimation(widget, b"pos")
        self._animation.setStartValue(self._start_pos)
        self._animation.setEndValue(self._start_pos)

        basic_duration = 1000
        shakes_per_second = 10

        num_shakes = max(1, int(shakes_per_second * basic_duration / 1000))
        for i in range(num_shakes):
            self._animation.setKeyValueAt((i + 1) / num_shakes, self._start_pos + (-1) ** i * QPoint(2, 0))

        self._animation.setDuration(basic_duration)
        self._animation.setLoopCount(-1)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)

    def shake(self):
        self._animation.start()

    def stop(self):
        self._animation.stop()
        self._widget.move(self._start_pos)


class ShakeToolTip(QObject):
    def __init__(self, widgets: Union[QWidget, Sequence[QWidget]], parent: Optional[QWidget] = None, do_until: TDoUntil = 5000):
        if not parent and widgets:
            if isinstance(widgets, (list, tuple)):
                parent = widgets[0]
            else:
                parent = widgets
        super().__init__(parent)

        self._shakers = []
        self.widgets = widgets if isinstance(widgets, (list, tuple)) else [widgets]

        for widget in self.widgets:
            shaker = _Shaker(widget)
            shaker.shake()
            self._shakers.append(shaker)

        run_do_until(do_until, self.stop_shaking)

    def stop_shaking(self):
        for shaker in self._shakers:
            shaker.stop()
        self.deleteLater()


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
            if "@" not in text:
                TextToolTip(self.line_edit,
                            text="Password must contain '@' symbol!" * 1,
                            do_until=self.line_edit.textChanged)
                HighlightToolTip(self.line_edit, do_until=self.line_edit.textChanged)
                ShakeToolTip(self.line_edit, do_until=self.line_edit.textChanged)


    app = QApplication(sys.argv)
    window = ValidatorWidget()
    window.show()
    sys.exit(app.exec_())
