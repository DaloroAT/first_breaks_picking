from typing import Any, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSlider, QWidget


class QSliderWithValues(QWidget):
    value_changed_signal = pyqtSignal(float)
    slider_released_signal = pyqtSignal()
    value_on_released_signal = pyqtSignal(float)

    def __init__(
        self,
        value: float = 1,
        min_value: float = -1,
        max_value: float = 1,
        step: float = 0.1,
        decimals: int = 1,
        ticks_interval: Optional[float] = None,
        margins: Optional[int] = None,
        slider_space_fraction: Optional[float] = None,
        block_mouse_scrolling: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.layout = QHBoxLayout()
        if margins is not None:
            self.layout.setContentsMargins(margins, margins, margins, margins)
        self.setLayout(self.layout)

        self.slider_value = value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.decimals = decimals
        self.num_steps = int((self.max_value - self.min_value) / self.step)

        self.slider_label = QLabel(str(self.slider_value), self)

        self.slider_widget = QSlider(Qt.Horizontal, self)

        self.slider_widget.setRange(0, self.num_steps)
        self.slider_widget.setValue(self.value2slider(self.slider_value))

        if slider_space_fraction is not None:
            slider_space = int(slider_space_fraction * 100)
            self.layout.addWidget(self.slider_widget, slider_space)
            self.layout.addWidget(self.slider_label, 100 - slider_space)
        else:
            self.layout.addWidget(self.slider_widget)
            self.layout.addWidget(self.slider_label)

        if ticks_interval is not None:
            self.slider_widget.setTickInterval(int(ticks_interval / self.step))
            self.slider_widget.setTickPosition(QSlider.TicksAbove)
            self.slider_widget.setSingleStep(1)

        if block_mouse_scrolling:
            self.slider_widget.wheelEvent = lambda *e_args: e_args[-1].ignore()  # block scrolling with wheel

        self.slider_widget.sliderPressed.connect(self.slider_pressed)
        self.slider_widget.sliderReleased.connect(self.slider_released)
        self.slider_widget.sliderMoved.connect(self.value_changed)

        self.show()

    def slider_pressed(self) -> None:
        self.value_changed()

    def slider_released(self) -> None:
        self.value_on_released_signal.emit(self.slider_value)
        self.slider_released_signal.emit()

    def value_changed(self, *args: Any) -> None:
        self.slider_value = self.slider2value(self.slider_widget.value())
        self.slider_label.setText(self.text())
        self.value_changed_signal.emit(self.slider_value)

    def text(self) -> str:
        return f"{self.slider_value:.{self.decimals}f}"

    def value2slider(self, value_float: float) -> int:
        return int((value_float - self.min_value) / self.step)

    def slider2value(self, value_from_slider: int) -> float:
        return self.min_value + (value_from_slider * self.step)

    def value(self) -> float:
        return self.slider_value


if __name__ == "__main__":
    app = QApplication([])
    window = QSliderWithValues(1, -10, 10, slider_space_fraction=0.7, margins=20, ticks_interval=5)
    # window = QSlider(Qt.Horizontal)
    # window.valueChanged.connect(lambda x: print(x))
    # window.rele
    window.show()
    app.exec_()
