from typing import Any, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from first_breaks.desktop.tooltip_widget import (
    HighlightToolTip,
    ShakeToolTip,
    TextToolTip,
)


class QBandFilterWidget(QWidget):

    def __init__(
        self,
        f1: Optional[float] = None,
        f2: Optional[float] = None,
        f3: Optional[float] = None,
        f4: Optional[float] = None,
        margins: Optional[int] = None,
        debug: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.layout = QHBoxLayout()
        if margins is not None:
            self.layout.setContentsMargins(margins, margins, margins, margins)
        self.setLayout(self.layout)

        self.freq_widgets = {}
        for i, f in enumerate([f1, f2, f3, f4], 1):
            label = QLabel(self._get_freq_str(i))
            self.layout.addWidget(label)

            freq_widget = QLineEdit()
            validator = QDoubleValidator()
            validator.setBottom(0.0)
            freq_widget.setValidator(validator)

            self.layout.addWidget(freq_widget)
            self.freq_widgets[i] = freq_widget
            if debug:
                freq_widget.textChanged.connect(self.validate_and_get_values)

    def enable_freqs_fields(self):
        for v in self.freq_widgets.values():
            v.setEnabled(True)

    def disable_freqs_fields(self):
        for v in self.freq_widgets.values():
            v.setEnabled(False)

    @staticmethod
    def _get_freq_str(index: int):
        return f"f<sub>{index}</sub>"

    def get_raw_freqs(self):
        freqs = {}
        for i in range(1, 5):
            value = self.freq_widgets[i].text()
            value = float(value) if value else None
            freqs[i] = value
        return freqs

    def validate_and_get_values(self):
        freqs = self.get_raw_freqs()

        problematic_widgets = set()
        recommendation_tips = []

        for idx1, idx2 in [[1, 2], [3, 4]]:
            if (freqs[idx1] is None and freqs[idx2] is not None) or (freqs[idx1] is not None and freqs[idx2] is None):
                text = (
                    f"Parameters {self._get_freq_str(idx1)} and {self._get_freq_str(idx2)} must "
                    f"either be both specified or both empty"
                )
                recommendation_tips.append(text)
                problematic_widgets.add(self.freq_widgets[idx1])
                problematic_widgets.add(self.freq_widgets[idx2])
            elif (freqs[idx1] is not None and freqs[idx2] is not None) and (freqs[idx1] >= freqs[idx2]):
                text = f"{self._get_freq_str(idx2)} must be greater than or equal to {self._get_freq_str(idx1)}"
                recommendation_tips.append(text)
                problematic_widgets.add(self.freq_widgets[idx1])
                problematic_widgets.add(self.freq_widgets[idx2])

        if (freqs[2] is not None and freqs[3] is not None) and (freqs[2] >= freqs[3]):
            text = (
                f"{self._get_freq_str(3)} must be greater than or equal to {self._get_freq_str(2)} if both of "
                f"them are not empty"
            )
            recommendation_tips.append(text)
            problematic_widgets.add(self.freq_widgets[2])
            problematic_widgets.add(self.freq_widgets[3])

        if problematic_widgets:
            if len(recommendation_tips) > 1:
                recommendation_tips = [f"<span>&#8226;</span>{tip}" for tip in recommendation_tips]
                recommendations = "<br>".join(recommendation_tips)
            else:
                recommendations = recommendation_tips[0]

            text_changed_signals = [w.textChanged for w in problematic_widgets]
            problematic_widgets = list(problematic_widgets)

            TextToolTip(widget=self, text=recommendations, do_until=text_changed_signals)
            ShakeToolTip(widgets=list(problematic_widgets), parent=self, do_until=2000)
            HighlightToolTip(widgets=problematic_widgets, parent=self, do_until=text_changed_signals)
            return None
        else:
            return freqs


if __name__ == "__main__":
    app = QApplication([])
    window = QBandFilterWidget()
    window.show()
    app.exec_()
