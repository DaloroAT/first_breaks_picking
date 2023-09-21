from typing import Any, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSlider, QWidget, QLineEdit

from first_breaks.desktop.tooltip_widget import QErrorToolTip


class QBandFilterWidget(QWidget):
    def __init__(self, margins: Optional[int] = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.layout = QHBoxLayout()
        if margins is not None:
            self.layout.setContentsMargins(margins, margins, margins, margins)
        self.setLayout(self.layout)

        self.storage = {}
        for i in range(1, 5):
            label = QLabel(self._get_freq_str(i))
            self.layout.addWidget(label)

            freq_widget = QLineEdit()
            validator = QDoubleValidator()
            validator.setBottom(0.0)
            freq_widget.setValidator(validator)
            self.layout.addWidget(freq_widget)
            self.storage[i] = freq_widget
            freq_widget.textChanged.connect(self.validate_and_get_values)

    def _get_freq_str(self, index: int):
        return f"f<sub>{index}</sub>"

    def validate_and_get_values(self):
        freqs = {}
        for i in range(1, 5):
            value = self.storage[i].text()
            value = float(value) if value else None
            freqs[i] = value

        print(freqs)

        error_tooltip_kwargs = {"text_position": "right",
                                "shake_duration": 3000,
                                "highlight_duration": 3000}

        # for idx1, idx2 in [[1, 2], [3, 4]]:
        #     widgets = [self.storage[idx1], self.storage[idx2]]
        #     if (freqs[idx1] is None and freqs[idx2] is not None) or (freqs[idx1] is not None and freqs[idx2] is None):
        #         text = (f"Parameters {self._get_freq_str(idx1)} and {self._get_freq_str(idx2)} must "
        #                 f"either be both specified or both empty")
        #         QErrorToolTip(self, text=text, shake_widgets=widgets, highlight_widgets=widgets, **error_tooltip_kwargs)
        #         # return
        #     elif freqs[idx1] > freqs[idx2]:
        #         text = f"{self._get_freq_str(idx2)} must be greater than or equal to {self._get_freq_str(idx1)}"
        #         QErrorToolTip(self, text=text, shake_widgets=widgets, highlight_widgets=widgets, **error_tooltip_kwargs)
        #         # return

        problematic_widgets = set()
        recommendation_tips = []

        for idx1, idx2 in [[1, 2], [3, 4]]:
            if (freqs[idx1] is None and freqs[idx2] is not None) or (freqs[idx1] is not None and freqs[idx2] is None):
                text = (f"Parameters {self._get_freq_str(idx1)} and {self._get_freq_str(idx2)} must "
                        f"either be both specified or both empty")
                recommendation_tips.append(text)
                problematic_widgets.add(self.storage[idx1])
                problematic_widgets.add(self.storage[idx2])
            elif (freqs[idx1] is not None and freqs[idx2] is not None) and (freqs[idx1] > freqs[idx2]):
                text = f"{self._get_freq_str(idx2)} must be greater than or equal to {self._get_freq_str(idx1)}"
                recommendation_tips.append(text)
                problematic_widgets.add(self.storage[idx1])
                problematic_widgets.add(self.storage[idx2])

        if (freqs[2] is not None and freqs[3] is not None) and (freqs[2] > freqs[3]):
            text = (f"{self._get_freq_str(3)} must be greater than or equal to {self._get_freq_str(2)} if both of "
                    f"them are not empty")
            recommendation_tips.append(text)
            problematic_widgets.add(self.storage[2])
            problematic_widgets.add(self.storage[3])

        if problematic_widgets:
            if len(recommendation_tips) > 1:
                recommendation_tips = [f"<span>&#8226;</span>{tip}" for tip in recommendation_tips]
                recommendations = "<br>".join(recommendation_tips)
            else:
                recommendations = recommendation_tips[0]
            QErrorToolTip(self,
                          text=recommendations,
                          shake_widgets=list(problematic_widgets),
                          highlight_widgets=list(problematic_widgets),
                          **error_tooltip_kwargs)
            return None
        else:
            return freqs




if __name__ == "__main__":
    app = QApplication([])
    window = QBandFilterWidget()
    # window = QSlider(Qt.Horizontal)
    # window.valueChanged.connect(lambda x: print(x))
    # window.rele
    window.show()
    app.exec_()
