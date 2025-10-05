import numpy as np


def savgol_coeffs(window_length: int, polyorder: int, deriv: int = 0) -> np.ndarray:
    assert window_length % 2 == 1, "The value must be odd"
    assert polyorder % 2 == 1, "The value must be odd"
    assert polyorder < window_length

    half_window = (window_length - 1) // 2
    a = np.zeros((window_length, polyorder + 1))
    for i in range(window_length):
        for j in range(polyorder + 1):
            a[i, j] = (i - half_window) ** j
    atr_a = np.dot(a.T, a)
    atr = np.linalg.pinv(atr_a)
    b = np.dot(atr, a.T)
    coeffs = b[deriv]
    return coeffs


def apply_savgol_filter(data: np.ndarray, window_length: int, polyorder: int, deriv: int = 0) -> np.ndarray:
    """
    https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    """
    assert 1 <= data.ndim <= 2
    coeffs = savgol_coeffs(window_length, polyorder, deriv)

    half_window = (window_length - 1) // 2
    pad_mode = "reflect"

    if data.ndim == 1:
        padding = (half_window, half_window)
    else:
        padding = ((half_window, half_window), (0, 0))  # type: ignore

    padded_data = np.pad(data, padding, mode=pad_mode)

    filtered_data = np.apply_along_axis(lambda m: np.convolve(m, coeffs, mode="valid"), axis=0, arr=padded_data)
    return filtered_data
