from typing import Optional, Tuple

import numpy as np
from numpy import fft


def fourier_transform(input_signal: np.ndarray) -> np.ndarray:
    spectrum = fft.fft(input_signal, axis=0)
    spectrum = _remove_negative_frequencies_of_spectrum(spectrum)
    return spectrum


def get_frequencies(len_signal: int, fs: float) -> np.ndarray:
    return fft.rfftfreq(len_signal, 1 / fs)[_get_slice_positive_frequencies(len_signal)]


def inverse_fourier_transform(spectrum: np.ndarray, len_signal: int) -> np.ndarray:
    reconstructed_spectrum = _reconstruct_negative_frequencies_for_spectrum(spectrum, len_signal)
    inverted_signal = fft.ifft(reconstructed_spectrum, n=len_signal, axis=0)
    inverted_signal = np.real(inverted_signal)
    return inverted_signal


def _remove_negative_frequencies_of_spectrum(raw_fft: np.ndarray) -> np.ndarray:
    return raw_fft[_get_slice_positive_frequencies(len(raw_fft))]


def _get_slice_positive_frequencies(len_signal: int) -> slice:
    if len_signal % 2 == 0:
        return slice(0, len_signal // 2 + 1, 1)
    else:
        return slice(0, (len_signal + 1) // 2, 1)


def _reconstruct_negative_frequencies_for_spectrum(spectrum: np.ndarray, len_signal: int) -> np.ndarray:
    if len_signal % 2 == 0:  # Even number of samples
        negative_freqs = np.conj(spectrum[-2:0:-1])  # Exclude DC and Nyquist
    else:  # Odd number of samples
        negative_freqs = np.conj(spectrum[-1:0:-1])  # Exclude only DC
    new_spectrum = np.concatenate((spectrum, negative_freqs))
    return new_spectrum


def get_mean_amplitude_spectrum(data: np.ndarray, fs: float, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    spec = fourier_transform(data)
    spec = np.abs(spec)
    if data.ndim == 2:
        spec = np.mean(spec, axis=1)
    if normalize:
        spec /= len(data)
    freq = get_frequencies(len(data), fs)
    return freq, spec


def build_amplitude_filter(
    frequencies: np.ndarray,
    f1_f2: Optional[Tuple[float, float]] = None,
    f3_f4: Optional[Tuple[float, float]] = None,
    filter_type: str = "pass",
) -> np.ndarray:
    assert filter_type in ["pass", "reject"]
    assert np.all(frequencies >= 0)
    assert all(
        pair is None or (isinstance(pair, (list, tuple, np.array)) and len(pair) == 2) for pair in [f1_f2, f3_f4]
    )

    if f1_f2 is None and f3_f4 is None:
        if filter_type == "pass":
            return np.ones_like(frequencies)
        else:
            return np.zeros_like(frequencies)

    else:
        amp_filter = np.zeros_like(frequencies)

        if f1_f2:
            f1, f2 = f1_f2
            assert f2 > f1

            rise_part = 0.5 - 0.5 * np.cos(np.pi * (frequencies - f1) / (f2 - f1))
            rise_part[frequencies < f1] = 0
            rise_part[frequencies > f2] = 1

            amp_filter += rise_part

        if f3_f4:
            f3, f4 = f3_f4
            assert f4 > f3

            fall_part = 0.5 + 0.5 * np.cos(np.pi * (frequencies - f3) / (f4 - f3))
            fall_part[frequencies < f3] = 1
            fall_part[frequencies > f4] = 0

            amp_filter += fall_part

        if f1_f2 and f3_f4:
            amp_filter -= 1

        if filter_type == "reject":
            amp_filter = 1 - amp_filter

        return amp_filter


def get_filtered_data(
    data: np.ndarray,
    fs: float,
    f1_f2: Optional[Tuple[float, float]] = None,
    f3_f4: Optional[Tuple[float, float]] = None,
    filter_type: str = "pass",
) -> np.ndarray:
    src_dtype = data.dtype
    length_signal = len(data)
    spectrum = fourier_transform(data)
    frequencies = get_frequencies(length_signal, fs)
    amp_filter = build_amplitude_filter(frequencies, f1_f2, f3_f4, filter_type)
    if data.ndim == 2:
        amp_filter = amp_filter[:, None]
    filtered_spectrum = spectrum * amp_filter
    filtered_data = inverse_fourier_transform(filtered_spectrum, length_signal)
    filtered_data = filtered_data.astype(src_dtype)
    return filtered_data


if __name__ == "__main__":
    t = np.linspace(0, 1)
    y = np.sin(2 * np.pi * t)
    print(get_mean_amplitude_spectrum(y, 1 / (t[1] - t[0])))
