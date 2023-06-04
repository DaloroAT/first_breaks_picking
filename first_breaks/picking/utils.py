import numpy as np


def preprocess_gather(
    data: np.ndarray, gain: float = 1.0, clip: float = 1.0, normalize: bool = True, copy: bool = True
) -> np.ndarray:
    if copy:
        data = data.copy()
    if normalize:
        norma = np.mean(np.abs(data), axis=0)
        norma[np.abs(norma) < 1e-9 * np.max(np.abs(norma))] = 1
        data = data / norma

    data = data * gain
    data = np.clip(data, -clip, clip)
    return data
