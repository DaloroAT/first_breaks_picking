from typing import Optional

import numpy as np


def preprocess_gather(
    data: np.ndarray, gain: float = 1.0, clip: float = 1.0, normalize: Optional[str] = 'trace', copy: bool = True
) -> np.ndarray:
    assert normalize in ['trace', 'gather', None]

    if copy:
        data = data.copy()

    norma = 1.0
    if normalize is None:
        pass
    elif normalize == 'trace':
        norma = np.nanmean(np.abs(data), axis=0)
        norma[norma < 1e-9 * np.max(norma)] = 1.0
    elif normalize == 'gather':
        norma = np.nanmean(np.abs(data))
        norma = norma if norma > 1e-9 else 1.0

    data = data / norma

    data = data * gain
    data = np.clip(data, -clip, clip)
    return data
