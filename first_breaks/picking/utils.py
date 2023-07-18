from typing import Optional

import numpy as np

from first_breaks.data_models.independent import TNormalize
from first_breaks.data_models.initialised_defaults import DEFAULTS


def preprocess_gather(
    data: np.ndarray,
    gain: float = DEFAULTS.gain,
    clip: float = DEFAULTS.clip,
    normalize: TNormalize = DEFAULTS.normalize,
    copy: bool = True,
) -> np.ndarray:
    assert normalize in ["trace", "gather", None]

    if copy:
        data = data.copy()

    norma = 1.0
    if normalize is None:
        norma = 1.0
    elif isinstance(normalize, (float, int, np.number)):
        norma = normalize
    elif normalize == "trace":
        norma = np.nanmean(np.abs(data), axis=0)
        norma[norma < 1e-9 * np.max(norma)] = 1.0
    elif normalize == "gather":
        norma = np.nanmean(np.abs(data))
        norma = norma if norma > 1e-9 else 1.0
    else:
        raise ValueError("Unsupported `normalize` type and value")

    data = data / norma

    data = data * gain
    data = np.clip(data, -clip, clip)
    return data
