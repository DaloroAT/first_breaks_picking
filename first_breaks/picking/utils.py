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
    assert data.ndim == 2, "Only 2D arrays are acceptable"
    assert isinstance(gain, (int, float, np.number))
    assert isinstance(clip, (int, float, np.number)) and clip >= 0

    if copy:
        data = data.copy()

    if normalize is None:
        norma = 1.0
    elif isinstance(normalize, (float, int, np.number)):
        norma = normalize
    elif isinstance(normalize, np.ndarray):
        if normalize.ndim == 1:
            assert (
                len(normalize) == data.shape[1]
            ), "Normalize with numpy have to have the same length as number of traces"
            norma = normalize[None, :]
        elif normalize.ndim == 2:
            assert normalize.shape == (
                1,
                data.shape[1],
            ), "Normalize with numpy have to have the same length as number of traces"
            norma = normalize
        else:
            raise ValueError("Only 1D or 1xD arrays can be used for normalize with numpy")
    elif isinstance(normalize, (list, tuple)):
        assert all(
            isinstance(e, (int, float, np.number)) for e in normalize
        ), "Only numbers can be used for normalize with sequence"
        assert (
            len(normalize) == data.shape[1]
        ), "Normalize with sequence have to have the same length as number of traces"
        norma = np.array(normalize)
    elif normalize == "trace":
        norma = np.nanmean(np.abs(data), axis=0)
        norma[norma < 1e-9 * np.max(norma)] = 1.0  # type: ignore
    elif normalize == "gather":
        norma = np.nanmean(np.abs(data))
        norma = norma if norma > 1e-9 else 1.0
    else:
        raise ValueError("Unsupported `normalize` type and value")

    data = data / norma

    data = data * gain
    data = np.clip(data, -clip, clip)
    return data
