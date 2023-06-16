from pathlib import Path

import numpy as np

from first_breaks.sgy.reader import SGY


def test_reader_open_different_sources(demo_sgy: Path) -> None:
    sgy_from_path = SGY(demo_sgy)
    traces_from_path = sgy_from_path.read()

    sgy_from_str = SGY(str(demo_sgy))
    traces_from_str = sgy_from_str.read()

    with open(demo_sgy, 'rb') as f_io:
        sgy_from_bytes = SGY(f_io.read())
        traces_from_bytes = sgy_from_bytes.read()

    assert np.all(traces_from_path == traces_from_str)
    assert np.all(traces_from_path == traces_from_bytes)

    sgy_from_ndarray = SGY(traces_from_path, dt_mcs=1e3)
    traces_from_ndarray = sgy_from_ndarray.read()

    assert np.all(traces_from_path == traces_from_ndarray)



