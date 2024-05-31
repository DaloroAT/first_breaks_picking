from pathlib import Path
from random import randint
from typing import Type

import numpy as np
import pytest

from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import multiply_iterable_by


def test_reader_open_different_sources(demo_sgy: Path) -> None:
    sgy_from_path = SGY(demo_sgy)
    traces_from_path = sgy_from_path.read()

    sgy_from_str = SGY(str(demo_sgy))
    traces_from_str = sgy_from_str.read()

    with open(demo_sgy, "rb") as f_io:
        sgy_from_bytes = SGY(f_io.read())
        traces_from_bytes = sgy_from_bytes.read()

    assert np.all(traces_from_path == traces_from_str)
    assert np.all(traces_from_path == traces_from_bytes)

    sgy_from_ndarray = SGY(traces_from_path, dt_mcs=1e3)
    traces_from_ndarray = sgy_from_ndarray.read()

    assert np.all(traces_from_path == traces_from_ndarray)


@pytest.mark.parametrize("picks_in_samples_type", [list, np.ndarray])
def test_export_picks(demo_sgy: Path, picks_in_samples_type: Type, logs_dir_for_tests: Path) -> None:  # type: ignore
    sgy = SGY(demo_sgy)
    picks_col_name = sgy._traces_headers_schema.fb_pick_default

    assert np.all(sgy.traces_headers[picks_col_name] == 0), sgy.traces_headers[picks_col_name]

    if picks_in_samples_type == list:
        picks_in_samples = [randint(0, sgy.num_samples) for _ in range(sgy.num_traces)]
    elif picks_in_samples_type == np.ndarray:
        picks_in_samples = np.random.randint(0, sgy.num_samples, sgy.num_traces)
    else:
        raise TypeError("Invalid type")

    picks_in_mcs = multiply_iterable_by(picks_in_samples, sgy.dt_mcs, cast_to=int)

    sgy_with_picks_path = logs_dir_for_tests / "sgy_with_picks.sgy"
    sgy.export_sgy_with_picks(sgy_with_picks_path, picks_in_mcs)  # type: ignore
    sgy_with_picks = SGY(sgy_with_picks_path)

    assert np.all(picks_in_mcs == sgy_with_picks.traces_headers[picks_col_name])
