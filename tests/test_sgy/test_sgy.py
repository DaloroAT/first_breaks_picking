from pathlib import Path
from random import randint
from typing import Type

import numpy as np
import pytest

from first_breaks.const import PROJECT_ROOT
from first_breaks.sgy.headers import TraceHeaderField, FileHeadersPython, TraceHeadersPython

from first_breaks.sgy.sgy import SGY
from first_breaks.sgy.traces import TracesBackendArray
from first_breaks.sgy.types import Endianness
from first_breaks.utils.utils import multiply_iterable_by, calc_hash


ROUND_TRIP_FILES = sorted((PROJECT_ROOT / "tests/data").glob("*.sgy"))


@pytest.mark.parametrize("file", ROUND_TRIP_FILES, ids=lambda x: x.stem)
def test_round_trip(file: Path, tmp_path: Path) -> None:
    sgy = SGY(file)
    tmp_path = tmp_path / file.stem
    sgy.write(output_path=tmp_path)
    assert calc_hash(file) == calc_hash(tmp_path)


@pytest.mark.parametrize("file", ROUND_TRIP_FILES, ids=lambda x: x.stem)
def test_round_trip_with_explicit_same_layout_params(file: Path, tmp_path: Path) -> None:
    sgy = SGY(file)
    output_path = tmp_path / file.name

    sgy.write(output_path=output_path, data_format=sgy.sample_format, endianness=sgy.endianness)

    assert calc_hash(file) == calc_hash(output_path)


@pytest.mark.parametrize("file", ROUND_TRIP_FILES, ids=lambda x: x.stem)
def test_write_with_layout_change_preserves_traces_semantically(file: Path, tmp_path: Path) -> None:
    sgy = SGY(file)
    output_path = tmp_path / file.name
    output_endianness = Endianness.LITTLE if sgy.endianness == Endianness.BIG else Endianness.BIG

    sgy.write(output_path=output_path, endianness=output_endianness)
    written = SGY(output_path)

    assert written.endianness == output_endianness
    assert written.shape == sgy.shape
    assert np.allclose(written.read(), sgy.read())
    assert calc_hash(file) != calc_hash(output_path)


@pytest.mark.parametrize("file", ROUND_TRIP_FILES, ids=lambda x: x.stem)
def test_write_round_trip_through_python(file: Path, tmp_path: Path) -> None:
    sgy = SGY(file)
    layout = sgy.layout
    traces = TracesBackendArray(array=sgy.read(), layout=layout)
    file_headers = FileHeadersPython(values=sgy.general_headers, layout=layout)
    traces_headers = TraceHeadersPython(values=sgy.trace_headers.raw(), layout=layout)
    output_path = tmp_path / file.name
    with open(output_path, "wb+") as f:
        file_headers.write_to_sgy_pointer(f)
        traces_headers.write_to_sgy_pointer(f)
        traces.write_to_sgy_pointer(f, output_layout=None)

    assert calc_hash(file) == calc_hash(output_path)


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
    picks_col_name = TraceHeaderField.FB_PICK.name

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
