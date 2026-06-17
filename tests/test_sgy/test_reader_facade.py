from __future__ import annotations

import numpy as np

from first_breaks.sgy.headers import FileHeaderField
from first_breaks.sgy.sgy import SGY
from first_breaks.sgy.types import Endianness


def test_sgy_array_source_reads_traces() -> None:
    traces = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)

    sgy = SGY(traces, dt_mcs=500)

    assert sgy.shape == (3, 2)
    assert sgy.dt_mcs == 500
    assert sgy.is_source_ndarray
    assert np.allclose(sgy.read(), traces)
    assert np.allclose(sgy.read_traces_by_ids([1], min_sample=1, max_sample=3), [[20.0], [30.0]])


def test_sgy_file_and_bytes_sources_read_written_traces(tmp_path) -> None:  # type: ignore[no-untyped-def]
    traces = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    output_path = tmp_path / "synthetic.sgy"
    SGY(traces, dt_mcs=1000).write(output_path)

    from_file = SGY(output_path)
    from_bytes = SGY(output_path.read_bytes())

    assert not from_file.is_source_ndarray
    assert from_file.shape == traces.shape
    assert np.allclose(from_file.read(), traces)
    assert np.allclose(from_bytes.read_traces_by_ids([1, 0], min_sample=0, max_sample=2), [[10.0, 1.0], [20.0, 2.0]])


def test_sgy_write_accepts_endianness_and_file_header_values_are_enum_keyed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    traces = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    output_path = tmp_path / "synthetic_big_endian.sgy"
    sgy = SGY(traces, dt_mcs=1000)

    sgy.write(output_path, endianness=Endianness.BIG)
    written = SGY(output_path)

    assert np.allclose(written.read(), traces)
    assert FileHeaderField.DT in written.file_headers.values()
    assert written.file_headers.values()[FileHeaderField.DATA_SAMPLE_FORMAT] == int(written.sample_format)


def test_sgy_chunked_reader_uses_current_source() -> None:
    traces = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]], dtype=np.float32)
    sgy = SGY(traces, dt_mcs=1000)

    chunks = list(sgy.get_chunked_reader(chunk_size=2))

    assert len(chunks) == 2
    assert np.allclose(chunks[0], traces[:, :2])
    assert np.allclose(chunks[1], traces[:, 2:])
