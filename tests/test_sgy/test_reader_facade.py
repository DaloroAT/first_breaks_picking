from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

from first_breaks.sgy.headers import (
    FileHeaderField,
    HeaderInfo,
    InvalidHeaders,
    read_custom_traces_header,
    write_custom_traces_header,
)
from first_breaks.sgy.sgy import SGY
from first_breaks.sgy.types import (
    REV0_FILE_HEADER_SIZE,
    REV0_TRACE_HEADER_SIZE,
    DataFormat,
    Endianness,
    NotImplementedReader,
    SGYLayout,
    SGYRevision,
)


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


def test_sgy_write_accepts_endianness_and_file_header_values_are_enum_keyed(tmp_path) -> None:
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


def test_custom_trace_header_helpers_read_and_write_values() -> None:
    layout = _make_layout()
    pointer = BytesIO(bytearray(layout.file_header_size + layout.num_traces * layout.trace_block_size))
    info = HeaderInfo(220, "I")
    values = np.array([10, 20, 30], dtype=np.uint32)

    write_custom_traces_header(pointer, values, info, layout)

    assert read_custom_traces_header(pointer, info, layout) == tuple(values.tolist())


def test_custom_trace_header_helpers_reject_values_outside_trace_header() -> None:
    layout = _make_layout()
    pointer = BytesIO(bytearray(layout.file_header_size + layout.num_traces * layout.trace_block_size))

    with pytest.raises(InvalidHeaders, match="exceeds trace header size"):
        read_custom_traces_header(pointer, HeaderInfo(238, "I"), layout)


def test_sgy_reads_custom_trace_header_from_arbitrary_offset(tmp_path) -> None:  # type: ignore[no-untyped-def]
    traces = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]], dtype=np.float32)
    output_path = tmp_path / "custom_headers.sgy"
    info = HeaderInfo(220, "I")
    values = np.array([100, 200, 300], dtype=np.uint32)

    SGY(traces, dt_mcs=1000).write(output_path)
    layout = SGYLayout.from_file(output_path)
    with output_path.open("rb+") as pointer:
        write_custom_traces_header(pointer, values, info, layout)

    assert SGY(output_path).read_custom_trace_header(byte_position=220, encoding="I") == tuple(values.tolist())


def test_sgy_reads_custom_trace_header_from_bytes_source(tmp_path) -> None:  # type: ignore[no-untyped-def]
    traces = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]], dtype=np.float32)
    output_path = tmp_path / "custom_headers.sgy"
    info = HeaderInfo(220, "I")
    values = np.array([100, 200, 300], dtype=np.uint32)

    SGY(traces, dt_mcs=1000).write(output_path)
    layout = SGYLayout.from_file(output_path)
    with output_path.open("rb+") as pointer:
        write_custom_traces_header(pointer, values, info, layout)

    assert SGY(output_path.read_bytes()).read_custom_trace_header(byte_position=220, encoding="I") == tuple(
        values.tolist()
    )


def test_sgy_exports_picks_to_custom_trace_header(tmp_path) -> None:  # type: ignore[no-untyped-def]
    traces = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]], dtype=np.float32)
    source_path = tmp_path / "source.sgy"
    output_path = tmp_path / "exported.sgy"
    picks = [1000, 2000, 3000]
    SGY(traces, dt_mcs=1000).write(source_path)

    SGY(source_path).export_sgy_with_picks(output_path, picks, byte_position=220, encoding="I")
    exported = SGY(output_path)

    assert exported.read_custom_trace_header(byte_position=220, encoding="I") == tuple(picks)
    assert np.allclose(exported.read(), traces)


def test_sgy_array_source_rejects_custom_trace_header_read() -> None:
    traces = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]], dtype=np.float32)

    with pytest.raises(NotImplementedReader, match="file and bytes sources"):
        SGY(traces, dt_mcs=1000).read_custom_trace_header(byte_position=220, encoding="I")


def _make_layout() -> SGYLayout:
    return SGYLayout(
        dt_mcs=1000,
        num_samples=4,
        num_traces=3,
        data_format=DataFormat.IEEE_FLOAT,
        endianness=Endianness.BIG,
        revision=SGYRevision.REV_0,
        file_header_size=REV0_FILE_HEADER_SIZE,
        trace_header_size=REV0_TRACE_HEADER_SIZE,
    )
