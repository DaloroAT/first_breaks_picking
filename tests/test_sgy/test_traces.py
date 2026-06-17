from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

import first_breaks.sgy.traces as traces_module
from first_breaks.sgy.traces import (
    TracesBackendBytes,
    TracesBackendFile,
    decode_blocks,
    encode_blocks,
    get_traces_backend,
    read_traces,
    write_traces,
)
from first_breaks.sgy.types import (
    REV0_FILE_HEADER_SIZE,
    REV0_TRACE_HEADER_SIZE,
    DataFormat,
    Endianness,
    InvalidSamplesSlice,
    SGYLayout,
    SGYRevision,
)


def make_layout(data_format: DataFormat = DataFormat.IEEE_FLOAT) -> SGYLayout:
    return SGYLayout(
        dt_mcs=1000,
        num_samples=4,
        num_traces=3,
        data_format=data_format,
        endianness=Endianness.BIG,
        revision=SGYRevision.REV_0,
        file_header_size=REV0_FILE_HEADER_SIZE,
        trace_header_size=REV0_TRACE_HEADER_SIZE,
    )


def make_pointer(traces: np.ndarray, layout: SGYLayout) -> BytesIO:
    payload = bytearray(layout.file_header_size)
    for block in encode_blocks(traces, layout):
        payload.extend(bytearray(layout.trace_header_size))
        payload.extend(block)
    return BytesIO(payload)


def make_payload(traces: np.ndarray, layout: SGYLayout) -> bytes:
    return make_pointer(traces, layout).getvalue()


def test_decode_blocks_returns_samples_by_traces() -> None:
    layout = make_layout()
    traces = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=np.float32,
    )

    decoded = decode_blocks(encode_blocks(traces, layout), layout)

    assert decoded.shape == (4, 2)
    assert np.allclose(decoded, traces)


def test_decode_blocks_keeps_single_block_2d() -> None:
    layout = make_layout()
    raw = [np.array([1.0, 2.0, 3.0, 4.0], dtype=">f4").tobytes()]

    decoded = decode_blocks(raw, layout)

    assert decoded.shape == (4, 1)
    assert np.allclose(decoded[:, 0], [1.0, 2.0, 3.0, 4.0])


def test_decode_blocks_rejects_unequal_blocks() -> None:
    layout = make_layout()

    with pytest.raises(ValueError, match="equal size"):
        decode_blocks([b"\x00\x00\x00\x00", b"\x00\x00\x00\x00\x00\x00\x00\x00"], layout)


def test_decode_blocks_rejects_non_sample_aligned_block() -> None:
    layout = make_layout()

    with pytest.raises(ValueError, match="divisible"):
        decode_blocks([b"\x00\x00"], layout)


def test_read_traces_reads_selected_trace_ids_and_sample_slice() -> None:
    layout = make_layout()
    traces = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=np.float32,
    )
    pointer = make_pointer(traces, layout)

    decoded = read_traces(pointer, trace_ids=[2, 0], layout=layout, min_sample=1, max_sample=3)

    assert decoded.shape == (2, 2)
    assert np.allclose(decoded, [[200.0, 2.0], [300.0, 3.0]])


def test_read_traces_empty_trace_ids_keeps_sample_dimension() -> None:
    layout = make_layout()
    pointer = make_pointer(np.zeros(layout.shape, dtype=np.float32), layout)

    decoded = read_traces(pointer, trace_ids=[], layout=layout, min_sample=1, max_sample=3)

    assert decoded.shape == (2, 0)


def test_read_traces_rejects_invalid_sample_slice() -> None:
    layout = make_layout()
    pointer = make_pointer(np.zeros(layout.shape, dtype=np.float32), layout)

    with pytest.raises(InvalidSamplesSlice):
        read_traces(pointer, trace_ids=[0], layout=layout, min_sample=3, max_sample=3)


def test_write_traces_writes_partial_samples() -> None:
    layout = make_layout()
    pointer = make_pointer(np.zeros(layout.shape, dtype=np.float32), layout)
    update = np.array([[2.0, 20.0], [3.0, 30.0]], dtype=np.float32)

    write_traces(pointer, trace_ids=[0, 2], traces=update, layout=layout, start_sample=1)

    decoded = read_traces(pointer, trace_ids=[0, 1, 2], layout=layout)
    assert np.allclose(
        decoded,
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 20.0],
            [3.0, 0.0, 30.0],
            [0.0, 0.0, 0.0],
        ],
    )


def test_write_traces_rejects_trace_count_mismatch() -> None:
    layout = make_layout()
    pointer = make_pointer(np.zeros(layout.shape, dtype=np.float32), layout)

    with pytest.raises(ValueError, match="trace ids"):
        write_traces(pointer, trace_ids=[0, 1], traces=np.ones((4, 1), dtype=np.float32), layout=layout)


def test_ibm_encode_decode_roundtrip() -> None:
    layout = make_layout(DataFormat.IBM_FLOAT)
    traces = np.array(
        [
            [0.0, 1.0],
            [-2.0, 16.0],
            [0.5, -32.0],
            [10.25, 100.0],
        ],
        dtype=np.float32,
    )

    decoded = decode_blocks(encode_blocks(traces, layout), layout)

    assert decoded.dtype == np.float32
    assert np.allclose(decoded, traces, rtol=1e-6)


def test_array_backend_reads_traces() -> None:
    layout = make_layout()
    traces = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=np.float32,
    )

    backend = get_traces_backend(traces, layout)

    assert np.allclose(backend.read(), traces)
    assert np.allclose(backend.read_traces_by_ids([2, 0], min_sample=1, max_sample=3), [[200.0, 2.0], [300.0, 3.0]])


def test_bytes_backend_caches_after_full_read() -> None:
    layout = make_layout()
    traces = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=np.float32,
    )
    backend = TracesBackendBytes(make_payload(traces, layout), layout)

    first_read = backend.read()
    second_read = backend.read_traces_by_ids([1], min_sample=1, max_sample=3)

    assert np.allclose(first_read, traces)
    assert np.allclose(second_read, [[20.0], [30.0]])


def test_file_backend_caches_after_full_read(tmp_path) -> None:  # type: ignore[no-untyped-def]
    layout = make_layout()
    traces = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=np.float32,
    )
    source_path = tmp_path / "source.sgy"
    source_path.write_bytes(make_payload(traces, layout))
    backend = TracesBackendFile(source_path, layout)

    first_read = backend.read()
    second_read = backend.read_traces_by_ids([2, 0], min_sample=0, max_sample=2)

    assert np.allclose(first_read, traces)
    assert np.allclose(second_read, [[100.0, 1.0], [200.0, 2.0]])


def test_backend_write_to_sgy_pointer_uses_explicit_output_layout() -> None:
    layout = make_layout()
    output_layout = SGYLayout(
        dt_mcs=layout.dt_mcs,
        num_samples=layout.num_samples,
        num_traces=layout.num_traces,
        data_format=DataFormat.INT16,
        endianness=Endianness.BIG,
        revision=layout.revision,
        file_header_size=layout.file_header_size,
        trace_header_size=layout.trace_header_size,
    )
    traces = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=np.float32,
    )
    backend = get_traces_backend(traces, layout)
    pointer = BytesIO(bytearray(output_layout.file_header_size + output_layout.num_traces * output_layout.trace_block_size))

    backend.write_to_sgy_pointer(pointer, output_layout=output_layout)
    decoded = read_traces(pointer, trace_ids=range(output_layout.num_traces), layout=output_layout)

    assert np.array_equal(decoded, traces.astype(np.int16))


def test_backend_write_to_sgy_pointer_accepts_explicit_none_output_layout() -> None:
    layout = make_layout()
    traces = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=np.float32,
    )
    backend = get_traces_backend(traces, layout)
    pointer = BytesIO(bytearray(layout.file_header_size + layout.num_traces * layout.trace_block_size))

    backend.write_to_sgy_pointer(pointer, output_layout=None)
    decoded = read_traces(pointer, trace_ids=range(layout.num_traces), layout=layout)

    assert np.allclose(decoded, traces)


def test_bytes_backend_same_layout_write_copies_raw_trace_data_without_encoding(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    layout = make_layout()
    traces = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=np.float32,
    )
    backend = TracesBackendBytes(make_payload(traces, layout), layout)
    output_layout = SGYLayout(
        dt_mcs=layout.dt_mcs,
        num_samples=layout.num_samples,
        num_traces=layout.num_traces,
        data_format=layout.data_format,
        endianness=layout.endianness,
        revision=layout.revision,
        file_header_size=layout.file_header_size,
        trace_header_size=layout.trace_header_size,
    )
    pointer = BytesIO(bytearray(output_layout.file_header_size + output_layout.num_traces * output_layout.trace_block_size))

    def fail_write_traces(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError("same-layout bytes backend write must not encode traces")

    monkeypatch.setattr(traces_module, "write_traces", fail_write_traces)

    backend.write_to_sgy_pointer(pointer, output_layout=output_layout)
    decoded = read_traces(pointer, trace_ids=range(output_layout.num_traces), layout=output_layout)

    assert np.allclose(decoded, traces)


def test_file_backend_same_layout_write_copies_raw_trace_data_without_encoding(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    layout = make_layout()
    traces = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=np.float32,
    )
    source_path = tmp_path / "source.sgy"
    source_path.write_bytes(make_payload(traces, layout))
    backend = TracesBackendFile(source_path, layout)
    pointer = BytesIO(bytearray(layout.file_header_size + layout.num_traces * layout.trace_block_size))

    def fail_write_traces(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError("same-layout file backend write must not encode traces")

    monkeypatch.setattr(traces_module, "write_traces", fail_write_traces)

    backend.write_to_sgy_pointer(pointer, output_layout=None)
    decoded = read_traces(pointer, trace_ids=range(layout.num_traces), layout=layout)

    assert np.allclose(decoded, traces)
