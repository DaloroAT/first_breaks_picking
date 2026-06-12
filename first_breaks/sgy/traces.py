from __future__ import annotations

from typing import Optional, Sequence, List, IO, Generator

import numpy as np

from first_breaks.sgy.types import SGYLayout, SourceKind


def decode_blocks(raw: List[bytes] | List[bytearray], layout: SGYLayout) -> np.ndarray:
    # 1) use collection of bytes and sgy_layout.endianness with layout.data_format to interpret bytes as values
    # 2) it's not necessary that raw are full traces, it might be some slice of traces
    # 3) bytes are cooked/sliced externally
    # 4) you may use this function to dispatch into individual data_format implementations
    # 5) for IBM you can look at "sgy_old", maybe you will find already optimized readers
    # 6) function should always return 2D array even for a list with 1 raw block
    # 7) you may want to join input data into single buffer to call np.from_buffer once; or make some batching,
    #   so it's not individual trace and not super blob - might control via global, e.g. BATCH_TRACES=32 if necessary
    raise NotImplementedError


def read_traces(
    pointer: IO[bytes],
    trace_ids: Sequence[int],
    layout: SGYLayout,
    min_sample: Optional[int] = None,
    max_sample: Optional[int] = None,
) -> np.ndarray:
    # 1) this function prepare a list of block for "decode_blocks" function
    # 2) you should validate trace_ids, min_sample and max_sample against layout
    raise NotImplementedError


def get_chunked_reader(
    pointer: IO[bytes],
    chunk_size: int,
    layout: SGYLayout,
    min_sample: Optional[int] = None,
    max_sample: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    if chunk_size <= 0:
        raise ValueError("Argument 'chunk_size' must be positive")
    for start in range(0, layout.num_traces, chunk_size):
        stop = min(start + chunk_size, layout.num_traces)
        yield read_traces(
            pointer=pointer,
            trace_ids=list(range(start, stop)),
            min_sample=min_sample,
            max_sample=max_sample,
            layout=layout,
        )


def encode_blocks(traces: np.ndarray, layout: SGYLayout) -> List[bytearray | bytes]:
    # 1) it should be inverse to "decode_blocks"
    # 2) we don't control that num samples in traces is equal to layout.num_samples, we just encode what we get
    # 3) but need to verify amount of bytes
    # 4) Use all notes and recommendations from "decode_blocks", they are valid here too
    raise NotImplementedError


def write_traces(
    pointer: IO[bytes],
    trace_ids: Sequence[int],
    traces: np.ndarray,
    layout: SGYLayout,
    start_sample: Optional[int] = None,
) -> None:
    # 1) we may be able to write block of traces
    # 2) block might be started from start_sample
    # 3) check that start_sample + len(traces) fit into layout
    raise NotImplementedError
