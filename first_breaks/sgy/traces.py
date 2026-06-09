from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from first_breaks.sgy.types import InvalidSamplesSlice, SGYInitParamsError, SGYLayout, SGYSource, SourceKind


class Traces:
    def __init__(self, layout: SGYLayout, source: SGYSource, array: Optional[np.ndarray] = None) -> None:
        self.layout = layout
        self.source = source
        self.__array = array

    @classmethod
    def from_array(cls, array: np.ndarray, layout: SGYLayout, *, copy: bool = False) -> "Traces":
        normalized = cls.__normalize_array(array)
        if normalized.shape != layout.shape:
            raise SGYInitParamsError(f"Trace array shape {normalized.shape} does not match layout shape {layout.shape}")
        if copy:
            normalized = normalized.copy()
        return cls(layout=layout, source=SGYSource(kind=SourceKind.ARRAY, value=array), array=normalized)

    @property
    def is_array_backed(self) -> bool:
        return self.source.kind == SourceKind.ARRAY

    def read(self, min_sample: Optional[int] = None, max_sample: Optional[int] = None) -> np.ndarray:
        return self.read_by_ids(range(self.layout.num_traces), min_sample=min_sample, max_sample=max_sample)

    def read_by_ids(
        self,
        ids: Sequence[int],
        *,
        min_sample: Optional[int] = None,
        max_sample: Optional[int] = None,
    ) -> np.ndarray:
        if self.__array is None:
            raise NotImplementedError("Trace materialization from file/bytes sources is not implemented yet")

        min_idx, max_idx = self.__normalize_sample_slice(min_sample, max_sample)
        trace_ids = list(ids)
        return self.__array[min_idx:max_idx, trace_ids]

    def replace_array(self, array: np.ndarray, *, copy: bool = False) -> None:
        normalized = self.__normalize_array(array)
        if normalized.shape != self.layout.shape:
            raise SGYInitParamsError(f"Trace array shape {normalized.shape} does not match layout shape {self.layout.shape}")
        self.__array = normalized.copy() if copy else normalized
        self.source = SGYSource(kind=SourceKind.ARRAY, value=array)

    def to_numpy(self, *, copy: bool = True) -> np.ndarray:
        if self.__array is None:
            raise NotImplementedError("Trace materialization from file/bytes sources is not implemented yet")
        return self.__array.copy() if copy else self.__array

    def __normalize_sample_slice(
        self,
        min_sample: Optional[int],
        max_sample: Optional[int],
    ) -> tuple[int, int]:
        min_idx = 0 if min_sample is None else int(min_sample)
        max_idx = self.layout.num_samples if max_sample is None else int(max_sample)
        if min_idx < 0 or max_idx < 0 or min_idx > max_idx or max_idx > self.layout.num_samples:
            raise InvalidSamplesSlice(
                f"Invalid sample slice [{min_idx}:{max_idx}] for {self.layout.num_samples} samples"
            )
        return min_idx, max_idx

    @staticmethod
    def __normalize_array(array: np.ndarray) -> np.ndarray:
        if array.ndim == 1:
            return array.reshape((-1, 1))
        if array.ndim == 2:
            return array
        raise SGYInitParamsError("Only 1D and 2D arrays can be used as SGY traces")
