from typing import Tuple, Optional, Generator, Sequence, List

import numpy as np

from first_breaks.utils.utils import chunk_iterable

SizeHW = Tuple[int, int]


class InvalidSamplesSlice(Exception):
    pass


class ISGY:
    @property
    def content_hash(self) -> str:
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self.content_hash)

    @property
    def dt(self) -> int:
        raise NotImplementedError

    @property
    def ns(self) -> int:
        raise NotImplementedError

    @property
    def ntr(self) -> int:
        raise NotImplementedError

    @property
    def dt_ms(self) -> float:
        return self.dt * 1e-3

    @property
    def num_samples(self) -> int:
        return self.ns

    @property
    def num_traces(self) -> int:
        return self.ntr

    @property
    def shape(self) -> SizeHW:
        return self.ns, self.ntr

    def close(self):
        pass

    def get_bytes(self) -> bytes:
        raise NotImplementedError

    def read_traces_by_ids(
            self, ids: Sequence[int], min_sample: Optional[int] = None, max_sample: Optional[int] = None
    ) -> np.ndarray:
        raise NotImplementedError

    def read(self, min_sample: Optional[int] = None, max_sample: Optional[int] = None) -> np.ndarray:
        ids = list(range(self.num_traces))
        return self.read_traces_by_ids(ids, min_sample, max_sample)

    def get_chunked_reader(
        self, chunk_size: int, min_sample: Optional[int] = None, max_sample: Optional[int] = None
    ) -> Generator[np.ndarray, None, None]:
        chunk_size = min(chunk_size, self.num_traces)
        all_ids = list(range(self.num_traces))

        for ids in chunk_iterable(all_ids, chunk_size):
            yield self.read_traces_by_ids(ids, min_sample, max_sample)

    def parse_reading_params(
        self, ids: Sequence[int], min_sample: Optional[int] = None, max_sample: Optional[int] = None
    ) -> Tuple[List[int], int, int]:

        if min_sample is not None:
            if min_sample < 0 or not isinstance(min_sample, int):
                raise InvalidSamplesSlice("Invalid minimum slice index")
            min_sample = int(np.clip(min_sample, 0, self.num_samples))
        else:
            min_sample = 0

        if max_sample is not None:
            if max_sample < 1 or not isinstance(max_sample, int):
                raise InvalidSamplesSlice("Invalid maximum slice index")
            max_sample = int(np.clip(max_sample, 0, self.num_samples))
        else:
            max_sample = self.num_samples

        if min_sample >= max_sample:
            raise InvalidSamplesSlice("Minimum slice index is greater or equal to maximum index")

        len_slice = max_sample - min_sample
        ids = [idx for idx in ids if idx < self.num_traces]

        if len(ids) == 0:
            raise ValueError("The requested IDs were not found in the file")

        return ids, min_sample, len_slice
