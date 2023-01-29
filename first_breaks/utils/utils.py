from itertools import islice
from typing import Any, Iterable, List, Tuple


def chunk_iterable(it: Iterable, size: int) -> List[Tuple[Any, ...]]:
    it = iter(it)
    return list(iter(lambda: tuple(islice(it, size)), ()))
