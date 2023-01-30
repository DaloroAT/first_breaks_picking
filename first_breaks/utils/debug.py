from time import perf_counter
from typing import Any, Optional


class Performance:
    def __init__(self, prefix: Optional[str] = None):
        self.start_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.end_time: Optional[float] = None
        self.prefix = "Duration" if prefix is None else prefix

    def __enter__(self) -> None:
        self.start_time = perf_counter()

    def __exit__(self, *excs: Any) -> None:
        self.end_time = perf_counter()
        self.duration = self.end_time - self.start_time
        msg = f"{self.prefix}: {self.duration}"
        print(msg, flush=True)
