from pathlib import Path
from typing import List

import pytest
from pydantic import ValidationError

from first_breaks.picking.task import Task, ProcessingParametersException
from first_breaks.sgy.reader import SGY


@pytest.fixture
def read_demo_sgy(demo_sgy: Path) -> SGY:
    return SGY(demo_sgy)


@pytest.mark.parametrize("tps, tps_parsed, tps_error", [
    (0, None, True),  # minimum 1 traces
    (12.1, None, True),  # only int is valid
    (13, 13, False),  # OK
    (100, 96, False),  # reduce amout of traces to amount of traces in sgy
])
@pytest.mark.parametrize("mtime, mtime_parsed, mtime_index, mtime_error", [
    (-1.1, None, 1, True),  # only positive time
    (0, 250, 1000, False),  # 0 is converted to length of sgy
    (100, 100, 400, False),  # OK
    (0.0000001, 250, 1000, False),  # time less than 1 sample, so it's the same as 0
])
# values should correlate with all values of tps
@pytest.mark.parametrize("tti, tti_parsed, tti_error", [
    (1, None, True),  # only list or int are available
    ([1, 2, 3.3], None, True),  # all elements should be int
    ([], [], False),  # nothing to inverse
    ([9, 2, 3, 6, 4, 3, 5], [1, 2, 3, 4, 5, 8], False),  # convert to python indices, sort, and remove duplicates
])
@pytest.mark.parametrize("gain, gain_parsed, gain_error", [
    (-2.1, -2.1, False),  # OK
])
@pytest.mark.parametrize("clip, clip_parsed, clip_error", [
    (0, None, True),  # != 0
    (-2, -2, True),  # Positive
    (3.3, 3.3, False),  # OK
])
def test_task_params(read_demo_sgy,
                     tps: int,
                     tps_parsed: int,
                     tps_error: bool,
                     mtime: float,
                     mtime_parsed: float,
                     mtime_index: int,
                     mtime_error: bool,
                     tti: List[int],
                     tti_parsed: List[int],
                     tti_error: bool,
                     gain: float,
                     gain_parsed: float,
                     gain_error: bool,
                     clip: float,
                     clip_parsed: float,
                     clip_error: bool) -> None:
    if any((tps_error, mtime_error, gain_error, clip_error, tti_error)):
        with pytest.raises(ValidationError):
            Task(source=read_demo_sgy,
                 traces_per_gather=tps,
                 maximum_time=mtime,
                 traces_to_inverse=tti,
                 gain=gain,
                 clip=clip)
    else:
        task = Task(source=read_demo_sgy,
                    traces_per_gather=tps,
                    maximum_time=mtime,
                    traces_to_inverse=tti,
                    gain=gain,
                    clip=clip)

        assert task.traces_per_gather == tps_parsed
        assert task.maximum_time == mtime_parsed
        assert task.maximum_time_sample == mtime_index
        assert task.traces_to_inverse == tti_parsed
        assert task.gain == gain_parsed
        assert task.clip == clip_parsed
