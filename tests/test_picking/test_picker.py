from pathlib import Path

import pytest

from first_breaks.picking.picker_onnx import PickerONNX
from first_breaks.picking.task import Task
from first_breaks.sgy.reader import SGY


@pytest.mark.parametrize("traces_per_gather", [48, 90])
@pytest.mark.parametrize("maximum_time", [0.0, 100.])
@pytest.mark.parametrize("gain", [-1.5, 2])
@pytest.mark.parametrize("clip", [0.5, 2])
def test_picking(demo_sgy: Path,
                 model_onnx: Path,
                 traces_per_gather: int,
                 maximum_time: float,
                 gain: float,
                 clip: float) -> None:
    sgy = SGY(demo_sgy)
    task = Task(source=sgy, traces_per_gather=traces_per_gather, maximum_time=maximum_time, gain=gain, clip=clip)
    picker = PickerONNX(model_onnx, device='cpu', show_progressbar=False)
    task = picker.process_task(task)

    assert task.success

    assert isinstance(task.picks_in_samples, list)
    assert len(task.picks_in_samples) == sgy.num_traces
    for pick in task.picks_in_samples:
        assert isinstance(pick, int)
        assert 0 <= pick <= sgy.num_samples

    assert isinstance(task.confidence, list)
    assert len(task.confidence) == sgy.num_traces
    for conf in task.confidence:
        assert isinstance(conf, float)
        assert 0 <= conf <= 1
