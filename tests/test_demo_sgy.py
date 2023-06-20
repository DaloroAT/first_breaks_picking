from pathlib import Path

from first_breaks.sgy.reader import SGY


def test_demo_sgy_params(demo_sgy: Path) -> None:
    sgy = SGY(demo_sgy)
    assert sgy.num_traces == sgy.ntr == 96
    assert sgy.num_samples == sgy.ns == 1000
    assert sgy.dt_ms == 0.25
    assert sgy.dt_mcs == 250
    assert sgy.read().shape == (1000, 96)
