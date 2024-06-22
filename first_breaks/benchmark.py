from pathlib import Path
from typing import Union, Literal

from first_breaks.const import FIRST_BYTE
from first_breaks.desktop.graph import export_image
from first_breaks.picking.picker_onnx import PickerONNX
from first_breaks.picking.picks import Picks
from first_breaks.picking.task import Task
from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import (
    download_demo_sgy,
    download_by_url,
    download_and_validate_file,
    calc_hash,
)
import numpy as np

fname = download_demo_sgy()
gain = 1
traces_per_gather = 12
maximum_time = 100


# def predict_picks(
#     sgy: SGY,
#     picker: PickerONNX,
#     gain: float,
#     traces_per_gather: int,
#     maximum_time: float,
# ):
#     task = Task(
#         source=sgy,
#         traces_per_gather=traces_per_gather,
#         maximum_time=maximum_time,
#         gain=gain,
#     )
#     task = picker.process_task(task)
#     picks = task.get_result()


def make_report(
    filename: Union[str, Path],
    sgy: SGY,
    saved_picks_byte_position: int,
    predicted_picks: Picks,
    gain: float,
    traces_per_gather: int,
    maximum_time: float,
):
    assert 1 <= saved_picks_byte_position <= 237
    saved_picks = Picks(
        values=sgy.read_custom_trace_header(saved_picks_byte_position - FIRST_BYTE, "i"),
        unit="mcs",
        dt_mcs=sgy.dt_mcs,
    )
    to_export = {}

    # ANONYMOUS DATA BLOCK

    # difference between manual picks and predicted picks is anonymous and expose nothing
    difference = (np.array(saved_picks.picks_in_mcs) - np.array(predicted_picks.picks_in_mcs)).astype(int).tolist()
    to_export["difference"] = difference

    # hash of traces allows me to inderstand if reports were created based on same data or different
    # without direct access to file: if 2 reports have same `traces_hash` it means that they were calculated based on
    # same traces, if not - files were different.
    # So I can understand how different parameters affect specific file analysing `difference` metric for several
    # files belongs to same `traces_hash`
    traces_hash = calc_hash(sgy.read().tobytes(order="C"))
    to_export["traces_hash"] = traces_hash

    channel_hashes = sgy.traces_headers["CHAN"].apply(lambda x: calc_hash(x)[:10])
    source_hashes = sgy.traces_headers["CHAN"].apply(lambda x: calc_hash(x)[:10])

    # basic shape, not anonymsed
    num_samples, num_traces = sgy.shape


def download_model_with_heatmap(destination: Union[str, Path]):
    model_hash = "afc03594f49b88ea61b5cf6ba8245be4"
    model_url = "https://oml.daloroserver.com/download/seis/fb_heatmap_afc03594f49b88ea61b5cf6ba8245be4.onnx"
    download_and_validate_file(url=model_url, md5=model_hash, fname=destination)


if __name__ == "__main__":
    destination = "model_with_heatmap.onnx"
    download_model_with_heatmap(destination)

    # fname = Path(fname)
    # assert fname.exists(), f"File {fname.resolve()} not found"
    #
    # sgy = SGY(fname)
    #
    # print(
    #     f"SGY: fname='{fname.resolve()}', num_traces={sgy.num_traces}, num_samples={sgy.num_samples}, dt_mcs={sgy.dt_mcs}"
    # )
    #
    # picker = PickerONNX()
