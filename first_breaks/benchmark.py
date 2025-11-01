import json
from itertools import product
from os import system
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from first_breaks.const import FIRST_BYTE
from first_breaks.desktop.graph import export_image
from first_breaks.picking.picker_onnx import PickerONNX
from first_breaks.picking.picks import Picks, PickingParameters
from first_breaks.picking.refiner import MinimalPhaseRefiner
from first_breaks.picking.task import Task
from first_breaks.sgy.reader import SGY
from first_breaks.utils.filtering import apply_savgol_filter
from first_breaks.utils.utils import as_list, calc_hash, download_and_validate_file


def download_model_with_heatmap(destination: Union[str, Path]) -> None:
    model_hash = "afc03594f49b88ea61b5cf6ba8245be4"
    model_url = "https://oml.daloroserver.com/download/seis/fb_heatmap_afc03594f49b88ea61b5cf6ba8245be4.onnx"
    download_and_validate_file(url=model_url, md5=model_hash, fname=destination)


def plot_picks_on_small_section_chunk(sgy: SGY, manual_picks: Picks, predicted_picks: Optional[Picks] = None) -> None:
    limit_for_validation = 10
    val_image = "chunk.png"
    val_sgy = SGY(source=sgy.read_traces_by_ids(list(range(limit_for_validation))), dt_mcs=sgy.dt_mcs)
    val_manual_picks = Picks(
        values=manual_picks.picks_in_mcs[:limit_for_validation],
        unit="mcs",
        dt_mcs=val_sgy.dt_mcs,
        color=(255, 0, 0),
    )
    picks = [val_manual_picks]

    if predicted_picks:
        val_predicted_picks = Picks(
            values=predicted_picks.picks_in_mcs[:limit_for_validation],
            unit="mcs",
            dt_mcs=val_sgy.dt_mcs,
            color=(0, 0, 255),
        )
        picks.append(val_predicted_picks)

    export_image(
        source=val_sgy,
        image_filename=val_image,
        picks_list=picks,
        height=1000,
        width=1000,
    )
    system(val_image)


def calc_snr10(traces: np.ndarray, picks: Picks, smooth: bool = False, symmetric: bool = True) -> List[float]:
    if smooth:
        traces = apply_savgol_filter(traces, polyorder=3, window_length=11, deriv=0)

    snr = np.ones(traces.shape[1])

    for idx, pick in enumerate(picks.picks_in_samples):
        if pick > 0:
            noise = traces[:pick, idx]
            if symmetric:
                signal_and_noise = traces[pick : pick + len(noise), idx]
            else:
                signal_and_noise = traces[pick:, idx]

            p_noise = np.mean(np.square(noise))
            p_signal_and_noise = np.mean(np.square(signal_and_noise))
            snr[idx] = (p_signal_and_noise - p_noise) / p_noise

    snr10 = np.log10(snr)
    snr10[np.isnan(snr10)] = -1000
    snr10[np.isinf(snr10)] = -2000
    snr10 = snr10.tolist()

    return snr10


def benchmark(sgy: SGY, manual_picks: Picks, predicted_picks: Picks, model_hash: Optional[str] = None):
    assert predicted_picks.created_by_nn and predicted_picks.picking_parameters is not None

    to_export = {"model_hash": model_hash, "refined": predicted_picks.refined}

    to_export["picking_parameters"] = predicted_picks.picking_parameters.model_dump()

    difference = (np.array(manual_picks.picks_in_mcs) - np.array(predicted_picks.picks_in_mcs)).astype(int).tolist()
    to_export["difference"] = difference

    confidence = as_list(predicted_picks.confidence)
    to_export["confidence"] = confidence

    traces = sgy.read()
    traces_hash = calc_hash(traces.tobytes(order="C"))
    to_export["traces_hash"] = traces_hash

    for header in ["CHAN", "SOURCE", "FFID"]:
        to_export[header] = sgy.traces_headers[header].apply(lambda x: calc_hash(str(x).encode())[:10]).tolist()

    to_export["shape"] = sgy.shape
    to_export["dt_mcs"] = sgy.dt_mcs

    # I want to analyse how picking parameters and result correlate with SNR
    to_export["SNR10"] = []
    for smooth, symmetric in product((True, False), (True, False)):
        snr10 = calc_snr10(traces, manual_picks, smooth=smooth, symmetric=symmetric)
        to_export["SNR10"].append({"smooth": smooth, "symmetric": symmetric, "values": snr10})

    return to_export


def benchmark_grid(
    sgy_filename: Union[str, Path],
    model_filename: Union[str, Path],
    report_filename: Union[str, Path],
    gain_list: List[float],
    maximum_time_list: List[float],
    traces_per_gather_list: List[int],
    saved_picks_byte_position: int,
):
    sgy_filename = Path(sgy_filename).resolve()
    assert sgy_filename.exists(), f"File {sgy_filename} not found"
    sgy = SGY(source=sgy_filename)
    print(f"SGY: {sgy_filename}; shape={sgy.shape}, dt_mcs={sgy.dt_mcs}")

    assert 1 <= saved_picks_byte_position <= 237
    saved_picks = Picks(
        values=sgy.read_custom_trace_header(saved_picks_byte_position - FIRST_BYTE, "i"),
        unit="mcs",
        dt_mcs=sgy.dt_mcs,
    )

    plot_picks_on_small_section_chunk(sgy=sgy, manual_picks=saved_picks)

    download_model_with_heatmap(model_filename)

    report_filename = Path(report_filename)
    report_filename.parent.mkdir(exist_ok=True, parents=True)

    picker = PickerONNX(model_path=model_filename, show_progressbar=True)

    to_export = {"confidence": [], "difference": [], "model_hash": picker.model_hash}

    total = len(gain_list) * len(maximum_time_list) * len(traces_per_gather_list)
    for idx, (gain, maximum_time, tps) in enumerate(product(gain_list, maximum_time_list, traces_per_gather_list)):
        task = Task(
            source=sgy,
            traces_per_gather=tps,
            maximum_time=maximum_time,
            gain=gain,
        )
        print(f"Task {idx}/{total} started (gain={gain}, max_time={maximum_time}, tps={tps}) ...", flush=True)
        task = picker.process_task(task)
        predicted_picks = task.get_result()

        confidence = as_list(predicted_picks.confidence)
        # difference between manual picks and predicted picks is anonymous and expose nothing, but allows me to compare
        # performance with different parameters
        difference_raw = (
            (np.array(saved_picks.picks_in_mcs) - np.array(predicted_picks.picks_in_mcs)).astype(int).tolist()
        )

        refined_picks = predicted_picks.create_duplicate()
        refiner = MinimalPhaseRefiner()
        refined_picks = refiner.refine(sgy=sgy, picks=refined_picks)

        difference_refined = (
            (np.array(saved_picks.picks_in_mcs) - np.array(refined_picks.picks_in_mcs)).astype(int).tolist()
        )

        to_export["confidence"].append(
            {"gain": gain, "maximum_time": maximum_time, "traces_per_gather": tps, "values": confidence}
        )
        to_export["difference"].append(
            {
                "gain": gain,
                "maximum_time": maximum_time,
                "traces_per_gather": tps,
                "refined": False,
                "values": difference_raw,
            }
        )
        to_export["difference"].append(
            {
                "gain": gain,
                "maximum_time": maximum_time,
                "traces_per_gather": tps,
                "refined": True,
                "values": difference_refined,
            }
        )

    # FILE LEVEL STATS

    # hash of traces allows me to inderstand if reports were created based on same data or different
    # without direct access to file: if 2 reports have same `traces_hash` it means that they were calculated based on
    # same traces, if not - files were different.
    # So I can understand how different parameters affect specific file analysing `difference` metric for several
    # files belongs to same `traces_hash`
    traces = sgy.read()
    traces_hash = calc_hash(traces.tobytes(order="C"))
    to_export["traces_hash"] = traces_hash

    # I would like to have anonymized base headers to better understand the number of seismic traces for each shot,
    # and the number of shots. I want to try to automate the selection of parameter `traces_per_gather` based on this.
    # I'm not interested in exact values of these headers, but rather in their distribution, so hashed values
    # are sufficient.
    for header in ["CHAN", "SOURCE", "FFID"]:
        to_export[header] = sgy.traces_headers[header].apply(lambda x: calc_hash(str(x).encode())[:10]).tolist()

    to_export["shape"] = sgy.shape
    to_export["dt_mcs"] = sgy.dt_mcs

    # I want to analyse how picking parameters and result correlate with SNR
    to_export["SNR10"] = []
    for smooth, symmetric in product((True, False), (True, False)):
        snr10 = calc_snr10(traces, saved_picks, smooth=smooth, symmetric=symmetric)
        to_export["SNR10"].append({"smooth": smooth, "symmetric": symmetric, "values": snr10})

    with open(report_filename, "w") as f:
        json.dump(to_export, f)


if __name__ == "__main__":
    sgy_filename_ = "my_data.sgy"
    model_filename_ = "fb_heatmap_afc03594f49b88ea61b5cf6ba8245be4.onnx"
    report_filename_ = "report.json"
    gain_list_ = [0.1, 0.5, 1]
    maximum_time_list_ = [100, 200]
    traces_per_gather_list_ = [12]
    saved_picks_byte_position_ = 237

    benchmark_grid(
        sgy_filename=sgy_filename_,
        model_filename=model_filename_,
        report_filename=report_filename_,
        gain_list=gain_list_,
        maximum_time_list=maximum_time_list_,
        traces_per_gather_list=traces_per_gather_list_,
        saved_picks_byte_position=saved_picks_byte_position_,
    )
