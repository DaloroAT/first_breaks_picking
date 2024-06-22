import numpy as np
import matplotlib.pyplot as plt

from first_breaks.exports.export_picks import export_to_sgy
from first_breaks.picking.picker_onnx import PickerONNX
from first_breaks.picking.picks import Picks
from first_breaks.picking.task import Task
from first_breaks.picking.utils import preprocess_gather
from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import download_demo_sgy, generate_color
from first_breaks.utils.visualizations import plotseis


def savgol_coeffs(window_length, polyorder, deriv=0):
    half_window = (window_length - 1) // 2
    A = np.zeros((window_length, polyorder + 1))
    for i in range(window_length):
        for j in range(polyorder + 1):
            A[i, j] = (i - half_window) ** j
    ATA = np.dot(A.T, A)
    AT = np.linalg.pinv(ATA)
    B = np.dot(AT, A.T)
    coeffs = B[deriv]
    return coeffs


def apply_savgol_filter(data, window_length, polyorder, deriv=0):
    coeffs = savgol_coeffs(window_length, polyorder, deriv)

    print(coeffs)
    half_window = (window_length - 1) // 2
    pad_mode = "reflect"

    padded_data = np.pad(data, ((half_window, half_window), (0, 0)), mode=pad_mode)

    filtered_data = np.apply_along_axis(
        lambda m: np.convolve(m, coeffs, mode="valid"), axis=0, arr=padded_data
    )
    return filtered_data


# # Example seismogram (replace with your data)
# seismogram = np.array([
#     # Your seismic data here
# ])
#
# # Parameters for Savitzky-Golay filter
# window_length = 11  # Choose an appropriate window length (must be odd)
# polyorder = 3      # Polynomial order
#
# # Apply Savitzky-Golay filter to smooth the data along the N dimension
# smoothed_seismogram = apply_savgol_filter(seismogram, window_length, polyorder)
#
# # Compute the first derivative using Savitzky-Golay filter
# first_derivative_seismogram = apply_savgol_filter(seismogram, window_length, polyorder, deriv=1)
#
# # Example initial picks from neural network (replace with your actual picks)
# initial_picks = np.array([/* your initial picks here */])
#
# # Refine the picks using the tangent point and intersection method
# refined_picks = []
# for trace in range(seismogram.shape[0]):
#     trace_picks = []
#     for pick in initial_picks[trace]:
#         tangent_point, intersection = find_tangent_and_intersection(seismogram[trace], first_derivative_seismogram[trace])
#         trace_picks.append(intersection)
#     refined_picks.append(trace_picks)
#
# # Convert refined_picks to a 2D array
# refined_picks = np.array(refined_picks)
#
# # Visualize results for a specific trace
# trace_idx = 0  # Change as needed
# plt.figure(figsize=(12, 6))
# plt.plot(seismogram[trace_idx], label='Original Seismic Trace')
# plt.plot(smoothed_seismogram[trace_idx], label='Smoothed Seismic Trace', linestyle='dashed')
# plt.plot(first_derivative_seismogram[trace_idx], label='First Derivative', linestyle='dotted')
# plt.scatter(initial_picks[trace_idx], seismogram[trace_idx][initial_picks[trace_idx]], color='red', label='Initial Picks')
# plt.scatter(refined_picks[trace_idx], seismogram[trace_idx][refined_picks[trace_idx].astype(int)], color='green', label='Refined Picks', marker='x')
# plt.legend()
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.title('Seismic Trace and Picks Refinement')
# plt.show()


# sgy = SGY(download_demo_sgy())
# picker = PickerONNX(model_path="fb_heatmap_afc03594f49b88ea61b5cf6ba8245be4.onnx", batch_size=3)
# task = Task(source=sgy, traces_per_gather=12, maximum_time=100)
# task = picker.process_task(task)
#
# plt.imshow(task.picks.heatmap)
# plt.show()
# print(task.picks.picks_in_mcs)
# sgy = SGY("with_picks.sgy")
# picks = Picks(
#     values=sgy.read_custom_trace_header(236, "i"), unit="mcs", dt_mcs=sgy.dt_mcs
# )
# print(picks.picks_in_mcs)
# print(task.picks.heatmap.max(), task.picks.heatmap.min())
#
# np.save("heatmap.npy", task.picks.heatmap)
#
#
# assert False
# export_to_sgy(sgy, filename="with_picks.sgy", picks=task.picks)


def find_candidates(first_derivative, window=None, neighbor_range=1):
    """Find all extrema of the first derivative within the window using extended neighborhood checks."""
    if window is None:
        start = 0
        end = len(first_derivative)
    else:
        start, end = window

    segment = first_derivative[start:end, ...]

    # Initialize masks for maxima and minima
    maxima_mask = np.ones(segment.shape, dtype=bool)
    minima_mask = np.ones(segment.shape, dtype=bool)
    ids = np.arange(len(segment))

    for shift in range(1, neighbor_range + 1):
        shifted_segment_left = segment.take(ids + shift, mode="clip", axis=0)
        shifted_segment_right = segment.take(ids - shift, mode="clip", axis=0)

        maxima_mask &= (segment > shifted_segment_left) & (
            segment > shifted_segment_right
        )
        minima_mask &= (segment < shifted_segment_left) & (
            segment < shifted_segment_right
        )

    extrema_mask = maxima_mask | minima_mask

    extrema = np.where(extrema_mask)[0] + start
    return extrema


def calculate_intersections(smoothed_trace, first_derivative, cand_points):
    """Calculate the intersection of the tangent line at the tangent point with the time axis."""
    slope = first_derivative[cand_points]
    intercept = smoothed_trace[cand_points] - slope * cand_points
    return -intercept / slope

    # if slope != 0:
    #     intersection = -intercept / slope
    # else:
    #     intersection = cand_points
    #
    # return intersection

    # slope = first_derivative[cand_points]
    # intercept = smoothed_trace[cand_points] - slope * cand_points
    # return intercept


sgy = SGY("with_picks.sgy")
picks = Picks(
    values=sgy.read_custom_trace_header(236, "i"), unit="mcs", dt_mcs=sgy.dt_mcs
)


# plotseis(sgy.read(max_sample=500), picking=picks.picks_in_samples, normalizing="trace")

data = -preprocess_gather(sgy.read(), clip=3, gain=1, normalize="trace")
heatmap = np.load("heatmap.npy")

# trace_idx = 40
# start_ms = 25
# end_ms = 42

# trace_idx = 34
# start_ms = 25
# end_ms = 42

# trace_idx = 10
# start_ms = 40
# end_ms = 80

trace_idx = 93
start_ms = 40
end_ms = 60

start_sample = sgy.units_converter.ms2index(start_ms)
end_sample = sgy.units_converter.ms2index(end_ms)
# start_sample = 60
# end_sample = 170
sub = data[start_sample:end_sample, trace_idx]
#
plt.plot(sub, color="b")
plt.plot(
    [
        picks.picks_in_samples[trace_idx] - start_sample,
        picks.picks_in_samples[trace_idx] - start_sample,
    ],
    [min(sub), max(sub)],
)
# plt.show()

window = 11
order = 3

smoothed = apply_savgol_filter(
    sub[:, None], window_length=window, polyorder=order, deriv=0
)
plt.plot(smoothed[:, 0], color="k")

first_deriv = apply_savgol_filter(
    sub[:, None], window_length=window, polyorder=order, deriv=1
)
plt.plot((2 * first_deriv[:, 0]), color="r")
cand = find_candidates(first_deriv[:, 0], neighbor_range=3)

refined = calculate_intersection(smoothed[:, 0], -first_deriv[:, 0], cand)

mask_refined = (refined < len(sub)) & (refined > 0)
refined = refined[mask_refined]
cand = cand[mask_refined]
print(cand.astype(int))
print(refined.astype(int))
print(
    (100 * heatmap[start_sample + refined.astype(int), trace_idx]).astype(int),
    "confidence",
)
print(picks.picks_in_samples[trace_idx] - start_sample)

for i in range(len(cand)):
    color_p = [c / 255 for c in generate_color()]
    plt.scatter([cand[i]], [0], color=color_p)
    plt.scatter([refined[i]], [0], marker="*", color=color_p)

# second_deriv = apply_savgol_filter(
#     sub[:, None], window_length=window, polyorder=order, deriv=2
# )

plt.plot(heatmap[start_sample:end_sample, trace_idx], linestyle="--")

# plt.plot(
#     3 * heatmap[start_sample:end_sample, trace_idx] * first_deriv[:, 0],
#     linestyle="dashdot",
# )

# plt.plot(np.abs(4 * second_deriv[:, 0]), color="g")
plt.grid()
plt.show()
