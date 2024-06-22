from pprint import pprint
from typing import Tuple

import numpy as np

from first_breaks.const import PROJECT_ROOT
from first_breaks.picking.picks import Picks
from first_breaks.sgy.reader import SGY
from first_breaks.utils.debug import Performance
from first_breaks.utils.filtering import apply_savgol_filter


class Refiner:
    def refine(self, sgy: SGY, picks: Picks):
        raise NotImplementedError


def find_extrema_mask(data: np.ndarray, neighbor_range: int = 1) -> np.ndarray:
    assert data.ndim == 2

    maxima_mask = np.ones(data.shape, dtype=bool)
    minima_mask = np.ones(data.shape, dtype=bool)
    ids = np.arange(len(data))

    for shift in range(1, neighbor_range + 1):
        shifted_data_left = data.take(ids + shift, mode="clip", axis=0)
        shifted_data_right = data.take(ids - shift, mode="clip", axis=0)

        maxima_mask &= (data > shifted_data_left) & (data > shifted_data_right)
        minima_mask &= (data < shifted_data_left) & (data < shifted_data_right)

    extrema_mask = maxima_mask | minima_mask

    return extrema_mask


def calc_intersection(
    data: np.ndarray, data_derivative: np.ndarray, tangent_points: np.ndarray
) -> np.ndarray:
    slope = data_derivative[tangent_points]
    intercept = data[tangent_points] - slope * tangent_points
    return -intercept / slope


def calc_intersection_vectorized(
    data: np.ndarray, data_derivative: np.ndarray, extrema_mask: np.ndarray
):
    assert all(arr.ndim == 2 for arr in [data, data_derivative, extrema_mask])
    assert extrema_mask.dtype == np.bool_

    extrema_indices = np.where(extrema_mask)
    row_indices, col_indices = extrema_indices

    sorted_indices = np.lexsort((row_indices, col_indices))
    sorted_row_indices = row_indices[sorted_indices]
    sorted_col_indices = col_indices[sorted_indices]

    slope = data_derivative[sorted_row_indices, sorted_col_indices]
    intercept = (
        data[sorted_row_indices, sorted_col_indices] - slope * sorted_row_indices
    )
    intersection = -intercept / slope

    to_keep = (
        (intersection >= 0)
        & (intersection < (len(data) - 1))
        & (intersection != np.inf)
    )

    intersection = intersection[to_keep]
    sorted_col_indices = sorted_col_indices[to_keep]

    unique_cols, start_indices = np.unique(sorted_col_indices, return_index=True)
    intersections = dict(zip(unique_cols, np.split(intersection, start_indices[1:])))

    return intersections


def get_band_mask(
    data: np.ndarray, band_ids: np.ndarray, width_before: int, width_after: int
) -> Tuple[np.ndarray, np.ndarray]:
    num_rows, num_cols = data.shape
    row_indices = np.arange(-width_before, width_after + 1).reshape(-1, 1) + band_ids
    row_indices_clipped = np.clip(row_indices, 0, num_rows - 1)
    return row_indices_clipped, np.arange(num_cols)


def refine_picks(
    raw_picks: np.ndarray,
    probability_heatmap: np.ndarray,
    traces2intersections,
    minimum_probability_to_refine: float = 0.9,
) -> np.ndarray:
    refined_picks = raw_picks.copy()

    for trace, intersections in traces2intersections.items():
        intersections_int = intersections.astype(int)
        prob = probability_heatmap[intersections_int, trace]
        # print(
        #     trace,
        #     intersections_int,
        #     prob,
        #     [raw_picks[trace], probability_heatmap[raw_picks[trace], trace]],
        # )
        best_candidate = np.argmax(prob)
        if prob[best_candidate] > minimum_probability_to_refine:
            refined_picks[trace] = intersections_int[best_candidate]

    return refined_picks


class MinimalPhaseRefiner(Refiner):
    def __init__(
        self,
        analyse_window_before: int = 5,
        analyse_window_after: int = 15,
        smooth_window: int = 11,
        smooth_polyorder: int = 3,
        extrema_window: int = 3,
        min_probability_to_refine: float = 0.9,
    ):
        self.analyse_window_before = analyse_window_before
        self.analyse_window_after = analyse_window_after
        self.smooth_window = smooth_window
        self.smooth_polyorder = smooth_polyorder
        self.extrema_window = extrema_window
        self.min_probability_to_refine = min_probability_to_refine

    def refine(self, sgy: SGY, picks: Picks) -> Picks:
        assert picks.heatmap is not None
        # intersection point doesn't depend on scale of data, so we don't process raw data
        raw = sgy.read()
        picks_in_samples = picks.picks_in_samples.copy()

        filtered = apply_savgol_filter(
            data=raw,
            polyorder=self.smooth_polyorder,
            window_length=self.smooth_window,
            deriv=0,
        )
        first_derivateive = apply_savgol_filter(
            data=raw,
            polyorder=self.smooth_polyorder,
            window_length=self.smooth_window,
            deriv=1,
        )

        band_mask = get_band_mask(
            data=raw,
            band_ids=picks_in_samples,
            width_before=self.analyse_window_before,
            width_after=self.analyse_window_after,
        )

        extrema = find_extrema_mask(
            data=first_derivateive[band_mask], neighbor_range=self.extrema_window
        )

        tr2intersections = calc_intersection_vectorized(
            data=filtered[band_mask],
            data_derivative=-first_derivateive[band_mask],
            extrema_mask=extrema,
        )

        # previous intersections obtained based on band data. We need to map band intersections to initail coordintates
        tr2intersections = {
            tr: band_mask[0][inter.round().astype(int), tr]
            for tr, inter in tr2intersections.items()
        }

        refined_picks = refine_picks(
            raw_picks=picks_in_samples,
            probability_heatmap=picks.heatmap,
            traces2intersections=tr2intersections,
            minimum_probability_to_refine=self.min_probability_to_refine,
        )

        picks.from_samples(refined_picks)

        return picks


if __name__ == "__main__":
    sgy = SGY(PROJECT_ROOT / "with_picks.sgy")
    heatmap = np.load(PROJECT_ROOT / "heatmap.npy")
    src_picks = Picks(
        values=sgy.read_custom_trace_header(236, "i"),
        unit="mcs",
        dt_mcs=sgy.dt_mcs,
        heatmap=heatmap,
    )

    new_picks = src_picks.create_duplicate()

    refiner = MinimalPhaseRefiner()
    with Performance():
        new_picks = refiner.refine(sgy, new_picks)

    print(src_picks.picks_in_samples)
    print(new_picks.picks_in_samples)

    # num_tr = 20
    # num_samples = 20
    # window_smooth = 11
    # order = 3
    # window_extrema = 3
    # window_analyse_before = 5
    # window_analyse_after = 5
    # min_probability_to_refine = 0.9
    #
    #
    # raw = np.random.uniform(size=(num_samples, num_tr))
    # picks = np.random.randint(0, num_samples, size=num_tr).astype(int)
    # heatmap = np.random.randint(1, 3, size=raw.shape)
    #
    # with Performance():
    #     filtered = apply_savgol_filter(
    #         data=raw, polyorder=order, window_length=window_smooth, deriv=0
    #     )
    #     first_derivateive = apply_savgol_filter(
    #         data=raw, polyorder=order, window_length=window_smooth, deriv=1
    #     )
    #
    #     band_mask = get_band_mask(
    #         data=raw,
    #         band_ids=picks,
    #         width_before=window_analyse_before,
    #         width_after=window_analyse_after,
    #     )
    #
    #     extrema = find_extrema_mask(
    #         data=first_derivateive[band_mask], neighbor_range=window_extrema
    #     )
    #
    #     tr2intersections = calc_intersection_vectorized(
    #         data=filtered[band_mask],
    #         data_derivative=-first_derivateive[band_mask],
    #         extrema_mask=extrema,
    #     )
    #     # pprint(band_mask[0])
    #     # pprint(picks)
    #     # pprint(tr2intersections)
    #     # band_start = band_mask[0][0, :]
    #     # pprint(band_start)
    #     # tr2intersections = {
    #     #     tr: inter + band_start[tr] for tr, inter in tr2intersections.items()
    #     # }
    #
    #     # pprint(band_mask[0])
    #     # pprint(tr2intersections)
    #
    #     # previous intersections obtained based on band data. We need to map band intersections to initail coordintates
    #     tr2intersections = {
    #         tr: band_mask[0][inter.round().astype(int), tr]
    #         for tr, inter in tr2intersections.items()
    #     }
    #
    #     refined_picks = refine_picks(
    #         raw_picks=picks,
    #         probability_heatmap=heatmap,
    #         traces2intersections=tr2intersections,
    #         minimum_probability_to_refine=min_probability_to_refine,
    #     )

    # d = np.random.uniform(size=(20, 20))
    # der = np.random.uniform(size=(20, 20))
    #
    #
    # picks = np.array([5] * 20)
    #
    # # picks[3] = 1
    #
    # d[picks - 3 : picks + 3, :]
    #
    #
    # d[:, 1:10] = 100
    # # tang = []
    #
    # # aa = np.random.randint(0, 2, size=(3, 3)).astype(bool)
    # # print(aa)
    # # print(aa.nonzero())
    #
    # with Performance():
    #     extrema = find_extrema_mask(d)
    #
    #
    # # print(res)
    # # print(np.where(res))
    #
    # with Performance():
    #     tr2intersection = {}
    #
    #     res = np.where(extrema)
    #
    #     for i in np.unique(res[1]):
    #         extrema_tr = res[0][res[1] == i]
    #         tr2intersection[i] = calc_intersection(d[:, i], der[:, i], extrema_tr)
    #
    #
    # print(extrema.shape)
    #
    # with Performance():
    #     v = calc_intersection_vectorized(d, der, extrema)
    #
    #
    # # print(len(v), len(tr2intersection))
    # #
    # # print(v[0])
    # # print(tr2intersection[0])
    #
    #
    # assert all(np.allclose(tr2intersection[i], v[i]) for i in tr2intersection.keys())
    #
    # print(tr2intersection[5])

    # d = np.arange(10)[:, None]
    # d = np.tile(d, (1, 5))
    # picks = np.array([1, 9, 3, 4, 5])
    #
    # band_mask = get_band_mask(d, picks, 3, 2)
    #
    # print(d)
    # print(band_mask)
    # print(d[band_mask])
