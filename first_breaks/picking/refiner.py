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


def calc_intersection(data: np.ndarray, data_derivative: np.ndarray, tangent_points: np.ndarray) -> np.ndarray:
    slope = data_derivative[tangent_points]
    intercept = data[tangent_points] - slope * tangent_points
    return -intercept / slope


def calc_intersection_vectorized(data: np.ndarray, data_derivative: np.ndarray, extrema_mask: np.ndarray):
    assert all(arr.ndim == 2 for arr in [data, data_derivative, extrema_mask])
    assert extrema_mask.dtype == np.bool_

    extrema_indices = np.where(extrema_mask)
    row_indices, col_indices = extrema_indices

    sorted_indices = np.lexsort((row_indices, col_indices))
    sorted_row_indices = row_indices[sorted_indices]
    sorted_col_indices = col_indices[sorted_indices]

    slope = data_derivative[sorted_row_indices, sorted_col_indices]
    intercept = data[sorted_row_indices, sorted_col_indices] - slope * sorted_row_indices
    intersection = -intercept / slope

    to_keep = (intersection >= 0) & (intersection < (len(data) - 1)) & (intersection != np.inf)

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

        extrema = find_extrema_mask(data=first_derivateive[band_mask], neighbor_range=self.extrema_window)

        tr2intersections = calc_intersection_vectorized(
            data=filtered[band_mask],
            data_derivative=-first_derivateive[band_mask],
            extrema_mask=extrema,
        )

        # previous intersections obtained based on band data. We need to map band intersections to initail coordintates
        tr2intersections = {tr: band_mask[0][inter.round().astype(int), tr] for tr, inter in tr2intersections.items()}

        refined_picks = refine_picks(
            raw_picks=picks_in_samples,
            probability_heatmap=picks.heatmap,
            traces2intersections=tr2intersections,
            minimum_probability_to_refine=self.min_probability_to_refine,
        )

        picks.from_samples(refined_picks)

        return picks
