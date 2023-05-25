from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def plotseis(
    data: np.ndarray,
    picking: Optional[np.ndarray] = None,
    add_picking: Optional[np.ndarray] = None,
    normalizing: Optional[Union[str, int]] = "indiv",
    clip: float = 0.9,
    ampl: float = 1.0,
    patch: bool = True,
    colorseis: bool = False,
    wiggle: bool = True,
    background: Optional[np.ndarray] = None,
    colorbar: bool = False,
    dt: float = 1.0,
    show: bool = True,
    dpi: int = 300,
    figsize: Tuple[int, int] = (5, 5),
) -> matplotlib.figure.Figure:

    num_time, num_trace = np.shape(data)

    if normalizing == "indiv":
        norm_factor = np.mean(np.abs(data), axis=0)
        norm_factor[np.abs(norm_factor) < 1e-9 * np.max(np.abs(norm_factor))] = 1
    elif normalizing == "entire":
        norm_factor = np.tile(np.mean(np.abs(data)), (1, num_trace))
    elif np.size(normalizing) == 1 and normalizing is not None:
        norm_factor = np.tile(normalizing, (1, num_trace))
    elif np.size(normalizing) == num_trace:
        norm_factor = np.reshape(normalizing, (1, num_trace))
    elif normalizing is None:
        norm_factor = np.ones(data.shape[1])
    else:
        raise ValueError('Wrong value of "normalizing"')

    data = data / norm_factor * ampl

    mask_overflow = np.abs(data) > clip
    data[mask_overflow] = np.sign(data[mask_overflow]) * clip

    data_time = np.tile((np.arange(num_time) + 1)[:, np.newaxis], (1, num_trace)) * dt

    fig, ax = plt.subplots(figsize=figsize)
    fig.set_dpi(dpi)

    plt.xlim((0, num_trace + 1))
    plt.ylim((0, num_time * dt))
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    if wiggle:
        data_to_wiggle = data + (np.arange(num_trace) + 1)[np.newaxis, :]

        ax.plot(data_to_wiggle, data_time, color=(0, 0, 0))

    if colorseis:
        if not (wiggle or patch):
            ax.imshow(
                data,
                aspect="auto",
                interpolation="bilinear",
                alpha=1,
                extent=(1, num_trace, (num_time - 0.5) * dt, -0.5 * dt),
                cmap="gray",
            )
        else:
            ax.imshow(
                data,
                aspect="auto",
                interpolation="bilinear",
                alpha=1,
                extent=(-0.5, num_trace + 2 - 0.5, (num_time - 0.5) * dt, -0.5 * dt),
                cmap="gray",
            )

    if patch:
        data_to_patch = data
        data_to_patch[data_to_patch < 0] = 0

        for k_trace in range(num_trace):
            patch_data = (
                (data_to_patch[:, k_trace] + k_trace + 1)[:, np.newaxis],
                data_time[:, k_trace][:, np.newaxis],
            )
            patch_data = np.hstack(patch_data)

            head = np.array((k_trace + 1, 0))[np.newaxis, :]
            tail = np.array((k_trace + 1, num_time * dt))[np.newaxis, :]
            patch_data = np.vstack((head, patch_data, tail))

            polygon = Polygon(patch_data, closed=True, facecolor="black", edgecolor=None)
            ax.add_patch(polygon)

    if picking is not None:
        picking = np.array(picking)
        ax.plot(np.arange(num_trace) + 1, picking * dt, linewidth=1, color="blue")

    if add_picking is not None:
        add_picking = np.array(add_picking)
        ax.plot(np.arange(num_trace) + 1, add_picking * dt, linewidth=1, color="green")

    if background is not None:
        bg = ax.imshow(
            background,
            aspect="auto",
            extent=(0.5, num_trace + 1 - 0.5, (num_time - 0.5) * dt, -0.5 * dt),
            cmap="YlOrRd",
        )

        if colorbar:
            plt.colorbar(mappable=bg)

    if show:
        plt.show()
    return fig
