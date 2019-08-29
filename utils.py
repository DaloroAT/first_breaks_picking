import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt


def sinc_interp(x, t_prev, t_new):
    shape_x = np.shape(x)
    period = t_prev[1] - t_prev[0]

    if len(shape_x) == 1:
        t_prev = np.reshape(t_prev, (1, np.size(t_prev)))
        t_new = np.reshape(t_new, (1, np.size(t_new)))
        time_matrix = np.tile(t_new, (len(t_prev), 1)) - np.tile(t_prev.transpose(), (1, len(t_new)))
        return np.dot(x, np.sinc(time_matrix / period))
    elif shape_x[0] == 1 and shape_x[1] > 1:
        t_prev = np.reshape(t_prev, (1, np.size(t_prev)))
        t_new = np.reshape(t_new, (1, np.size(t_new)))
        time_matrix = np.tile(t_new, (len(t_prev), 1)) - np.tile(t_prev.transpose(), (1, len(t_new)))
        return np.dot(x, np.sinc(time_matrix / period))
    elif shape_x[0] > 1 and shape_x[1] == 1:
        t_prev = np.reshape(t_prev, (np.size(t_prev), 1))
        t_new = np.reshape(t_new, (np.size(t_new), 1))
        time_matrix = np.tile(t_new, (1, len(t_prev))) - np.tile(t_prev.transpose(), (len(t_new), 1))
        return np.dot(np.sinc(time_matrix / period), x)


def sinc_interp_factor(x, factor):
    num_elem = np.max(np.shape(x))
    t_prev = np.linspace(0, num_elem - 1, num_elem)
    t_new = np.linspace(0, num_elem - 1, (num_elem - 1) * factor + 1)
    return sinc_interp(x, t_prev, t_new)


def plotseis(data, picking=None,
             normalizing='entire',
             clip=0.9,
             ampl=1,
             patch=True,
             colorseis=False,
             wiggle=True,
             dt=1):

    num_time, num_trace = np.shape(data)

    if normalizing == 'indiv':
        norm_factor = np.max(np.abs(data), axis=0)
    elif normalizing == 'entire':
        norm_factor = np.tile(np.max(np.abs(data)), (1, num_trace))
    elif np.size(normalizing) == 1 and not None:
        norm_factor = np.tile(normalizing, (1, num_trace))
    elif np.size(normalizing) == num_trace:
        norm_factor = np.reshape(normalizing, (1, num_trace))
    else:
        raise ValueError('Wrong value of "normalizing"')

    data = data / norm_factor * ampl

    mask_overflow = np.abs(data) > clip
    data[mask_overflow] = np.sign(data[mask_overflow]) * clip

    data_time = np.tile((np.arange(num_time) + 1)[:, np.newaxis], (1, num_trace)) * dt

    fig, ax = plt.subplots()

    plt.xlim((0, num_trace + 1))
    plt.ylim((0, num_time * dt))
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    if wiggle:
        data_to_wiggle = data + (np.arange(num_trace) + 1)[np.newaxis, :]

        wiggle_param = {'color': (0, 0, 0)}
        ax.plot(data_to_wiggle, data_time, **wiggle_param)

    if colorseis:
        colorseis_param = {'aspect': 'auto',
                           'interpolation': 'bilinear',
                           'alpha': 1,
                           'extent': (-0.5, num_trace + 2 - 0.5, (num_time - 0.5) * dt, -0.5 * dt),
                           'cmap': 'gray'}
        ax.imshow(data, **colorseis_param)

    if patch:
        patch_param = {'closed': True,
                       'facecolor': 'black',
                       'edgecolor': None}
        data_to_patch = data
        data_to_patch[data_to_patch < 0] = 0

        for k_trace in range(num_trace):
            patch_data = ((data_to_patch[:, k_trace] + k_trace + 1)[:, np.newaxis],
                          data_time[:, k_trace][:, np.newaxis])
            patch_data = np.hstack(patch_data)

            head = np.array((k_trace + 1, 0))[np.newaxis, :]
            tail = np.array((k_trace + 1, num_time * dt))[np.newaxis, :]
            patch_data = np.vstack((head, patch_data, tail))

            polygon = Polygon(patch_data, **patch_param)
            ax.add_patch(polygon)

    if picking is not None:
        picking_param = {'linewidth': 1,
                         'color': 'red'}
        ax.plot(np.arange(num_trace) + 1, picking * dt, **picking_param)

    plt.show()
    return fig

