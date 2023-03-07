import base64
import re
from pathlib import Path
from typing import Union

import numpy as np
from plotly import graph_objects as go
from plotly.graph_objs import Layout
from plotly_resampler import FigureResampler

from first_breaks.sgy.reader import SGY


def markdown_images(markdown: str):
    # example image markdown:
    # ![Test image](images/test.png "Alternate text")
    images = re.findall(r'(!\[(?P<image_title>[^\]]+)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))', markdown)
    return images


def img_to_bytes(img_path: Path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path: Path, img_alt: str):
    img_format = Path(img_path).suffix
    img_html = f'<img src="data:image/{img_format.lower()};base64,{img_to_bytes(img_path)}" alt="{img_alt}" style="max-width: 100%;">'

    return img_html


def markdown_insert_images(markdown: str) -> str:
    images = markdown_images(markdown)

    for image in images:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if Path(image_path).exists():
            markdown = markdown.replace(image_markdown, img_to_html(image_path, image_alt))
    return markdown


def read_markdown(fname: Union[Path, str]) -> str:
    with open(fname) as f:
        intro = f.read()
    return markdown_insert_images(intro)


# def view(sgy: SGY, amplification: float = 1, clip: float = 0.9, height: int = 800, grid: bool = True, use_wiggle: bool = True, use_gl: bool = True) -> go.Figure:
#     np.random.seed(1)
#     shift = 1
#     print(use_gl)
#     plot_func = go.Scattergl if use_gl else go.Scatter
#     # traces = np.array(sgy.read()[:50, 20][:, None], dtype=np.float32)
#
#     traces = sgy.read()
#     traces = np.array(traces, dtype=np.float32)
#     traces = traces / np.mean(np.abs(traces), 0)
#
#     traces[0, :] = 0
#     traces[-1, :] = 0
#
#     traces *= amplification
#
#     traces[np.abs(traces) > clip] = clip
#
#     num_samples, num_traces = traces.shape
#     t = np.arange(0, num_samples) * sgy.dt / 1e3
#     layout = Layout(
#         paper_bgcolor='rgb(255, 255, 255)',
#         plot_bgcolor='rgb(255, 255, 255)',
#         hovermode=False,
#         # xaxis={'range': [- shift, shift * (num_traces + 1)]},
#         # yaxis={'range': [t[0], t[-1]]},
#         height=height
#     )
#     # fig = go.Figure(layout=layout)
#     fig = FigureResampler(go.Figure(layout=layout))
#
#     # fig = go.Figure()
#     # fig = FigureResampler(go.Figure())
#
#     if use_wiggle:
#         for idx in range(traces.shape[1]):
#             trace_shifted = traces[:, idx] + idx * shift
#             # trace_shifted = np.random.normal(size=(len(t)))
#             # trace_shifted[0] = 0.0
#             # trace_shifted[-1] = 0.0
#             # t[0] = np.nan
#             # t[-1] = np.nan
#             fig.add_trace(
#                 plot_func(fill='toself', line=dict(color='rgb(0, 0, 0)'), fillcolor='rgb(0, 0, 0)', mode='lines',
#                           # x = t, y = trace_shifted,
#                           orientation='v'),
#                 # hf_x=trace_shifted, hf_y=t
#                 hf_x=t, hf_y=trace_shifted
#             )
#             # x = trace_shifted, y = t,
#             # fig.add_trace(
#             #     plot_func(
#             #         x=[idx * shift, idx * shift, max(trace_shifted), max(trace_shifted)],
#             #         y=[t[0], t[-1], t[-1], t[0]],
#             #         fill='toself',
#             #         fillcolor='rgb(255, 255, 255)',
#             #         line=dict(color='rgb(255, 255, 255)'),
#             #         mode='lines',
#             #         showlegend=False,
#             #         orientation='h'),
#             #     # hf_x=[idx * shift, idx * shift, max(trace_shifted), max(trace_shifted)],
#             #     # hf_y=[t[0], t[-1], t[-1], t[0]]
#             # )
#
#             fig.add_trace(
#                 plot_func(line=dict(color='rgb(0, 0, 0)'), mode='lines', showlegend=False,
#                           # x=t, y=trace_shifted,
#                           orientation='v'
#                           ),
#                 # hf_x=trace_shifted, hf_y=t
#                 hf_x=t, hf_y=trace_shifted
#             )
#
#     t_tick = np.repeat(np.array([t[0], t[-2], t[-2]])[:, None], num_traces, axis=1)
#     t_tick[-1, :] = np.nan
#     x_tick = shift * np.arange(stop=num_traces, dtype=float)[None, :]
#     x_tick = np.repeat(x_tick, len(t_tick), axis=0)
#     x_tick[-1, :] = np.nan
#
#     alpha_tick = 50 if grid else 0
#     # fig.add_trace(
#     #     plot_func(x=x_tick.flatten(order='F').tolist(),
#     #               y=t_tick.flatten(order='F').tolist(),
#     #               line=dict(color=f'rgba(230, 230, 230, {alpha_tick})'), mode='lines',
#     #               showlegend=False)
#     # )
#
#     traces_shifted = traces + shift * np.arange(stop=num_traces)[None, :]
#     traces_shifted[-1, :] = np.nan
#
#     t_arr = np.repeat(t[:, None], num_traces, axis=1)
#     t_arr[-1, :] = np.nan
#
#     # fig.add_trace(
#     #     plot_func(x=traces_shifted.flatten(order='F').tolist(), y=t_arr.flatten(order='F').tolist(),
#     #               line=dict(color='rgb(0, 0, 0)'), mode='lines',
#     #               showlegend=False)
#     # )
#
#     # fig.update_traces(showlegend=False)
#     # fig.update_layout(xaxis={'fixedrange': False, "range": [-shift, num_traces * shift], 'side': 'top'},
#     #                   yaxis={'fixedrange': False, "range": [min(t), max(t)], 'autorange': 'reversed', "title_text": "ms"},
#     #                   showlegend=False)
#
#     return fig


def view(sgy: SGY, amplification: float = 1, clip: float = 0.9, height: int = 800, grid: bool = True, use_wiggle: bool = True, use_gl: bool = True) -> go.Figure:
    np.random.seed(1)
    shift = 1
    print(use_gl)
    plot_func = go.Scattergl if use_gl else go.Scatter
    # traces = np.array(sgy.read()[:50, 20][:, None], dtype=np.float32)

    traces = sgy.read()
    traces = np.array(traces, dtype=np.float32)
    traces = traces / np.mean(np.abs(traces), 0)

    traces[0, :] = 0
    traces[-1, :] = 0

    traces *= amplification

    traces[np.abs(traces) > clip] = clip

    num_samples, num_traces = traces.shape
    t = np.arange(0, num_samples) * sgy.dt / 1e3
    layout = Layout(
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255, 255, 255)',
        hovermode=False,
        xaxis={'range': [- shift, shift * (num_traces + 1)]},
        yaxis={'range': [t[0], t[-1]]},
        # height=height
    )
    fig = go.Figure(layout=layout)
    # fig = FigureResampler(go.Figure(layout=layout))
    # fig = go.Figure()

    if use_wiggle:
        for idx in range(traces.shape[1]):
            trace_shifted = traces[:, idx] + idx * shift
            # trace_shifted = np.random.normal(size=(len(t)))
            # trace_shifted[0] = 0.0
            # trace_shifted[-1] = 0.0
            # t[0] = np.nan
            # t[-1] = np.nan
            fig.add_trace(
                plot_func(fill='toself', line=dict(color='rgb(0, 0, 0)'), fillcolor='rgb(0, 0, 0)', mode='lines',
                          x = trace_shifted, y = t,
                          orientation='h'),
                # hf_x=trace_shifted, hf_y=t
            )
            # x = trace_shifted, y = t,
            # fig.add_trace(
            #     plot_func(
            #         x=[idx * shift, idx * shift, max(trace_shifted), max(trace_shifted)],
            #         y=[t[0], t[-1], t[-1], t[0]],
            #         fill='toself',
            #         fillcolor='rgb(255, 255, 255)',
            #         line=dict(color='rgb(255, 255, 255)'),
            #         mode='lines',
            #         showlegend=False,
            #         orientation='h'),
            #     # hf_x=[idx * shift, idx * shift, max(trace_shifted), max(trace_shifted)],
            #     # hf_y=[t[0], t[-1], t[-1], t[0]]
            # )

            fig.add_trace(
                plot_func(line=dict(color='rgb(0, 0, 0)'), mode='lines', showlegend=False,
                          x=trace_shifted, y=t,
                          orientation='h'
                          ),
                # hf_x=trace_shifted, hf_y=t
            )

    t_tick = np.repeat(np.array([t[0], t[-2], t[-2]])[:, None], num_traces, axis=1)
    t_tick[-1, :] = np.nan
    x_tick = shift * np.arange(stop=num_traces, dtype=float)[None, :]
    x_tick = np.repeat(x_tick, len(t_tick), axis=0)
    x_tick[-1, :] = np.nan

    alpha_tick = 50 if grid else 0
    # fig.add_trace(
    #     plot_func(x=x_tick.flatten(order='F').tolist(),
    #               y=t_tick.flatten(order='F').tolist(),
    #               line=dict(color=f'rgba(230, 230, 230, {alpha_tick})'), mode='lines',
    #               showlegend=False)
    # )

    traces_shifted = traces + shift * np.arange(stop=num_traces)[None, :]
    traces_shifted[-1, :] = np.nan

    t_arr = np.repeat(t[:, None], num_traces, axis=1)
    t_arr[-1, :] = np.nan

    # fig.add_trace(
    #     plot_func(x=traces_shifted.flatten(order='F').tolist(), y=t_arr.flatten(order='F').tolist(),
    #               line=dict(color='rgb(0, 0, 0)'), mode='lines',
    #               showlegend=False)
    # )

    fig.update_traces(showlegend=False)
    fig.update_layout(xaxis={'fixedrange': False, "range": [-shift, num_traces * shift], 'side': 'top'},
                      yaxis={'fixedrange': False, "range": [min(t), max(t)], 'autorange': 'reversed', "title_text": "ms"},
                      showlegend=False)
    return fig
