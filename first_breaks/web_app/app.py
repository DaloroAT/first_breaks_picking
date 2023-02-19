import sys

import pandas as pd
from first_breaks.sgy.reader import SGY
from first_breaks.utils.debug import Performance
from first_breaks.utils.visualizations import plotseis
import plotly.graph_objects as go
import numpy as np

import streamlit as st
from plotly.graph_objs import Layout
from streamlit import runtime
from streamlit.web import cli


st.set_page_config(layout="wide", page_title="seis")

# st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
#         width: 500px;
#     }
#     [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
#         width: 500px;
#         margin-left: -500px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


def view(sgy: SGY, amplification: float = 1, clip: float = None, height: int = 100, grid: bool = True):
    shift = 1
    plot_func = go.Scattergl

    traces = sgy.read()
    traces = traces / np.mean(np.abs(traces), 0)

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
        height=height,
    )
    fig = go.Figure(layout=layout)

    t_tick = np.repeat(np.array([t[0], t[-2], t[-2]])[:, None], num_traces, axis=1)
    t_tick[-1, :] = np.nan
    x_tick = shift * np.arange(stop=num_traces, dtype=float)[None, :]
    x_tick = np.repeat(x_tick, len(t_tick), axis=0)
    x_tick[-1, :] = np.nan

    alpha_tick = 50 if grid else 0
    fig.add_trace(
        plot_func(x=x_tick.flatten(order='F').tolist(),
                  y=t_tick.flatten(order='F').tolist(),
                  line=dict(color=f'rgba(230, 230, 230, {alpha_tick})'), mode='lines',
                  showlegend=False)
    )

    traces_shifted = traces + shift * np.arange(stop=num_traces)[None, :]
    traces_shifted[-1, :] = np.nan

    t_arr = np.repeat(t[:, None], num_traces, axis=1)
    t_arr[-1, :] = np.nan

    fig.add_trace(
        plot_func(x=traces_shifted.flatten(order='F').tolist(), y=t_arr.flatten(order='F').tolist(),
                  line=dict(color='rgb(0, 0, 0)'), mode='lines',
                  showlegend=False)
    )

    fig.update_traces(showlegend=False)
    fig.update_layout(xaxis={'fixedrange': False, "range": [-shift, num_traces * shift], 'side': 'top'},
                      yaxis={'fixedrange': False, "range": [min(t), max(t)], 'autorange': 'reversed', "title_text": "ms"},
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def app():

    file = st.sidebar.file_uploader('SGY file', type=['.sgy', '.segy'])

    if file is not None:
        with open(file.name, 'wb+') as f:
            f.write(file.getvalue())
        try:
            sgy = SGY(file.name)
        except Exception as e:
            st.exception(e)
            return
    # else:
    #     return

    with st.sidebar.expander("Visualization settings"):
        amplification = st.number_input("Amplification", value=1., step=0.2, format="%.1f")
        clip = st.number_input("Clip", value=1., min_value=0., step=0.2, format="%.1f")
        height = st.number_input("Height", format="%d", step=50, min_value=800, max_value=2000)
        grid = st.checkbox("Traces grid", value=True)

    with st.sidebar.expander("First breaks picking"):
        traces_per_shot = st.number_input("Traces per shot", format="%d", step=1, min_value=12)

    sgy = SGY('real_gather.sgy')

    tab_view, tab_traces_headers, tab_general_headers = st.tabs(["View", "Trace headers", "Gather headers"])

    headers_to_remove = {'textual_file_header', 'unassigned1', 'unassigned2'}
    gen_headers = {k: [v] for k, v in sgy._general_headers.items() if k not in headers_to_remove}
    df_general = pd.DataFrame(data=gen_headers)

    df_traces = sgy._traces_headers_df

    with tab_view:
        view(sgy, amplification=amplification, clip=clip, height=height, grid=grid)

    with tab_traces_headers:
        filter_by_category = st.multiselect("Filter headers", list(df_traces.columns))
        if filter_by_category:
            df_traces = df_traces[filter_by_category]

        df_traces.index += 1
        st.dataframe(df_traces, use_container_width=True, height=height)

    with tab_general_headers:
        st.table(df_general)


if __name__ == '__main__':
    if runtime.exists():
        app()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(cli.main())
