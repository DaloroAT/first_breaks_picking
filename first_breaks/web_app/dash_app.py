import base64

from dash import Dash, dcc, html, Input, Output, State

from dash.exceptions import PreventUpdate

from first_breaks.const import DEMO_SGY
from first_breaks.web_app.utils import view
from first_breaks.sgy.reader import SGY
import plotly.graph_objects as go


def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


app = Dash(__name__)

MAX_SIZE = 1024 * 1024 * 8


app.layout = html.Div([
    dcc.Upload(
        id='id-123-upload-widget',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False,
        max_size=MAX_SIZE
    ),
    html.Div(id='id-123-print-widget'),
    dcc.Checklist(id='id-123-toogle-widget', options=['Use GL'], value=[]),
    # dcc.Graph(id='id-123-plot-widget', figure=blank_fig(), config={'staticPlot': True}),
dcc.Graph(id='id-123-plot-widget', figure=view(SGY(DEMO_SGY), use_gl=True)),
    dcc.Store(id='id-123-store')
])


def decode_content(content):
    data = content.encode("utf8").split(b";base64,")[1]
    return base64.decodebytes(data)


@app.callback(
    Output("id-123-store", "data"),
    Input("id-123-upload-widget", "contents"),
)
def callback_upload(contents):
    return contents


@app.callback(
    [Output("id-123-plot-widget", "figure"), Output("id-123-plot-widget", "config")],
    [Input("id-123-store", "data"), Input('id-123-toogle-widget', 'value')]
)
def callback_plot(content, toogle_gl):
    return plot(content, toogle_gl)


def plot(content, toogle_gl):
    if content is None:
        raise PreventUpdate
    raw = decode_content(content)
    sgy = SGY(raw)
    print(toogle_gl)
    return view(sgy, use_wiggle=True, use_gl=bool(toogle_gl)), {'staticPlot': False}


# df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')
#
# app.layout = html.Div([
#     html.Div([
#
#         html.Div([
#             dcc.Dropdown(
#                 df['Indicator Name'].unique(),
#                 'Fertility rate, total (births per woman)',
#                 id='xaxis-column'
#             ),
#             dcc.RadioItems(
#                 ['Linear', 'Log'],
#                 'Linear',
#                 id='xaxis-type',
#                 inline=True
#             )
#         ], style={'width': '48%', 'display': 'inline-block'}),
#
#         html.Div([
#             dcc.Dropdown(
#                 df['Indicator Name'].unique(),
#                 'Life expectancy at birth, total (years)',
#                 id='yaxis-column'
#             ),
#             dcc.RadioItems(
#                 ['Linear', 'Log'],
#                 'Linear',
#                 id='yaxis-type',
#                 inline=True
#             )
#         ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
#     ]),
#
#     dcc.Graph(id='indicator-graphic'),
#
#     dcc.Slider(
#         df['Year'].min(),
#         df['Year'].max(),
#         step=None,
#         id='year--slider',
#         value=df['Year'].max(),
#         marks={str(year): str(year) for year in df['Year'].unique()},
#
#     )
# ])
#
#
# @app.callback(
#     Output('indicator-graphic', 'figure'),
#     Input('xaxis-column', 'value'),
#     Input('yaxis-column', 'value'),
#     Input('xaxis-type', 'value'),
#     Input('yaxis-type', 'value'),
#     Input('year--slider', 'value'))
# def update_graph(xaxis_column_name, yaxis_column_name,
#                  xaxis_type, yaxis_type,
#                  year_value):
#     dff = df[df['Year'] == year_value]
#
#     fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
#                      y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
#                      hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])
#
#     fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
#
#     fig.update_xaxes(title=xaxis_column_name,
#                      type='linear' if xaxis_type == 'Linear' else 'log')
#
#     fig.update_yaxes(title=yaxis_column_name,
#                      type='linear' if yaxis_type == 'Linear' else 'log')
#
#     return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=9999)
