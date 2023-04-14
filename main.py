import base64
import io

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, callback, Output, Input, callback_context, no_update, State
from dash.exceptions import PreventUpdate

from pca import run_pca

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1(children='PCA Analysis', style={'textAlign': 'center'}),
    dbc.Row(
        [dcc.Upload(id='upload-data',
                    children=['Drag and Drop or ',
                              html.A('Select Hyperspectral File')]
                    , style={
                'width': '50%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center'
            },
                    multiple=False,
                    max_size=-1,
                    )
         ],
        align="center"
    ),
    dbc.Row(dbc.Col([dcc.Input(id="height-input", type="number", placeholder="enter image height"),
                     dcc.Input(id="width-input", type="number", placeholder="enter image width")], align="center")),
    dbc.Row(html.Button('Run PCA', id='pca-button', n_clicks=0), align="center"),
    dbc.Row(dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id="loading-output-1")
    )),
    dbc.Row(id='spectral-graph-content')
])


@callback(
    [Output("upload-data", "children"),
     Output("loading-output-1", "children")],
    [Input("upload-data", 'contents'),
     Input("pca-button", 'n_clicks'),
     Input("height-input", "value"),
     Input("width-input", "value")],
    State('upload-data', 'filename'),
)
def render_pca(upload_file, pca_button, height, width, filename):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if upload_file is None:
        raise PreventUpdate

    if "pca-button" not in changed_id:
        return filename, no_update
    else:

        content_type, content_string = upload_file.split(",")
        decoded = base64.b64decode(content_string)
        hsi_data = np.genfromtxt(io.StringIO(decoded.decode("utf-8")), delimiter='\t')
        run_pca(hsi_data, height, width)

        global pca_im
        global wavelengths
        global hsi_im

        pca_im = np.load('output/pca_im.npy')
        hsi_im = np.load('output/hsi_im.npy')
        wavelengths = hsi_data[0]

        graphs_children = [
            dbc.Col([dcc.Dropdown([i for i in range(1, pca_im.shape[2] + 1)], 1,
                                  id='dropdown-selection'),
                     dcc.Graph(id='graph-content',
                               hoverData={'points': [{'y': 0, 'x': 0}]}
                               )], width="auto")
        ]

    return filename, graphs_children


@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_pca_im(value):
    pca_sl = pca_im[:, :, value - 1]
    return px.imshow(pca_sl)


@callback(
    Output('spectral-graph-content', 'children'),
    Input('graph-content', 'hoverData')
)
def update_spectral_graph(hoverData):
    df = pd.DataFrame({'wavelength': wavelengths,
                       'intensity': hsi_im[hoverData['points'][0]['y'], hoverData['points'][0]['x'], :]})
    fig = px.line(df, x='wavelength', y='intensity', title='Hyperspectral Data of Pixel')
    return dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
