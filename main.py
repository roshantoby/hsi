from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np

chicken_hsi = np.load('output/chicken_hsi_im.npy')
chicken_pca = np.load('output/chicken_pca_im.npy')
wavelengths = np.genfromtxt('input/chicken_HYSPEC.csv', delimiter='\t')[0]

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='PCA Analysis', style={'textAlign': 'center'}),
    dbc.Row([
        dbc.Col([dcc.Dropdown([i for i in range(1, chicken_pca.shape[2] + 1)], 1,
                              id='dropdown-selection'),
                dcc.Graph(id='graph-content',
                          hoverData={'points': [{'y': 0, 'x': 0}]}
                          )], width="auto"),
        dbc.Col(dcc.Graph(id='spectral-content'), width="auto")
    ])
])


@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_pca_im(value):
    pca_sl = chicken_pca[:, :, value - 1]
    return px.imshow(pca_sl)


@callback(
    Output('spectral-content', 'figure'),
    Input('graph-content', 'hoverData')
)
def update_spectral_graph(hoverData):
    df = pd.DataFrame({'wavelength': wavelengths,
                       'intensity': chicken_hsi[hoverData['points'][0]['y'], hoverData['points'][0]['x'], :]})

    return px.line(df, x='wavelength', y='intensity', title='Hyperspectral Data of Pixel')


if __name__ == '__main__':
    app.run_server(debug=True)
