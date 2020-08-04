import matplotlib
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web
import math
import plotly.graph_objs as go


def start_ml(df, forecast_out):
    #forecast_out = int(math.ceil(0.01 * len(df)))
    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    df.dropna(inplace=True)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(confidence)
    forecast_set = clf.predict(X_lately)
    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    return df, confidence


def get_stock(stock):
    start = '1990-01-01'
    end = '2019-02-01'
    df = web.DataReader(stock, 'yahoo', start, end)
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Adj Close'] * 100.0
    df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0

    df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
    forecast_col = 'Adj Close'
    df.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(0.05 * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)

    df, confidence = start_ml(df, forecast_out)
    df = df.reset_index()

    return df, confidence


# print(get_stock('TSLA').head())
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.0/cerulean/bootstrap.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H3('Stock future prediction'),
    # Id input
    html.P([
        html.Label('ID'),
        dcc.Input(
            id='stock ticker',
            # placeholder = 'forecast_from_notebook',
            type='text',
            value='TSLA'
        )
    ]),

    # Key
    # html.P([
    #     html.Label("Select dataset"),
    #     dcc.Dropdown(id='path',
    #                  options=get_filenames('ss-forecasting', 'web_app_tutorial/training_target'),
    #                  style={'width': '400px',
    #                         'fontSize': '20px',
    #                         # 'padding-left' : '100px',
    #                         })
    # ]),
    dbc.Button('Visualise', id="Button"),
    html.Div(id='roro'),

    # dcc.Link('Go to App 1', href='/apps/app1'),
    # dcc.Link('Go to App 2', href='/apps/app2'),
    # dcc.Link('Go to App 4', href='/apps/app4'),
    # dcc.Link('Go to App 5', href='/apps/app5')

])


@app.callback(Output('roro', 'children'),
              [Input('Button', 'n_clicks')],
              [State('stock ticker', 'value')])
def update_output(Button, stock):
    print(Button)
    if Button is not None:
        # stock = value
        # print(stock)
        out, confidence = get_stock(stock)
        # print(out.head())
        #print(out.tail())
        yhat = go.Scatter(
            x=out['Date'],
            y=out['Adj Close'],
            mode='lines',
            marker={
                'color': 'rgba(0, 0, 255, 0.3)'
            },
            line={
                'width': 3
            },
            name='Adj Close',
        )

        yhat_lower = go.Scatter(
            x=out['Date'],
            y=out['Forecast'],
            mode='lines',
            marker={
                'color': 'rgba(255, 0, 0, 0.3)'
            },
            line={
                'width': 3
            },
            name='forecast',
        )

        layout = go.Layout(

            hovermode='x',

            margin={
                't': 20,
                'b': 50,
                'l': 60,
                'r': 10
            },
            legend={
                'bgcolor': 'rgba(0,0,0,0)'
            }
        )
        data = [yhat_lower, yhat]

        fig = go.Figure(data=data, layout=layout)

        children = [
            dcc.Graph(id='timeseries',
                      config={'displayModeBar': False},
                      animate=True,
                      figure=fig
                      ),
            html.H5('Confidence of the model is {}'.format(confidence)),

        ]
        return children


if __name__ == '__main__':
    # transform()
    app.run_server(host='0.0.0.0', port=3000, debug=True)
