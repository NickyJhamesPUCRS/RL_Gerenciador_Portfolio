def plot_drl_min_var_DJIA(dataframe_daily_return, cumpod, dji_cumpod, min_var_cumpod):
    import plotly.graph_objs as go
    import pandas as pd

    def PlotFigure(traces):
        fig = go.Figure()
        for trace in traces:
            fig.add_trace(trace)

        fig.update_layout(
            legend=dict(
                x=0, y=1,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=15,
                    color="black"
                ),
                bgcolor="White",
                bordercolor="white",
                borderwidth=2),
            title={
                'y':0.85,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            paper_bgcolor='rgba(1,1,0,0)',
            plot_bgcolor='rgba(1, 1, 0, 0)',
            yaxis_title="Cumulative Return",
            xaxis={'type': 'date',
            'tick0': time_ind[0],
            'tickmode': 'linear',
            'dtick': 86400000.0 *80})

        fig.update_xaxes(showline=True, linecolor='black', showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',
                         mirror=True)
        fig.update_yaxes(showline=True, linecolor='black', showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',
                         mirror=True)
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightSteelBlue')

        fig.show()

    time_ind = pd.Series(dataframe_daily_return.date)
    trace0_portfolio = go.Scatter(x=time_ind, y=cumpod, mode='lines', name='A2C (Portfolio Allocation)')
    trace1_portfolio = go.Scatter(x=time_ind, y=dji_cumpod, mode='lines', name='DJIA')
    trace2_portfolio = go.Scatter(x=time_ind, y=min_var_cumpod, mode='lines', name='Min-Variance')

    traces = [trace0_portfolio, trace1_portfolio, trace2_portfolio]

    PlotFigure(traces)