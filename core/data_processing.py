'''
########################### Pre-Processing #############################
'''

def add_technical_indicators(dataframe, tecnical_indicators):
    import pandas as pd
    from stockstats import StockDataFrame as Sdf
    def clean_data(dataframe):
        df = dataframe.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        return df

    def add_technical_indicator(data, tecnical_indicators):
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        print("Getting Default Stock Dataframe")
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tecnical_indicators:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.tic == unique_ticker[i]][
                        "date"
                    ].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
            )
        df = df.sort_values(by=["date", "tic"])
        return df

    dataframe = clean_data(dataframe)
    dataframe = add_technical_indicator(dataframe, tecnical_indicators)
    print("Technical indicators added to the Stock!")

    return dataframe.fillna(method="ffill").fillna(method="bfill")


def add_covariance_matrix(dataframe):
    import pandas as pd

    dataframe = dataframe.sort_values(['date', 'tic'], ignore_index=True)
    dataframe.index = dataframe.date.factorize()[0]

    cov_list = []
    return_list = []

    lookback = 252
    for i in range(lookback, len(dataframe.index.unique())):
        data_lookback = dataframe.loc[i - lookback:i, :]
        price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)

        covs = return_lookback.cov().values
        cov_list.append(covs)

    df_cov = pd.DataFrame({'date': dataframe.date.unique()[lookback:], 'cov_list': cov_list, 'return_list': return_list})
    dataframe = dataframe.merge(df_cov, on='date')
    return dataframe.sort_values(['date', 'tic']).reset_index(drop=True)


def data_split(df, start, end):
    data = df[(df["date"] >= start) & (df["date"] < end)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data["date"].factorize()[0]
    return data

'''
########################### Pos-Processing #############################
'''

def port_min_variance(dataframe, dataframe_test):
    from pypfopt.efficient_frontier import EfficientFrontier
    import pandas as pd
    import numpy as np

    unique_tic = dataframe_test.tic.unique()
    unique_trade_date = dataframe_test.date.unique()

    portfolio = pd.DataFrame(index=range(1), columns=unique_trade_date)
    initial_capital = 1000000
    portfolio.loc[0, unique_trade_date[0]] = initial_capital

    for i in range(len(unique_trade_date) - 1):
        df_temp = dataframe[dataframe.date == unique_trade_date[i]].reset_index(drop=True)
        df_temp_next = dataframe[dataframe.date == unique_trade_date[i + 1]].reset_index(drop=True)
        # Sigma = risk_models.sample_cov(df_temp.return_list[0])
        # calculate covariance matrix
        Sigma = df_temp.return_list[0].cov()
        # portfolio allocation
        ef_min_var = EfficientFrontier(None, Sigma, weight_bounds=(0, 0.1))
        # minimum variance
        raw_weights_min_var = ef_min_var.min_volatility()
        # get weights
        cleaned_weights_min_var = ef_min_var.clean_weights()

        # current capital
        cap = portfolio.iloc[0, i]
        # current cash invested for each stock
        current_cash = [element * cap for element in list(cleaned_weights_min_var.values())]
        # current held shares
        current_shares = list(np.array(current_cash)
                              / np.array(df_temp.close))
        # next time period price
        next_price = np.array(df_temp_next.close)
        ##next_price * current share to calculate next total account value
        portfolio.iloc[0, i + 1] = np.dot(current_shares, next_price)

    portfolio = portfolio.T
    portfolio.columns = ['account_value']

    return portfolio

'''
########################### Strategies #############################
'''


def convert_daily2pyfolio(df):
    import pandas as pd
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    ts = pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)
    return ts


def get_daily_return(df):
    from copy import deepcopy
    import pandas as pd
    df = deepcopy(df)
    df["daily_return"] = df['close'].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


def stats(account_value):
    from pyfolio import timeseries

    dr_test = get_daily_return(account_value)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print(perf_stats_all)
    return perf_stats_all


def backtest_stats(dataframe_daily_return):
    from pyfolio import timeseries
    DRL_strat = convert_daily2pyfolio(dataframe_daily_return)

    return DRL_strat, timeseries.perf_stats(returns=DRL_strat,
                                factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")


def baseline_stats(dataframe_daily_return, ref_ticker):
    from .yfinance_api import YFinanceAPI
    baseline_dataframe = YFinanceAPI(start_date=dataframe_daily_return.loc[0, 'date'],
                                     end_date=dataframe_daily_return.loc[len(dataframe_daily_return) - 1, 'date'],
                                     tickers=[ref_ticker])

    return stats(baseline_dataframe)


def backtest_plot(dataframe_daily_return, DRL_strat, ref_ticker):
    import pyfolio
    from .yfinance_api import YFinanceAPI

    baseline_df = YFinanceAPI(start_date=dataframe_daily_return.loc[0, 'date'],
                              end_date=dataframe_daily_return.loc[len(dataframe_daily_return) - 1, 'date'],
                              tickers=[ref_ticker])

    baseline_returns = get_daily_return(baseline_df)

    with pyfolio.plotting.plotting_context(font_scale=1.1):
        return baseline_returns, pyfolio.create_full_tear_sheet(returns=DRL_strat,
                                                                benchmark_rets=baseline_returns, set_context=False)

