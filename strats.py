def backtest_stats(dataframe_daily_return):
    from finrl.plot import convert_daily_return_to_pyfolio_ts
    from pyfolio import timeseries
    DRL_strat = convert_daily_return_to_pyfolio_ts(dataframe_daily_return)
    perf_func = timeseries.perf_stats
    return DRL_strat, perf_func(returns=DRL_strat,
                               factor_returns=DRL_strat,
                               positions=None, transactions=None, turnover_denom="AGB")


def baseline_strats(dataframe_daily_return, ref_ticker="BOVA11.SA"):
    from finrl.plot import get_baseline, backtest_stats
    baseline_dataframe = get_baseline(
        ticker=ref_ticker,
        start=dataframe_daily_return.loc[0, 'date'],
        end=dataframe_daily_return.loc[len(dataframe_daily_return) - 1, 'date'])

    return baseline_dataframe, backtest_stats(baseline_dataframe, value_col_name='close')


def backtest_plot(baseline_dataframe, DRL_strat):
    import pyfolio
    from finrl.plot import get_daily_return

    baseline_returns = get_daily_return(baseline_dataframe, value_col_name="close")

    with pyfolio.plotting.plotting_context(context='paper', font_scale=1.1):
        return baseline_returns, pyfolio.create_full_tear_sheet(returns=DRL_strat,
                                       benchmark_rets=baseline_returns, set_context=False, )

