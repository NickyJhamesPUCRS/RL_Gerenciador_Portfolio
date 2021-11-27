def create_folders(config):
    import os

    if not os.path.exists(config.DATA_SAVE_DIR):
        os.makedirs(config.DATA_SAVE_DIR)
    if not os.path.exists(config.TRAINED_MODEL_DIR):
        os.makedirs(config.TRAINED_MODEL_DIR)
    if not os.path.exists(config.TENSORBOARD_LOG_DIR):
        os.makedirs(config.TENSORBOARD_LOG_DIR)
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)


def download_data(start_date, end_date, ticker_list):
    from finrl.neo_finrl.preprocessor.yahoodownloader import YahooDownloader

    return YahooDownloader(start_date=start_date,
                         end_date=end_date,
                         ticker_list=ticker_list).fetch_data()


def add_technical_indicators(dataframe):
    from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer

    fe = FeatureEngineer(
        use_technical_indicator=True,
        use_turbulence=False,
        user_defined_feature=False)

    return fe.preprocess_data(dataframe)


def add_covariance_matrix(dataframe):
    import pandas as pd

    dataframe = dataframe.sort_values(['date', 'tic'], ignore_index=True)
    dataframe.index = dataframe.date.factorize()[0]

    cov_list = []
    return_list = []

    # look back is one year
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


def data_split(dataframe, train_start_date, train_end_date, test_start_date, test_end_date):
    from finrl.neo_finrl.preprocessor.preprocessors import data_split
    return data_split(dataframe, train_start_date, train_end_date), data_split(dataframe, test_start_date, test_end_date)


if __name__ == '__main__':
    from finrl.apps import config
    from environment import StockPortfolioEnv
    import DRL, strats, utils, plots, tb_server

    create_folders(config)

    p = tb_server.run_tensorboard_server(config)

    try:
        dataframe = download_data(start_date=config.START_DATE, end_date=config.END_DATE, ticker_list=config.B3_TICKER)
        dataframe = add_technical_indicators(dataframe)
        dataframe = add_covariance_matrix(dataframe)

        dataframe_train, dataframe_test = data_split(dataframe,
                                                     train_start_date=config.TRAIN_START_DATE,
                                                     train_end_date=config.TRAIN_END_DATE,
                                                     test_start_date=config.TEST_START_DATE,
                                                     test_end_date=config.TEST_END_DATE)

        stock_dimension = len(dataframe_train.tic.unique())
        state_space = stock_dimension
        print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1000000,
            "transaction_cost_pct": 0.001,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4
        }

        e_train_gym = StockPortfolioEnv(df=dataframe_train, **env_kwargs)
        e_trade_gym = StockPortfolioEnv(df=dataframe_test, **env_kwargs)

        env_train, _ = e_train_gym.get_sb_env()

        agent, model = DRL.create_drl_agent_model(env_train, config)
        trained_agent = DRL.train_model(agent, model, config)
        trained_agent.save(config.TRAINED_MODEL_DIR + "/trained_{}.zip".format(config.CHOOSED_MODEL['log_name']))

        dataframe_daily_return, dataframe_actions = DRL.prediction(model, e_trade_gym)
        dataframe_daily_return.to_excel(
            config.RESULTS_DIR + "/df_daily_return_{}.xlsx".format(config.CHOOSED_MODEL['log_name']))
        dataframe_actions.to_excel(
            config.RESULTS_DIR + "/df_actions_{}.xlsx".format(config.CHOOSED_MODEL['log_name']))

        DRL_strat, perf_stats_all = strats.backtest_stats(dataframe_daily_return)
        print("==============DRL Strategy Stats===========")
        print(perf_stats_all)
        stats = strats.baseline_stats(dataframe_daily_return)
        print("==============Get Baseline Stats===========")
        print(stats)
        baseline_returns, tear_sheet = strats.backtest_plot(dataframe_daily_return, DRL_strat)

        portfolio = utils.port_min_variance(dataframe, dataframe_test)
        cumpod = (dataframe_daily_return.daily_return + 1).cumprod() - 1
        min_var_cumpod = (portfolio.account_value.pct_change() + 1).cumprod() - 1
        baseline_cumpod = (baseline_returns + 1).cumprod() - 1

        plots.plot_drl_min_var_baseline(config, dataframe_daily_return, cumpod, baseline_cumpod, min_var_cumpod)

        p.wait()

    except KeyboardInterrupt:
        try:
            p.terminate()
        except OSError:
            pass
        p.wait()