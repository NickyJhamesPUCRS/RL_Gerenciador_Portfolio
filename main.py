if __name__ == '__main__':
    '''Stock trading environment class with OpenAI gym'''
    from core.drl.environment import StockPortfolioEnv
    from core.drl import PPO_agent, generate_env_args, train_agent, prediction, dataframe2excel

    from core import download_data, add_technical_indicators, add_covariance_matrix, split_train_test, \
        create_folders, run_process_tensorboard, get_baseline_stats, plot_backtest, append_date_df

    from core.data_processing import backtest_stats, port_min_variance, port_min_variance_daily_return, merge_df
    from core.plotting import plot_drl_min_var_baseline

    create_folders()
    p = run_process_tensorboard()

    try:
        dataframe = download_data()
        dataframe = add_technical_indicators(dataframe)
        dataframe = add_covariance_matrix(dataframe)

        dataframe_train, dataframe_test = split_train_test(dataframe)

        env_kwargs = generate_env_args(len(dataframe_train.tic.unique()))

        e_train_gym = StockPortfolioEnv(df=dataframe_train, **env_kwargs)
        e_trade_gym = StockPortfolioEnv(df=dataframe_test, **env_kwargs)

        env_train, _ = e_train_gym.get_sb_env()

        agent = PPO_agent(env_train)
        trained_agent = train_agent(agent)

        dataframe_daily_return, dataframe_actions = prediction(trained_agent, e_trade_gym)

        dataframe2excel(dataframe_daily_return, "/df_daily_return_ppo.xlsx")
        dataframe2excel(dataframe_actions, "/df_actions_ppo.xlsx")

        DRL_strat, perf_stats_all = backtest_stats(dataframe_daily_return)
        perf_stats_all = append_date_df(perf_stats_all)

        stats = get_baseline_stats(dataframe_daily_return)
        stats = append_date_df(stats)

        '''
        calcula o portfolio de min-variance
        '''
        min_var_portfolio = port_min_variance_daily_return(dataframe, dataframe_test)
        stats_min_var = backtest_stats(min_var_portfolio)
        stats_minvar = append_date_df(stats_min_var[1])

        stats_all = merge_df([perf_stats_all, stats, stats_minvar], axis=1)
        stats_all.columns = ['DRL', 'Baseline', 'Min-Var']

        print(stats_all)

        dataframe2excel(stats_all, "/df_statistics_ppo_{}_{}_{}.xlsx", date="test")

        baseline_returns, tear_sheet = plot_backtest(dataframe_daily_return, DRL_strat)
        portfolio = port_min_variance(dataframe, dataframe_test)
        cumpod = (dataframe_daily_return.daily_return + 1).cumprod() - 1
        min_var_cumpod = (portfolio.account_value.pct_change() + 1).cumprod() - 1
        baseline_cumpod = (baseline_returns + 1).cumprod() - 1

        plot_drl_min_var_baseline(dataframe_daily_return, cumpod, baseline_cumpod, min_var_cumpod)

        p.wait()

    except KeyboardInterrupt:
        try:
            p.terminate()
        except OSError:
            pass
        p.wait()
