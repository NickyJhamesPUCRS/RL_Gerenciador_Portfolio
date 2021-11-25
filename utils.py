def port_min_variance(dataframe, dataframe_test):
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models
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