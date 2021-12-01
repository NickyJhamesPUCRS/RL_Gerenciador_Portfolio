def YFinanceAPI(start_date, end_date, tickers):
    import pandas as pd
    import yfinance as yf

    dataframe = pd.DataFrame()
    for tic in tickers:
        temp_df = yf.download(tic, start=start_date, end=end_date)
        temp_df["tic"] = tic
        dataframe = dataframe.append(temp_df)
    dataframe = dataframe.reset_index()

    dataframe.columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjcp",
        "volume",
        "tic",
    ]
    dataframe["close"] = dataframe["adjcp"]
    dataframe = dataframe.drop(labels="adjcp", axis=1)

    dataframe["day"] = dataframe["date"].dt.dayofweek
    dataframe["date"] = dataframe.date.apply(lambda x: x.strftime("%Y-%m-%d"))
    dataframe = dataframe.dropna()
    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.sort_values(by=["date", "tic"]).reset_index(drop=True)

    return dataframe