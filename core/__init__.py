import parameters


def create_folders():
    from .os_io import create_folders

    create_folders(parameters.DATA_SAVE_DIR,
                   parameters.TRAINED_MODEL_DIR,
                   parameters.TENSORBOARD_LOG_DIR,
                   parameters.RESULTS_DIR)


def run_process_tensorboard():
    from .tensorboard_stuff import run_tensorboard_server
    return run_tensorboard_server(parameters.TENSORBOARD_LOG_DIR)


def download_data():
    from .yfinance_api import YFinanceAPI

    return YFinanceAPI(start_date=parameters.START_DATE, end_date=parameters.END_DATE, tickers=parameters.B3_TICKER)


def add_technical_indicators(dataframe):
    from .data_processing import add_technical_indicators

    return add_technical_indicators(dataframe, parameters.TECHNICAL_INDICATORS_LIST)


def add_covariance_matrix(dataframe):
    from .data_processing import add_covariance_matrix

    return add_covariance_matrix(dataframe)


def split_train_test(dataframe):
    from .data_processing import data_split

    dataframe_train = data_split(dataframe, parameters.TRAIN_START_DATE, parameters.TRAIN_END_DATE)
    dataframe_test = data_split(dataframe, parameters.TEST_START_DATE, parameters.TEST_END_DATE)
    return dataframe_train, dataframe_test


def get_baseline_stats(dataframe_daily_return):
    from .data_processing import baseline_stats
    return baseline_stats(dataframe_daily_return, parameters.REF_TICKER)


def plot_backtest(dataframe_daily_return, DRL_strat):
    from .data_processing import backtest_plot
    return backtest_plot(dataframe_daily_return, DRL_strat, parameters.REF_TICKER)


def append_date_df(dataframe):
    from .data_processing import append_date_pddf
    return append_date_pddf(dataframe, parameters.TEST_START_DATE, parameters.TEST_END_DATE)

