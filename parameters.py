'''
########################### DIRS #############################
'''

DATA_SAVE_DIR = "outputs/datasets"
TRAINED_MODEL_DIR = "outputs/trained_models"
TENSORBOARD_LOG_DIR = "outputs/tensorboard_log"
RESULTS_DIR = "outputs/results"

'''
########################### DATES #############################
'''

import datetime
NOW = datetime.datetime.now()

START_DATE = '2015-01-01'
END_DATE = '2021-10-31'

TRAIN_START_DATE = '2018-05-01'
TRAIN_END_DATE = '2020-10-01'

TEST_START_DATE = '2020-10-02'
TEST_END_DATE = '2021-10-02'

START_TRADE_DATE = "2019-01-01"

DEFAULT_DATA_COLUMNS = ["date", "tic", "close"]

'''
########################### MODEL PARAMS #############################
'''
MODEL_PARAMS = {'policy': 'MlpPolicy',
                'total_timesteps': 100000,
                "n_steps": 2048,
                "ent_coef": 0.005,
                "learning_rate": 0.0001,
                "batch_size": 128,
                'verbose': 1
                }

'''
########################### TECHNICAL INDICATORS #############################
'''

TECHNICAL_INDICATORS_LIST = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

'''
########################### ENV_ARGS #############################
'''

ENV_ARGS = {
        "hmax": 100,
        "initial_amount": 10000,
        "transaction_cost_pct": 0.001,
        "tech_indicator_list": TECHNICAL_INDICATORS_LIST,
        "reward_scaling": 1e-4
    }

'''
########################### TICKERS #############################
'''

B3_TICKER = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "ABEV3.SA",
    "B3SA3.SA",
    "BBAS3.SA",
    "BBDC4.SA",
    "SULA11.SA",
    "SUZB3.SA",
    "KLBN4.SA",
    "VIVT3.SA",
    "ELET3.SA"
]

'''
########################### REFERENCE TICKER #############################
'''
REF_TICKER = "BOVA11.SA"