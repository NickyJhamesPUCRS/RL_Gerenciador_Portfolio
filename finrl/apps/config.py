'''
########################### DIRS #############################
'''

DATA_SAVE_DIR = "data/datasets"
TRAINED_MODEL_DIR = "data/trained_models"
TENSORBOARD_LOG_DIR = "data/tensorboard_log"
RESULTS_DIR = "data/results"

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
########################### MODELS PARAMETERS #############################
'''

A2C_PARAMS = {
    "n_steps": 5,
    "ent_coef": 0.01,
    "learning_rate": 0.0007
}

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.0001,
    "batch_size": 128,
}

DDPG_PARAMS = {
    "batch_size": 128,
    "buffer_size": 50000,
    "learning_rate": 0.001
}

TD3_PARAMS = {
    "batch_size": 100,
    "buffer_size": 1000000,
    "learning_rate": 0.001
}

SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512
}

RLlib_PARAMS = {
    "lr": 5e-5,
    "train_batch_size": 500,
    "gamma": 0.99
}

'''
########################### CHOOSED MODEL #############################
'''
#
# CHOOSED_MODEL = {'log_name': 'ppo',
#                  'total_timesteps': 80000,
#                  'model_kwargs': PPO_PARAMS}

CHOOSED_MODEL = {'total_timesteps': 100000,
                 'model_kwargs': PPO_PARAMS}

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

