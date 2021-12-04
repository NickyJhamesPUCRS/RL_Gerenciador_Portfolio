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

'''
TRAIN_START_DATE = '2018-05-01'
TRAIN_END_DATE = '2020-10-01'
TEST_START_DATE = '2020-10-02'
TEST_END_DATE = '2021-10-02'
'''

TRAIN_START_DATE = '2016-01-01'
TRAIN_END_DATE = '2020-12-31'

TEST_START_DATE = '2021-01-01'
TEST_END_DATE = '2021-10-30'

START_TRADE_DATE = "2019-01-01"

DEFAULT_DATA_COLUMNS = ["date", "tic", "close"]

'''
########################### MODEL PARAMS #############################
'''
'''
original
MODEL_PARAMS = {'policy': 'MlpPolicy',
                'total_timesteps': 200000,
                "n_steps": 2048,
                "ent_coef": 0.005,
                "learning_rate": 0.0002,
                "batch_size": 128,
                'verbose': 1
                }
'''
'''
[I 2021-12-03 01:19:01,222] Trial 9 finished with value: 0.14707928503337894 and parameters: 
    {'batch_size': 32, 
    'n_steps': 512, 
    'gamma': 0.9999, 
    'learning_rate': 0.0425204196357004, 
    'ent_coef': 0.00019674620838047797, 
    'clip_range': 0.2, 
    'n_epochs': 20, 
    'gae_lambda': 0.95, 
    'max_grad_norm': 0.9, 
    'vf_coef': 0.944765729882428, 
    'net_arch': 'small', 
    'activation_fn': 'tanh'}
    . Best is trial 9 with value: 0.14707928503337894.
'''
MODEL_PARAMS = {'policy': 'MlpPolicy',
                'total_timesteps': 100000,
                'n_steps': 512, 
                'gamma': 0.9999, 
                'ent_coef': 0.00019674620838047797,
                'learning_rate': 0.0425204196357004, 
                'clip_range': 0.2, 
                'batch_size': 32, 
                'verbose': 1,
                'n_epochs': 20, 
                'gae_lambda': 0.95, 
                'max_grad_norm': 0.9,
                'vf_coef': 0.944765729882428, 
                'net_arch': 'small', 
                'activation_fn': 'tanh'
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
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.0,
        "tech_indicator_list": TECHNICAL_INDICATORS_LIST,
        "reward_scaling": 1e-4
    }


'''
########################### TICKERS #############################
'''

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
    "ITSA4.SA",
    "PETR3.SA",
    "LREN3.SA",
    "RENT3.SA",
    "JBSS3.SA",
    "ELET3.SA",
    "BRFS3.SA",
    "BBDC3.SA",
    "CCRO3.SA",
    "EMBR3.SA",
    "EQTL3.SA",
    "GGBR4.SA",
    "SANB11.SA",
    "WEGE3.SA"
    ]


'''
########################### REFERENCE TICKER #############################
'''
REF_TICKER = "BOVA11.SA"

'''
########################### TUNE #############################
'''
TUNE = True
N_TRIALS = 100
TUNE_TIMESTEPS = 5000

'''
[I 2021-12-02 22:21:40,532] Trial 2 finished with value: 0.1372766069087547 and parameters: {'batch_size': 128, 'n_steps': 512, 'gamma': 0.99, 'learning_rate': 0.00012898495377182658, 'ent_coef': 6.90331310095056e-08, 'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.99, 'max_grad_norm': 1, 'vf_coef': 0.7616196153287176, 'net_arch': 'medium', 'activation_fn': 'relu'}. Best is trial 2 with value: 0.1372766069087547.
[I 2021-12-03 01:19:01,222] Trial 9 finished with value: 0.14707928503337894 and parameters: {'batch_size': 32, 'n_steps': 512, 'gamma': 0.9999, 'learning_rate': 0.0425204196357004, 'ent_coef': 0.00019674620838047797, 'clip_range': 0.2, 'n_epochs': 20, 'gae_lambda': 0.95, 'max_grad_norm': 0.9, 'vf_coef': 0.944765729882428, 'net_arch': 'small', 'activation_fn': 'tanh'}. Best is trial 9 with value: 0.14707928503337894.
'''