# DRL models from Stable Baselines 3

import time

import numpy as np
import pandas as pd
from finrl.apps import config
from finrl.neo_finrl.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.neo_finrl.preprocessor.preprocessors import data_split
from drl.ppo import PPO_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union, Optional
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

import gym

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True


class PPOAgent:

    def __init__(self, env, model_kwargs=None):
        self.env = env

        self.model = PPO_policy(
            policy="MlpPolicy",
            env=self.env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/ppo",
            verbose=1,
            **model_kwargs
        )

    def train(self, total_timesteps=5000):
        return self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name="ppo",
            callback=TensorboardCallback(),
        )

    def prediction(self, environment):
        test_env, test_obs = environment.get_sb_env()
        """make a prediction"""
        account_memory = []
        actions_memory = []
        test_env.reset()
        for i in range(len(environment.df.index.unique())):
            action, _states = self.model.predict(test_obs)
            # account_memory = test_env.env_method(method_name="save_asset_memory")
            # actions_memory = test_env.env_method(method_name="save_action_memory")
            test_obs, rewards, dones, info = test_env.step(action)
            if i == (len(environment.df.index.unique()) - 2):
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]