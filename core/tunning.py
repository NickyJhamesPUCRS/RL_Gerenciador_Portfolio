from torch import nn as nn
import optuna
# To allow big numbers (e^4...)
from typing import Union


class LoggingCallback:
    def __init__(self, threshold, trial_number, patience):
        self.threshold = threshold
        self.trial_number = trial_number
        self.patience = patience
        self.cb_list = [] #trials list

    def __call__(self, study:optuna.study, frozen_trial:optuna.Trial):
        study.set_user_attr("previous_best_value", study.best_value)

        if frozen_trial.number > self.trial_number:
            previous_best_value = study.user_attrs.get("previous_best_value", None)
            if previous_best_value * study.best_value >=0:
                if abs(previous_best_value-study.best_value) < self.threshold:
                    self.cb_list.append(frozen_trial.number)
                    if len(self.cb_list)>self.patience:
                        print('Stoping study from optuna...')
                        print(f'Trial:{frozen_trial.number} Value:{frozen_trial.value}')
                        print(f'Previous best Trial:{previous_best_value} Previous best Value:{study.best_value}')
                        study.stop()


def linear_schedule(initial_value: Union[float, str]):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float):

        return progress_remaining * initial_value

    return func


def ppo_hyperparameters(trial: optuna.Trial):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


def calculate_sharpe(dataframe):

    if dataframe['daily_return'].std != 0:
        sharpe = (252**0.5)*dataframe['daily_return'].mean()/dataframe['daily_return'].std()
        return sharpe
    else:
        return 0


def load_net_archs(hyperparameters):
    hyperparameters['net_arch'] = {"small": [dict(pi=[64, 64], vf=[64, 64])],
                               "medium": [dict(pi=[256, 256], vf=[256, 256])]}[hyperparameters['net_arch']]
    hyperparameters['activation_fn'] = {"tanh": nn.Tanh, "relu": nn.ReLU,
                                    "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[hyperparameters['activation_fn']]
    return hyperparameters


def create_study(environment, timesteps, e_trade_gym, policy, n_trials):

    def trial_until_hits(trial: optuna.Trial):
        from .drl.PPO_method import PPOLearner
        from .tensorboard_stuff import TensorboardCallback

        hyperparameters = ppo_hyperparameters(trial)
        agent = PPOLearner(env=environment, policy=policy, **hyperparameters)
        agent.learn(total_timesteps=timesteps,
                    tb_log_name="ppo",
                    callback=TensorboardCallback())

        dataframe_daily_return, dataframe_actions = agent.prediction(e_trade_gym)
        sharpe = calculate_sharpe(dataframe_daily_return)

        return sharpe

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(study_name="ppo_study", direction="maximize", sampler=sampler,
                                pruner=optuna.pruners.HyperbandPruner())

    logging_callback = LoggingCallback(threshold=1e-5, patience=20, trial_number=5)

    study.optimize(trial_until_hits, n_trials=n_trials, catch=(ValueError,), callbacks=[logging_callback])

    best_params = study.best_params
    best_params['policy'] = policy

    return load_net_archs(best_params)




