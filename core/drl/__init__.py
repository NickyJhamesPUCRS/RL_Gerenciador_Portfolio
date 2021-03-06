import parameters


def generate_env_args(stock_dimension):
    env_args = parameters.ENV_ARGS
    env_args["action_space"] = stock_dimension
    env_args["stock_dim"] = stock_dimension
    env_args["state_space"] = stock_dimension

    return env_args


def PPO_agent(environment, hyperparameters):
    from .PPO_method import PPOLearner

    agent = PPOLearner(env=environment, **hyperparameters)
    return agent


def train_agent(agent, total_steps):
    from ..tensorboard_stuff import TensorboardCallback
    agent.learn(total_timesteps=total_steps,
                    tb_log_name="ppo",
                    callback=TensorboardCallback())

    agent.save(parameters.TRAINED_MODEL_DIR + "/trained_ppo.zip")

    return agent


def prediction(agent, environment):
    return agent.prediction(environment=environment)


def dataframe2excel(dataframe, filename, date=None):
    if None == date:
        dataframe.to_excel(parameters.RESULTS_DIR + filename)
    elif date == "test":
        _date = parameters.NOW.strftime("%d_%b_%Y_%H_%M_%S")
        dataframe.to_excel(parameters.RESULTS_DIR +
                           filename.format(parameters.TEST_START_DATE, parameters.TEST_END_DATE, _date))
    elif date == "train":
        _date = parameters.NOW.strftime("%d_%b_%Y_%H_%M_%S")
        dataframe.to_excel(parameters.RESULTS_DIR +
                           filename.format(parameters.TRAIN_START_DATE, parameters.TRAIN_END_DATE, _date))


def create_agent_and_train(env_train, e_trade_gym):

    if parameters.TUNE:
        from ..tunning import create_study
        hyperparameters = create_study(env_train, parameters.TUNE_TIMESTEPS,
                                       e_trade_gym, parameters.MODEL_PARAMS['policy'], parameters.N_TRIALS)
    else:
        from ..tunning import load_net_archs

        hyperparameters = {'policy': parameters.MODEL_PARAMS['policy'],
                            'learning_rate': parameters.MODEL_PARAMS['learning_rate'],
                            'ent_coef': parameters.MODEL_PARAMS['ent_coef'],
                            'tensorboard_log': f"{parameters.TENSORBOARD_LOG_DIR}/ppo",
                            'verbose': parameters.MODEL_PARAMS['verbose'],
                            'n_steps': parameters.MODEL_PARAMS['n_steps'],
                            'gamma': parameters.MODEL_PARAMS['gamma'],
                            'clip_range': parameters.MODEL_PARAMS['clip_range'],
                            'batch_size': parameters.MODEL_PARAMS['batch_size'],
                            'n_epochs': parameters.MODEL_PARAMS['n_epochs'],
                            'gae_lambda': parameters.MODEL_PARAMS['gae_lambda'],
                            'max_grad_norm': parameters.MODEL_PARAMS['max_grad_norm'],
                            'vf_coef': parameters.MODEL_PARAMS['vf_coef'],
                            'net_arch': parameters.MODEL_PARAMS['net_arch'],
                            'activation_fn': parameters.MODEL_PARAMS['activation_fn']
                           }

        hyperparameters = load_net_archs(hyperparameters)

    agent = PPO_agent(env_train, hyperparameters)
    trained_agent = train_agent(agent, parameters.MODEL_PARAMS['total_timesteps'])

    return trained_agent