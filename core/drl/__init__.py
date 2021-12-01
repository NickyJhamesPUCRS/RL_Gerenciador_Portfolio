import parameters


def generate_env_args(stock_dimension):
    env_args = parameters.ENV_ARGS
    env_args["action_space"] = stock_dimension
    env_args["stock_dim"] = stock_dimension
    env_args["state_space"] = stock_dimension

    return env_args


def PPO_agent(environment):
    from .PPO_method import PPOLearner

    agent = PPOLearner(env=environment, policy=parameters.MODEL_PARAMS['policy'],
                       learning_rate=parameters.MODEL_PARAMS['learning_rate'],
                       ent_coef=parameters.MODEL_PARAMS['ent_coef'],
                       tensorboard_log=f"{parameters.TENSORBOARD_LOG_DIR}/ppo",
                       verbose=parameters.MODEL_PARAMS['verbose'])
    return agent


def train_agent(agent):
    from ..tensorboard_stuff import TensorboardCallback
    agent.learn(total_timesteps=parameters.MODEL_PARAMS['total_timesteps'],
                    tb_log_name="ppo",
                    callback=TensorboardCallback())

    agent.save(parameters.TRAINED_MODEL_DIR + "/trained_ppo.zip")

    return agent


def prediction(agent, environment):
    return agent.prediction(environment=environment)


def dataframe2excel(dataframe, filename):
    dataframe.to_excel(parameters.RESULTS_DIR + filename)
