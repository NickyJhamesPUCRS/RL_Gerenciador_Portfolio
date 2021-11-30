from drl.PPO_method import PPOLearner
from finrl.drl_agents.stablebaselines3.models import DRLAgent


# def create_drl_agent_model(environment, config):
#     agent = DRLAgent(env=environment)
#
#     return agent, agent.get_model(model_name=config.CHOOSED_MODEL['log_name'], model_kwargs=config.CHOOSED_MODEL['model_kwargs'])


def PPO_agent(environment, config):
    from drl import PPO_method
    agent = PPOLearner(env=environment, policy="MlpPolicy", learning_rate=0.0001, ent_coef=0.005, tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/ppo", verbose=1)
    #agent = agent.PPOAgent(env=environment, model_kwargs=config.CHOOSED_MODEL['model_kwargs'])

    agent.learn(total_timesteps=100000,
            tb_log_name="ppo",
            callback=PPO_method.TensorboardCallback())

    return agent


# def train_model(agent, model, config):
#     return agent.train_model(model=model,
#                              tb_log_name=config.CHOOSED_MODEL['log_name'],
#                              total_timesteps=config.CHOOSED_MODEL['total_timesteps'])
#
#
# def prediction(model, environment):
#     return DRLAgent.DRL_prediction(model=model,
#                         environment =environment)

def prediction(agent, environment):
    return agent.prediction(environment=environment)