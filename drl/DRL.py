from finrl.drl_agents.stablebaselines3.models import DRLAgent


def create_drl_agent_model(environment, config):
    agent = DRLAgent(env=environment)

    return agent, agent.get_model(model_name=config.CHOOSED_MODEL['log_name'], model_kwargs=config.CHOOSED_MODEL['model_kwargs'])


def PPO_agent(environment, config):
    from drl import agent
    agent = agent.PPOAgent(env=environment, model_kwargs=config.CHOOSED_MODEL['model_kwargs'])
    agent.train(total_timesteps=config.CHOOSED_MODEL['total_timesteps'])

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