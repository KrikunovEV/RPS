from RockPaperScissors import Environment3p
from Agent import Agent
import numpy as np


rounds = 1000
players = 3
path = 'test/'
eps = 0.9
eps_low = 0.05
eps_con = (eps - eps_low) / (rounds // 2)

env = Environment3p()
agents = [
    Agent(0, env.get_obs_space(), env.get_action_space()),
    Agent(1, env.get_obs_space(), env.get_action_space()),
    Agent(2, env.get_obs_space(), env.get_action_space()),
]

obs = env.obs
for r in range(rounds):
    print(f'Round {r}, eps={eps}')

    # Make a decision
    choices = [
        np.random.choice([np.random.randint(env.get_action_space()), agents[0](obs)],
                         1, p=[eps, 1 - eps]),
        np.random.choice([np.random.randint(env.get_action_space()), agents[1](obs)],
                         1, p=[eps, 1 - eps]),
        np.random.choice([np.random.randint(env.get_action_space()), agents[2](obs)],
                         1, p=[eps, 1 - eps]),
    ]

    # Take a reward
    obs, rewards = env.action(choices)
    for id, agent in enumerate(agents):
        agent.give_reward(rewards[id])

    if r < rounds // 2:
        eps -= eps_con
    else:
        eps = eps_low

    for agent in agents:
        agent.train()

for agent in agents:
    agent.save_agent_state(path)
