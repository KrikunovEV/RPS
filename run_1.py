from RockPaperScissors import Environment3p
from Agent import Agent
import numpy as np


rounds = 10000
players = 3
path = 'test/'
eps = 0.9
eps_low = 0.01
eps_rnd = 2500
eps_con = (eps - eps_low) / eps_rnd

env = Environment3p()
agents = [
    Agent(0, env.get_obs_space(), env.get_action_space(), negotiate=True),
    Agent(1, env.get_obs_space(), env.get_action_space()),
    Agent(2, env.get_obs_space(), env.get_action_space()),
]

obs = env.obs
for r in range(rounds):
    print(f'Round {r}, eps={eps}')

    # Make a guess
    guess = np.random.choice([np.random.randint(env.get_action_space()), agents[1](obs)],
                             1, p=[eps, 1 - eps])
    guess_one_hot = np.zeros(env.get_action_space())
    guess_one_hot[guess] = 1

    # Make a decision
    choices = [
        np.random.choice([np.random.randint(env.get_action_space()), agents[0]((obs, guess_one_hot))],
                         1, p=[eps, 1 - eps]),
        guess,
        np.random.choice([np.random.randint(env.get_action_space()), agents[2](obs)],
                         1, p=[eps, 1 - eps]),
    ]

    # Take a reward
    obs, rewards = env.action(choices)
    for id, agent in enumerate(agents):
        agent.give_reward(rewards[id])

    if r < eps_rnd:
        eps -= eps_con
    else:
        eps = eps_low

    for agent in agents:
        agent.train()

for agent in agents:
    agent.save_agent_state(path)
