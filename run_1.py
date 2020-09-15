from RockPaperScissors import Environment3p
from Agent import Agent
import numpy as np


rounds = 1000
players = 3
path = 'test2/'
eps = 0.9
eps_low = 0.01
eps_rnd = 500
eps_con = (eps - eps_low) / eps_rnd

env = Environment3p()
agent = Agent(0, 3, env.get_action_space(), negotiate=True)

obs = env.obs
for r in range(rounds):
    print(f'Round {r}, eps={eps}')

    # Make a guess
    guess = np.random.randint(env.get_action_space())
    guess_one_hot = np.zeros(env.get_action_space())
    guess_one_hot[guess] = 1

    # Make a decision
    choices = [
        agent(guess_one_hot),
        guess,
        np.random.randint(env.get_action_space())
    ]

    # Take a reward
    obs, rewards = env.action(choices)
    agent.give_reward(rewards[0])

    if r < eps_rnd:
        eps -= eps_con
    else:
        eps = eps_low

    agent.train()

agent.save_agent_state(path)
