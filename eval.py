from RockPaperScissors import Environment3p
from Agent import Agent
import numpy as np


rounds = 10000
players = 3
path = 'test/'
eval = False

env = Environment3p(debug=True)
agents = [
    Agent(0, env.get_obs_space(), env.get_action_space(), eval=eval, negotiate=True),
    Agent(1, env.get_obs_space(), env.get_action_space(), eval=eval),
    Agent(2, env.get_obs_space(), env.get_action_space(), eval=eval),
]

for id, agent in enumerate(agents):
    agent.load_agent_state(f'{path}{id}.pt')
    agent.parties_won = 0
    agent.reward_cum = [0]

obs = env.obs
for r in range(rounds):
    print(f'Round {r}')

    # Make a guess
    guess = agents[1](obs)
    guess_one_hot = np.zeros(env.get_action_space())
    guess_one_hot[guess] = 1
    #print(f'Guess {guess}')

    # Make a decision
    choices = [
        agents[0]((obs, guess_one_hot)),
        guess,
        agents[2](obs)
    ]

    # Take a reward
    obs, rewards = env.action(choices)
    for id, agent in enumerate(agents):
        agent.give_reward(rewards[id])

print('\nAgents\' scores')
for agent in agents:
    print(agent.reward_cum[-1])