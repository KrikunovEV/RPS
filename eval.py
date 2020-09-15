from RockPaperScissors import Environment3p
from Agent import Agent
import numpy as np


rounds = 10000
players = 3
path = 'test/'
eval = True

env = Environment3p(debug=True)
agent = Agent(0, 3, env.get_action_space(), negotiate=True, eval=eval)
agent.load_agent_state(f'{path}0.pt')
parties_won = [0, 0, 0]
reward_cum = [[0], [0], [0]]

cooperate, greedy = 0, 0

obs = env.obs
for r in range(rounds):
    print(f'Round {r}')

    # Make a guess
    guess = np.random.randint(env.get_action_space())
    guess_one_hot = np.zeros(env.get_action_space())
    guess_one_hot[guess] = 1

    # Make a decision
    choices = [
        agent.make_guess(guess_one_hot),
        guess,
        np.random.randint(env.get_action_space())
    ]

    # Take a reward
    obs, rewards = env.action(choices)
    for i, r in enumerate(rewards):
        if r > 0:
            parties_won[i] += 1
            reward_cum[i].append(reward_cum[i][-1] + r)
    if rewards[0] != 0 and rewards[1] != 0:
        cooperate += 1
    elif rewards[0] > 0 and rewards[1] == 0:
        greedy += 1

print('\nAgents\' scores')
print(parties_won)
print(cooperate, greedy)