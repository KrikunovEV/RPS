from RockPaperScissors import Environment3p
from Agent import Agent


rounds = 1000
players = 3
steps = 2
path = 'test/'

env = Environment3p(steps=steps)
agents = [
    Agent(0, env.get_obs_space(), env.get_action_space(), negotiate=True),
    Agent(1, env.get_obs_space(), env.get_action_space(), negotiate=False),
    Agent(2, env.get_obs_space(), env.get_action_space(), negotiate=False),
]

for r in range(rounds):
    print(f'Round {r}')
    obs = env.reset()

    for s in range(steps):

        # Make a guess

        guess = agents[1].make_guess(obs)

        # Make a decision

        choices = [
            agents[0]((obs, guess)),
            agents[1](obs),
            agents[2](obs),
        ]

        # Take a reward

        obs, rewards = env.action(choices)
        for id, agent in enumerate(agents):
            agent.give_reward(rewards[id])

    for agent in agents:
        agent.train()

for agent in agents:
    agent.save_agent_state(path)
