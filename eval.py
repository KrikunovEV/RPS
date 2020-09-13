from RockPaperScissors import Environment3p, Choice
from Agent import Agent


rounds = 1
players = 3
steps = 2
path = 'test/'

env = Environment3p(steps=steps, debug=True)
agents = [
    Agent(0, env.get_obs_space(), env.get_action_space(), negotiate=True),
    Agent(1, env.get_obs_space(), env.get_action_space(), negotiate=False),
    Agent(2, env.get_obs_space(), env.get_action_space(), negotiate=False),
]
for id, agent in enumerate(agents):
    agent.load_agent_state(f'{path}{id}.pt')
    agent.parties_won = 0
    agent.reward_cum = [0]

for r in range(rounds):
    print(f'Round {r}')
    obs = env.reset()

    for s in range(steps):

        # Make a guess

        guess = agents[1].make_guess(obs)
        #print(f'Guess')

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
