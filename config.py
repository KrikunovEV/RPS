

n_agents = 1  # number of agents who won't negotiate
train_episodes = 1000
test_episodes = 200
lr = 0.005
gamma = 0.9
entropy_coef = 0.001
metric_directory = None  # 'test'

negot_n_agents = 2  # number of agents who will negotiate
negot_steps = 2
is_channel_open = False

players = n_agents + negot_n_agents
