# mp
cores = 8
epochs = 50

n_agents = 1  # number of agents who won't negotiate
train_episodes = 2000
test_episodes = 50
lr = 0.00075
gamma = 0.9
entropy_coef = 0.1
epsilon = 0.25
epsilon_upper = 0.25
epsilon_lower = 0.01
epsilon_step = (epsilon - epsilon_lower) / train_episodes
metric_directory = None  # 'test'

negot_n_agents = 2  # number of agents who will negotiate
negot_steps = 2
is_channel_open = False

players = n_agents + negot_n_agents
