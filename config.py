from enum import Enum


class ModelType(Enum):
    baseline = 0,
    attention = 1,
    rnn = 2


# mp
cores = 8
epochs = 20

n_agents = 1  # number of agents who won't negotiate
train_episodes = 2000
test_episodes = 50
shuffle = True
lr = 0.00075
hidden_size = 32
gamma = 1.
entropy_coef = 0.1
epsilon_upper = 0.5
epsilon_lower = 0.001
epsilon_step = (epsilon_upper - epsilon_lower) / train_episodes
metric_directory = None  # 'test'

negot_n_agents = 2  # number of agents who will negotiate
negot_steps = 2
is_channel_open = False

players = n_agents + negot_n_agents
