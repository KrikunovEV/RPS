from enum import Enum


class ModelType(Enum):
    baseline_mlp = 0
    baseline_rnn = 1
    attention = 2



class LogType(Enum):
    no = 0
    local = 1
    mlflow = 2
    show = 3


# mp
cores = 8
epochs = 1000

# common
n_agents = 1  # number of agents who won't negotiate
train_episodes = 3000
test_episodes = 50
shuffle = True
Train = False  # always True. False only for Random (non-trainable) Agent!
lr = 0.00075
hidden_size = 32
gamma = 1.
entropy_coef = 0.1
epsilon_upper = 0.5
epsilon_lower = 0.001
epsilon_step = (epsilon_upper - epsilon_lower) / train_episodes
metric_directory = 'metrics'

# negotiation
negot_n_agents = 2  # number of agents who will negotiate
negot_steps = 2
is_channel_open = False

players = n_agents + negot_n_agents
