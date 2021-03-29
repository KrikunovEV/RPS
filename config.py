from enum import Enum


class ModelType(Enum):
    baseline_mlp = 0
    baseline_rnn = 1
    attention = 2
    per_agent_mlp = 3
    per_agent_rnn = 4


class LogType(Enum):
    no = 0
    local = 1
    mlflow = 2
    show = 3


# mp
cores = 8
epochs = 1000

# episodes
train_episodes = 3000
test_episodes = 100
rnn_episodes = 25

# learning
Train = True  # use False only for Random (non-trainable) Agents!
lr = 0.00075
hidden_size = 32
gamma = 1.

# exploration
entropy_coef = 0.1
epsilon_upper = 0.5
epsilon_lower = 0.001
epsilon_step = (epsilon_upper - epsilon_lower) / train_episodes

# negotiation
use_negotiation = False
use_embeddings = True
is_channel_open = False
negotiable_agents = 2
negotiation_steps = 2

# common
agents = 1  # number of agens who won't negotiate
players = agents + negotiable_agents
shuffle = True
metric_directory = 'metrics'
