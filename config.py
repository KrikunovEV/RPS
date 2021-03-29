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


def print_config():
    from sty import fg, ef, Style, RgbFg
    fg.orange = Style(RgbFg(255, 165, 0))
    fg.orange_dark = Style(RgbFg(229, 83, 0))
    fg.red_war = Style(RgbFg(255, 10, 10))

    print('\n' + fg.orange_dark + 'Multiprocessing:' + fg.rs)
    print(fg.orange + 'Cores: ' + fg.rs + f'{cores}')
    print(fg.orange + 'Epochs: ' + fg.rs + f'{epochs}')

    print('\n' + fg.orange_dark + 'Episodes:' + fg.rs)
    print(fg.orange + 'Train episodes: ' + fg.rs + f'{train_episodes}')
    print(fg.orange + 'Test episodes: ' + fg.rs + f'{test_episodes}')
    print(fg.orange + 'Rnn episodes: ' + fg.rs + f'{rnn_episodes}')

    print('\n' + fg.orange_dark + 'Learning:' + fg.rs)
    print(fg.orange + 'Train: ' + fg.rs + f'{Train} ' + ('' if Train else (fg.red_war + 'Is this expected?!' + fg.rs)))
    print(fg.orange + 'Learning rate: ' + fg.rs + f'{lr}')
    print(fg.orange + 'Hidden size: ' + fg.rs + f'{hidden_size}')
    print(fg.orange + 'Gamma: ' + fg.rs + f'{gamma}')

    print('\n' + fg.orange_dark + 'Exploration:' + fg.rs)
    print(fg.orange + 'Entropy coef: ' + fg.rs + f'{entropy_coef}')
    print(fg.orange + 'Epsilon upper: ' + fg.rs + f'{epsilon_upper}')
    print(fg.orange + 'Epsilon lower: ' + fg.rs + f'{epsilon_lower}')
    print(fg.orange + 'Epsilon step: ' + fg.rs + f'{round(epsilon_step, 7)}')

    print('\n' + fg.orange_dark + 'Negotiation:' + fg.rs)
    print(fg.orange + 'Use negotiation: ' + fg.rs + f'{use_negotiation}')
    print(fg.orange + 'Use embeddings: ' + fg.rs + f'{use_embeddings}')
    print(fg.orange + 'Is channel open: ' + fg.rs + f'{is_channel_open}')
    print(fg.orange + 'Negotiable agents: ' + fg.rs + f'{negotiable_agents}')
    print(fg.orange + 'Negotiation steps: ' + fg.rs + f'{negotiation_steps}')

    print('\n' + fg.orange_dark + 'Common:' + fg.rs)
    print(fg.orange + 'Agents: ' + fg.rs + f'{agents}')
    print(fg.orange + 'Players: ' + fg.rs + f'{players}')
    print(fg.orange + 'Shuffle: ' + fg.rs + f'{shuffle}')
    print(fg.orange + 'Metric directory: ' + fg.rs + f'{metric_directory}')


if __name__ == '__main__':
    print_config()
