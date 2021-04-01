from enum import Enum


class ModelType(Enum):
    baseline_mlp = 0
    baseline_rnn = 1
    attention = 2
    siam_mlp = 3
    siam_rnn = 4


class LogType(Enum):
    no = 0
    local = 1
    mlflow = 2
    show = 3


# mp
cores = 7
epochs = 1000
pickle_file = 'mp_siam_statistic.pickle'
mp_models = [ModelType.siam_mlp]

# episodes
train_episodes = 2000  # 3000
test_episodes = 100
round_episodes = 20

# learning
Train = True  # use False only for Random (non-trainable) Agents!
lr = 0.002  # 0.001
hidden_size = 32
gamma = 1.
value_loss_penalize = 0.5

# exploration
entropy_coef = 0.05
epsilon_upper = 0.5
epsilon_lower = 0.001
epsilon_step = (epsilon_upper - epsilon_lower) / train_episodes

# negotiation
use_negotiation = True
message_space = 3
use_embeddings = False
embedding_space = 32
is_channel_open = True
negotiable_agents = 2
negotiation_steps = 2

# common
agents = 1  # number of agens who won't negotiate
players = agents + negotiable_agents
shuffle = False
logging = LogType.local
metric_directory = 'metrics'
experiment_name = 'test'


def is_require_reset(model_type: ModelType):
    require = False
    if model_type == ModelType.baseline_rnn:
        require = True
    return require


def print_config():
    from sty import fg, ef, Style, RgbFg
    fg.orange = Style(RgbFg(255, 165, 0))
    fg.orange_dark = Style(RgbFg(229, 83, 0))
    fg.red_war = Style(RgbFg(255, 10, 10))

    print('\n' + fg.orange_dark + 'Multiprocessing:' + fg.rs)
    print(fg.orange + 'Cores: ' + fg.rs + f'{cores}')
    print(fg.orange + 'Epochs: ' + fg.rs + f'{epochs}')
    print(fg.orange + 'Pickle file: ' + fg.rs + f'{pickle_file}')
    print(fg.orange + 'Mp models: ' + fg.rs + f'{mp_models}')

    print('\n' + fg.orange_dark + 'Episodes:' + fg.rs)
    print(fg.orange + 'Train episodes: ' + fg.rs + f'{train_episodes}')
    print(fg.orange + 'Test episodes: ' + fg.rs + f'{test_episodes}')
    print(fg.orange + 'Round episodes: ' + fg.rs + f'{round_episodes}')

    print('\n' + fg.orange_dark + 'Learning:' + fg.rs)
    print(fg.orange + 'Train: ' + fg.rs + f'{Train} ' + ('' if Train else (fg.red_war + 'Is this expected?!' + fg.rs)))
    print(fg.orange + 'Learning rate: ' + fg.rs + f'{lr}')
    print(fg.orange + 'Hidden size: ' + fg.rs + f'{hidden_size}')
    print(fg.orange + 'Gamma: ' + fg.rs + f'{gamma}')
    print(fg.orange + 'Value loss penalize: ' + fg.rs + f'{value_loss_penalize}')

    print('\n' + fg.orange_dark + 'Exploration:' + fg.rs)
    print(fg.orange + 'Entropy coef: ' + fg.rs + f'{entropy_coef}')
    print(fg.orange + 'Epsilon upper: ' + fg.rs + f'{epsilon_upper}')
    print(fg.orange + 'Epsilon lower: ' + fg.rs + f'{epsilon_lower}')
    print(fg.orange + 'Epsilon step: ' + fg.rs + f'{round(epsilon_step, 7)}')

    print('\n' + fg.orange_dark + 'Negotiation:' + fg.rs)
    print(fg.orange + 'Use negotiation: ' + fg.rs + f'{use_negotiation}')
    print(fg.orange + 'Message space: ' + fg.rs + f'{message_space}')
    print(fg.orange + 'Use embeddings: ' + fg.rs + f'{use_embeddings}')
    print(fg.orange + 'Embedding space: ' + fg.rs + f'{embedding_space}')
    print(fg.orange + 'Is channel open: ' + fg.rs + f'{is_channel_open}')
    print(fg.orange + 'Negotiable agents: ' + fg.rs + f'{negotiable_agents}')
    print(fg.orange + 'Negotiation steps: ' + fg.rs + f'{negotiation_steps}')

    print('\n' + fg.orange_dark + 'Common:' + fg.rs)
    print(fg.orange + 'Agents: ' + fg.rs + f'{agents}')
    print(fg.orange + 'Players: ' + fg.rs + f'{players}')
    print(fg.orange + 'Shuffle: ' + fg.rs + f'{shuffle}')
    print(fg.orange + 'Logging: ' + fg.rs + f'{logging.name}')
    print(fg.orange + 'Metric directory: ' + fg.rs + f'{metric_directory}')
    print(fg.orange + 'Experiment name: ' + fg.rs + f'{experiment_name}')


if __name__ == '__main__':
    print_config()
