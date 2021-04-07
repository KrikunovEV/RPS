import pickle
import os
import matplotlib.pyplot as plt
import yaml
from easydict import EasyDict
from enum import Enum
from sty import fg, ef, Style, RgbFg


# CONSOLE OUTPUT COLORS
fg.category = Style(RgbFg(229, 83, 0))
fg.parameter = Style(RgbFg(255, 165, 0))
fg.warning = Style(RgbFg(255, 204, 0))
fg.error = Style(RgbFg(255, 51, 0))


# ENUM OBJECTS
class ModelType(Enum):
    baseline_mlp = 0
    baseline_rnn = 1
    attention = 2
    siam_mlp = 'siam_mlp'
    siam_rnn = 4


class LogType(Enum):
    no = 'no'
    local = 'local'
    local_randomly = 'local_randomly'
    mlflow = 'mlflow'
    show = 'show'


def load_config(path: str):
    with open(path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exception:
            raise Exception(ef.bold + path + ef.rs + ': ' + str(exception))

    config = EasyDict(config)

    # convert model names to objects
    if not isinstance(config.mp.model_list, list):
        raise Exception(ef.bold + 'config.mp.model_list' + ef.rs + f' must be a list of model names but got '
                                                                   f'{type(config.mp.model_list)}')
    model_list = config.mp.model_list.copy()
    config.mp.model_list = []
    for model_name in model_list:
        config.mp.model_list.append(ModelType(model_name))

    # convert log name to object
    if not isinstance(config.common.logging, str):
        raise Exception(ef.bold + 'config.common.logging' + ef.rs + f' must be a str but got '
                                                                    f'{type(config.common.logging)}')
    config.common.logging = LogType(config.common.logging)

    # determine epsilon step
    config.train.epsilon_step = (config.train.epsilon_upper - config.train.epsilon_lower) / config.train.episodes

    # check the number of negotiation agents
    if config.negotiation.players > config.common.players:
        raise Exception('The number of ' + ef.bold + f'config.negotiation.players = {config.negotiation.players}' +
                        ef.rs + ' must be <= than ' + ef.bold + f'config.common.players = {config.common.players}' +
                        ef.rs)

    # correct negotiation steps
    if not isinstance(config.negotiation.steps, list):
        raise Exception(ef.bold + 'config.negotiation.steps' + ef.rs + f' must be a list of ints but got '
                                                                       f'{type(config.negotiation.steps)}')
    if len(config.negotiation.steps) != 1 and len(config.negotiation.steps) != config.common.players:
        raise Exception(ef.bold + f'len(config.negotiation.steps) = {len(config.negotiation.steps)}' + ef.rs +
                        f' must be equal to 1 or the number of players ({config.common.players})')
    if len(config.negotiation.steps) == 1:
        steps = config.negotiation.steps[0]
        config.negotiation.steps = [steps for _ in range(config.common.players)]

    # check that there is any input state
    if not config.common.use_obs and not config.negotiation.use and not config.embeddings.use:
        raise Exception('There is no any input state: ' + ef.bold + f'config.common.use_obs = False' + ef.rs + ', ' +
                        ef.bold + f'config.negotiation.use = False' + ef.rs + ', ' +
                        ef.bold + f'config.embeddings.use = False' + ef.rs)

    # create experiment dir
    if not isinstance(config.common.experiment_dir, str):
        raise Exception(ef.bold + 'config.common.experiment_dir' + ef.rs + f' must be a str but got '
                                                                           f'{type(config.common.experiment_dir)}')
    if not isinstance(config.common.experiment_name, str):
        raise Exception(ef.bold + 'config.common.experiment_name' + ef.rs + f' must be a str but got '
                                                                            f'{type(config.common.experiment_name)}')
    os.makedirs(os.path.join(config.common.experiment_dir, config.common.experiment_name), exist_ok=True)

    # print config
    print()
    for (category_name, category_dict) in config.items():
        print(fg.category + category_name + fg.rs + ':')
        for (parameter_name, parameter_value) in category_dict.items():
            print('\t' + fg.parameter + parameter_name + fg.rs + f': {parameter_value}')

    # print warning when do_backward is false
    if not config.train.do_backward:
        print('\n' + fg.warning + f'config.train.do_backward = {config.train.do_backward}' + fg.rs + ' Is it expected?')

    return config


def is_require_reset(model_type: ModelType):
    require = False
    if model_type == ModelType.baseline_rnn:
        require = True
    return require


def stat_plot(cfg: EasyDict):
    file = os.path.join(cfg.common.experiment_dir, 'siam_mlp', 'embeddings_ones', 'epoch', cfg.mp.stat_file)
    with open(file, 'rb') as f:
        model_coops = pickle.load(f)['coops_dict']['siam_mlp']

    fig, ax = plt.subplots(1, 3)
    plt.suptitle(r'embeddings ~{1}$^{64}$')
    for i, (key, value) in enumerate(model_coops.items()):
        ax[i].set_title(key)
        ax[i].hist(value, bins=5)
        ax[i].set_ylim(0., 800.)

    plt.show()


if __name__ == '__main__':
    load_config('config/default.yaml')
