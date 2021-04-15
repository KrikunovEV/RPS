import pickle
import os
import matplotlib.pyplot as plt
import yaml
import numpy as np
import seaborn as sn
from sklearn.manifold import TSNE
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
    file = os.path.join(cfg.common.experiment_dir, cfg.common.experiment_name, 'mp_stat', cfg.mp.stat_file)
    with open(file, 'rb') as f:
        pair_coops = pickle.load(f)['model_pair_coops']['siam_mlp']

    fig, ax = plt.subplots(1, len(pair_coops))
    plt.suptitle(r'embeddings ~{1}$^{64}$')
    for i, (pair, coops) in enumerate(pair_coops.items()):
        ax[i].set_title(pair)
        ax[i].hist(coops, bins=10)
        ax[i].set_ylim(0., 300.)

    plt.show()


def print_game_stats(model_type: ModelType, metrics_dict: dict, cfg: EasyDict, pname: str = '[C1]'):
    text = '\n' + fg.category + f'{pname}' + fg.rs
    text += ': the game '
    text += fg.category + metrics_dict['game'] + fg.rs
    text += ' lasts '
    text += fg.category + f'{int(metrics_dict["time"] // 60)}m {round(metrics_dict["time"] % 60)}s' + fg.rs
    text += ' using '
    text += fg.category + model_type.name + fg.rs
    text += ' model with pair coops:'
    for (pair, coops) in metrics_dict['pair_coops'].items():
        text += '\n' + fg.parameter + f'{pair}' + fg.rs
        text += f': {coops}/{cfg.test.episodes} ' + ef.bold + f'({coops / cfg.test.episodes})' + ef.rs
    print(text)


def log_stats(model_pair_coops: dict, model_game_counter: dict, stat_directory, cfg: EasyDict):
    text = '\nLog stats'
    for model_type in cfg.mp.model_list:
        name = model_type.name
        text += '\n' + fg.category + f'{name}' + fg.rs + ':'

        for (pair, coops) in model_pair_coops[name].items():
            coops = np.sum(coops)
            total_coops = model_game_counter[name] * cfg.test.episodes
            text += '\n' + fg.parameter + f'{pair}' + fg.rs
            text += f': {coops}/{total_coops} ' + ef.bold + f'({coops / total_coops})' + ef.rs

    print(text)
    with open(os.path.join(stat_directory, cfg.mp.stat_file), 'wb') as f:
        pickle.dump({'model_pair_coops': model_pair_coops, 'model_game_counter': model_game_counter}, f)


def log_metrics(metrics_dict: dict, cfg: EasyDict):
    if cfg.common.logging == LogType.show:
        visualize_metrics(metrics_dict, cfg, directory=None)

    elif cfg.common.logging == LogType.local or cfg.common.logging == LogType.local_randomly:
        if cfg.common.logging == LogType.local_randomly and np.random.randint(3) != 0:
            return

        directory = os.path.join(cfg.common.experiment_dir, cfg.common.experiment_name, f'{metrics_dict["game"]}')
        os.makedirs(directory, exist_ok=True)
        visualize_metrics(metrics_dict, cfg, directory=directory)

    elif cfg.common.logging == LogType.mlflow:
        raise Exception(ef.bold + 'LogType.mlflow' + ef.rs + ' not implemented yet')


def visualize_metrics(metrics_dict: dict, cfg: EasyDict, directory: str = None):
    plt.close('all')

    agent_labels = metrics_dict['agent_labels_list']
    agent_id_labels = [int(label[:2]) for label in agent_labels]  # get only id, consider 0 <= id <= 99

    # LOSS
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_title('A2C loss', fontsize=18)
    ax.set_xlabel('# of episode', fontsize=16)
    ax.set_ylabel('loss value', fontsize=16)
    ax.tick_params(labelsize=13)
    for loss, label in zip(metrics_dict['loss_list'], agent_labels):
        plt.plot(loss, label=label)
    ax.legend()
    fig.tight_layout()
    if directory is not None:
        plt.savefig(f'{directory}/a2c_loss.png')

    # CUMULATIVE REWARD
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    ax[0].set_title(f'Cumulative reward on train', fontsize=18)
    ax[0].set_xlabel('# of episode', fontsize=16)
    ax[0].set_ylabel('reward value', fontsize=16)
    ax[0].tick_params(labelsize=13)
    ax[1].set_title(f'Cumulative reward on test', fontsize=18)
    ax[1].set_xlabel('# of episode', fontsize=16)
    ax[1].set_ylabel('reward value', fontsize=16)
    ax[1].tick_params(labelsize=13)
    for reward, reward_eval, label in zip(metrics_dict['reward_list'], metrics_dict['reward_eval_list'], agent_labels):
        ax[0].plot(np.cumsum(reward), label=label)
        ax[1].plot(np.cumsum(reward_eval), label=label)
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    if directory is not None:
        plt.savefig(f'{directory}/rewarding.png')

    # ELO
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    ax[0].set_title(f'ELO on train', fontsize=18)
    ax[0].set_xlabel('# of episode', fontsize=16)
    ax[0].set_ylabel('elo value', fontsize=16)
    ax[0].tick_params(labelsize=13)
    ax[1].set_title(f'ELO on test', fontsize=18)
    ax[1].set_xlabel('# of episode', fontsize=16)
    ax[1].set_ylabel('elo value', fontsize=16)
    ax[1].tick_params(labelsize=13)
    elo_train = np.array(metrics_dict['elo_train']).transpose()
    elo_test = np.array(metrics_dict['elo_test']).transpose()
    for elo0, elo1, label in zip(elo_train, elo_test, agent_labels):
        ax[0].plot(elo0, label=label)
        ax[1].plot(elo1, label=label)
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    if directory is not None:
        plt.savefig(f'{directory}/elo.png')

    # ACTIONS MATRIX
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    ax[0].set_title('Offends', fontsize=18)
    ax[0].tick_params(labelsize=13)
    ax[1].set_title('Defends', fontsize=18)
    ax[1].tick_params(labelsize=13)
    AM = np.stack(metrics_dict['attacks_list'])
    DM = np.stack(metrics_dict['defends_list'])
    sn.heatmap(AM, annot=True, cmap='Reds', xticklabels=agent_id_labels, yticklabels=agent_labels, ax=ax[0],
               square=True, cbar=False, fmt='g', annot_kws={"size": 15})
    sn.heatmap(DM, annot=True, cmap='Blues', xticklabels=agent_id_labels, yticklabels=agent_labels, ax=ax[1],
               square=True, cbar=False, fmt='g', annot_kws={"size": 15})
    for (a, d) in zip(ax[0].yaxis.get_ticklabels(), ax[1].yaxis.get_ticklabels()):
        a.set_verticalalignment('center')
        a.set_rotation('horizontal')
        d.set_verticalalignment('center')
        d.set_rotation('horizontal')
    fig.tight_layout()
    if directory is not None:
        plt.savefig(f'{directory}/action_matrix.png')

    if cfg.negotiation.use:
        steps = np.max(cfg.negotiation.steps)
        categories = cfg.negotiation.space + 1
        fig, ax = plt.subplots(1, steps, figsize=(16, 9))
        xlabels = np.arange(1, categories).tolist() + ['empty']
        locations = np.arange(categories)
        player_loc = np.linspace(-0.4, 0.4, cfg.common.players)
        width = 0.8 / cfg.common.players
        messages = metrics_dict['messages_list']
        for step in range(steps):
            for i in range(cfg.common.players):
                if step < cfg.negotiation.steps[i]:
                    ax[step].bar(locations + player_loc[i], messages[i][step], width, label=agent_labels[i])
            ax[step].set_ylabel('message count', fontsize=16)
            ax[step].set_xlabel('message category', fontsize=16)
            ax[step].set_title(f'Negotiation step {step + 1}', fontsize=18)
            ax[step].set_xticks(locations)
            ax[step].set_xticklabels(xlabels)
            ax[step].tick_params(labelsize=13)
            ax[step].legend()
        fig.tight_layout()
        if directory is not None:
            plt.savefig(f'{directory}/messages.png')

    if cfg.embeddings.use:
        embeddings = metrics_dict['embeddings_list']
        for p in range(cfg.common.players):
            fig, ax = plt.subplots(1, 1, figsize=(16, 9))
            ax.set_title(f'Player {p + 1}: embeddings ({cfg.embeddings.space}) PCA', fontsize=18)
            p_embeddings = np.stack(embeddings[p].data.detach().numpy())
            pca = TSNE()
            p_embeddings = pca.fit_transform(p_embeddings)
            for i, (emb, label, ann_label) in enumerate(zip(p_embeddings, agent_labels, agent_id_labels)):
                ax.scatter(emb[0], emb[1], label=label, s=150)
                ax.annotate(ann_label, emb + np.array([0, 0.1]), fontsize=14, ha='center')
            ax.legend()
            fig.tight_layout()
            if directory is not None:
                plt.savefig(f'{directory}/p{p + 1}_embeddings.png')

    if directory is None:
        plt.show()


if __name__ == '__main__':
    cfg = load_config('config/default.yaml')
    stat_plot(cfg)
