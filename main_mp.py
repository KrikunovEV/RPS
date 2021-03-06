import utility as util
import multiprocessing as mp
import os
import numpy as np
import torch
from main import run_game


class Task:
    def __init__(self, game_name: str, done: bool):
        self.game_name = game_name
        self.done = done


class Result:
    def __init__(self, metrics_dict: dict, done: bool, task: Task):
        self.metrics_dict = metrics_dict
        self.done = done
        self.task = task


def process_work(p_name: str, task_q: mp.Queue, result_q: mp.Queue, cfg):
    seed = np.abs(p_name.__hash__()) % 4294967296  # 2**32
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print('\n' + util.fg.category + f'{p_name}' + util.fg.rs + ' seed: ' + util.fg.parameter + f'{seed}' + util.fg.rs +
          ', np initial state: ' + util.fg.warning + f'{np.random.get_state()[1][0]}' + util.fg.rs +
          ', torch initial state: ' + util.fg.warning + f'{torch.get_rng_state()[0]}' + util.fg.rs)

    tasks_done = 0
    while True:
        task = task_q.get()

        if task.done:
            print('\n' + util.fg.category + f'{p_name}' + util.fg.rs + ' done ' +
                  util.fg.parameter + f'{tasks_done}' + util.fg.rs + ' tasks.')
            result_q.put(Result(metrics_dict=dict(), done=True, task=task))
            break

        metrics_dict = run_game(cfg, task.game_name)
        result_q.put(Result(metrics_dict=metrics_dict, done=False, task=task))
        tasks_done += 1
        util.print_game_stats(metrics_dict, cfg, p_name)


if __name__ == '__main__':
    cfg = util.load_config('config/default.yaml')

    manager = mp.Manager()
    task_q = manager.Queue()
    result_q = manager.Queue()

    # create processes
    processes = []
    for process in range(cfg.mp.cores):
        processes.append(mp.Process(target=process_work, args=(f'[C{process + 1}]', task_q, result_q, cfg)))
        processes[-1].start()

    # fill queue by tasks
    for game in range(cfg.mp.games):
        task_q.put(Task(game_name=str(game), done=False))
    for _ in range(len(processes)):
        task_q.put(Task(game_name='', done=True))

    # create mp directory and stats
    stat_dir = os.path.join(cfg.common.experiment_dir, cfg.common.experiment_name, 'mp_stat')
    os.makedirs(stat_dir, exist_ok=True)

    spent_time = 0
    pair_coops = dict()
    total_games = 0
    processes_done = 0
    while processes_done != cfg.mp.cores:

        result = result_q.get()
        if result.done:
            processes_done += 1
            continue

        # log metrics and stat data
        util.log_metrics(result.metrics_dict, cfg)
        spent_time += result.metrics_dict['time']

        for (pair, coops) in result.metrics_dict['pair_coops'].items():
            if pair not in pair_coops:
                pair_coops[pair] = []
            pair_coops[pair].append(coops)

        total_games += 1
        if total_games % 10 == 0:
            util.log_stats(pair_coops, total_games, stat_dir, cfg)

    util.log_stats(pair_coops, total_games, stat_dir, cfg)
    spent_time /= cfg.mp.cores
    print('Spent time in mean for 1 core ' +
          util.fg.warning + f'{int(spent_time // 60)}m {round(spent_time / 60)}s' + util.fg.rs)
