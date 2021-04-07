from OffendAndDefend import OADEnvironment
from AgentOrchestrator import Orchestrator
import utility as util
import time
import os


def run(cfg, game, model_type: util.ModelType, debug: bool = False):

    env = OADEnvironment(players=cfg.common.players, debug=debug)
    orchestrator = Orchestrator(obs_space=env.get_obs_space(),
                                action_space=env.get_action_space(),
                                model_type=model_type, cfg=cfg)

    orchestrator.set_eval(eval=False)
    epsilon = cfg.train.epsilon_upper
    obs = None
    for episode in range(cfg.train.episodes):
        if debug:
            print('Game: ' + util.fg.warning + f'{game}' + util.fg.rs + ', train episode: ' +
                  util.fg.warning + f'{episode + 1}/{cfg.train.episodes}' + util.fg.rs)

        if episode % cfg.common.round_episodes == 0:
            obs = env.reset()
            if util.is_require_reset(model_type):
                orchestrator.reset_memory()

        if cfg.common.shuffle:
            obs = orchestrator.shuffle(obs)

        if cfg.negotiation.use:
            orchestrator.negotiation(obs)

        choices = orchestrator.decisions(obs, epsilon)
        obs, rewards = env.play(choices)
        orchestrator.rewarding(rewards)
        orchestrator.train()
        epsilon -= cfg.train.epsilon_step

    orchestrator.set_eval(eval=True)
    obs = None
    result = {'1 & 2 vs 3': 0, '2 & 3 vs 1': 0, '1 & 3 vs 2': 0}
    for episode in range(cfg.test.episodes):
        if debug:
            print('Game: ' + util.fg.warning + f'{game}' + util.fg.rs + ', set episode: ' +
                  util.fg.warning + f'{episode + 1}/{cfg.train.episodes}' + util.fg.rs)

        if episode % cfg.common.round_episodes == 0:
            obs = env.reset()
            if util.is_require_reset(model_type):
                orchestrator.reset_memory()

        if cfg.common.shuffle:
            obs = orchestrator.shuffle(obs)

        if cfg.negotiation.use:
            orchestrator.negotiation(obs)

        choices = orchestrator.decisions(obs, epsilon)
        obs, rewards = env.play(choices)
        orchestrator.rewarding(rewards)

        if choices[2][0] == 0 and choices[2][1] == 0 and choices[1][0] == 0 and choices[1][1] == 0:
            result['2 & 3 vs 1'] += 1
        elif choices[2][0] == 1 and choices[2][1] == 1 and choices[0][0] == 1 and choices[0][1] == 1:
            result['1 & 3 vs 2'] += 1
        elif choices[0][0] == 2 and choices[0][1] == 2 and choices[1][0] == 2 and choices[1][1] == 2:
            result['1 & 2 vs 3'] += 1

    if cfg.common.logging == util.LogType.show:
        orchestrator.plot_metrics(directory=None)
    elif cfg.common.logging == util.LogType.local or cfg.common.logging == util.LogType.local_randomly:
        if cfg.common.logging == util.LogType.local_randomly:
            raise Exception(util.ef.bold + 'LogType.local_randomly' + util.ef.rs + ' not implemented yet')
        directory = os.path.join(cfg.common.experiment_dir, cfg.common.experiment_name, f'{game}')
        os.makedirs(directory, exist_ok=True)
        orchestrator.plot_metrics(directory=directory)
    elif cfg.common.logging == util.LogType.mlflow:
        raise Exception(util.ef.bold + 'LogType.mlflow' + util.ef.rs + ' not implemented yet')

    return result


if __name__ == '__main__':
    cfg = util.load_config('config/default.yaml')

    start_time = time.time()
    coops = run(cfg, 'test', util.ModelType.siam_mlp, debug=True)
    lasted_time = time.time() - start_time
    print('The experiment lasts ' + util.fg.warning + f'{lasted_time // 60}m {lasted_time % 60}s' + util.fg.rs)

    print('Coops:')
    for (key, value) in coops.items():
        print(f'{key}: {value}/{cfg.test.episodes} ({value / cfg.test.episodes})')
