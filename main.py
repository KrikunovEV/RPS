from OffendAndDefend import OADEnvironment
from Orchestrator import Orchestrator
import utility as util
import time


def run_game(cfg, game: str, debug: bool = False):
    start_time = time.time()

    env = OADEnvironment(players=cfg.common.players, debug=debug)
    orchestrator = Orchestrator(obs_space=env.get_obs_space(), action_space=env.get_action_space(), cfg=cfg)

    orchestrator.set_eval(eval=False)
    epsilon = cfg.train.epsilon_upper
    obs = None
    elo_train = []
    for episode in range(cfg.train.episodes):
        if debug:
            print('Game: ' + util.fg.warning + f'{game}' + util.fg.rs + ', train episode: ' +
                  util.fg.warning + f'{episode + 1}/{cfg.train.episodes}' + util.fg.rs)

        if episode % cfg.common.round_episodes == 0:
            obs = env.reset()
            orchestrator.reset_memory()

        if cfg.common.shuffle:
            orchestrator.shuffle()

        if cfg.negotiation.use:
            orchestrator.negotiation()

        choices = orchestrator.decisions(obs, epsilon)
        obs, rewards, elo = env.play(choices)
        elo_train.append(elo)
        orchestrator.rewarding(rewards)
        orchestrator.train()
        epsilon -= cfg.train.epsilon_step

    orchestrator.set_eval(eval=True)
    env.reset_elo()
    obs = None
    elo_test = []
    for episode in range(cfg.test.episodes):
        if debug:
            print('Game: ' + util.fg.warning + f'{game}' + util.fg.rs + ', test episode: ' +
                  util.fg.warning + f'{episode + 1}/{cfg.test.episodes}' + util.fg.rs)

        if episode % cfg.common.round_episodes == 0:
            obs = env.reset()
            orchestrator.reset_memory()

        if cfg.common.shuffle:
            orchestrator.shuffle()

        if cfg.negotiation.use:
            orchestrator.negotiation()

        choices = orchestrator.decisions(obs, epsilon)
        obs, rewards, elo = env.play(choices)
        elo_test.append(elo)
        orchestrator.rewarding(rewards)

    metrics = orchestrator.get_metrics()
    metrics['time'] = time.time() - start_time
    metrics['game'] = game
    metrics['elo_train'] = elo_train
    metrics['elo_test'] = elo_test
    return metrics


if __name__ == '__main__':
    cfg = util.load_config('config/default.yaml')

    metrics_dict = run_game(cfg=cfg, game='test2vs1_32nspace_0.0001', debug=True)

    util.log_metrics(metrics_dict, cfg)
    util.print_game_stats(metrics_dict, cfg)
