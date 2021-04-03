from OffendAndDefend import OADEnvironment
from AgentOrchestrator import Orchestrator
import config as cfg
import time
import os


def run(epoch, model_type: cfg.ModelType, debug: bool = False):

    if model_type == cfg.ModelType.attention and not cfg.use_negotiation and not cfg.use_embeddings:
        raise Exception(f'You can not use attention model if negotiation ({cfg.use_negotiation}) and embedding '
                        f'({cfg.use_embeddings}) are disabled both.')

    if isinstance(cfg.negotiation_steps, list) and len(cfg.negotiation_steps) != cfg.players:
        raise Exception('cfg.negotiation_steps is a list but len not equal to number of players.')

    env = OADEnvironment(players=cfg.players, debug=debug)
    orchestrator = Orchestrator(obs_space=env.get_obs_space(),
                                action_space=env.get_action_space(),
                                model_type=model_type, cfg=cfg)

    orchestrator.set_eval(eval=False)
    epsilon = cfg.epsilon_upper
    obs = None
    for episode in range(cfg.train_episodes):
        if debug:
            print(f'Epoch: {epoch}, train episode: {episode + 1}/{cfg.train_episodes}')

        if episode % cfg.round_episodes == 0:
            obs = env.reset()
            if cfg.is_require_reset(model_type):
                orchestrator.reset_h()

        if cfg.shuffle:
            obs = orchestrator.shuffle(obs)

        if cfg.use_negotiation:
            orchestrator.negotiation(obs)

        choices = orchestrator.decisions(obs, epsilon)
        obs, rewards = env.play(choices)
        orchestrator.rewarding(rewards)
        orchestrator.train()
        epsilon -= cfg.epsilon_step

    orchestrator.set_eval(eval=True)
    obs = None
    result = {'1 & 2 vs 3': 0, '2 & 3 vs 1': 0, '1 & 3 vs 2': 0}
    for episode in range(cfg.test_episodes):
        if debug:
            print(f'Epoch: {epoch}, test episode: {episode + 1}/{cfg.test_episodes}')

        if episode % cfg.round_episodes == 0:
            obs = env.reset()
            if cfg.is_require_reset(model_type):
                orchestrator.reset_h()

        if cfg.shuffle:
            obs = orchestrator.shuffle(obs)

        if cfg.use_negotiation:
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

    if cfg.logging == cfg.LogType.show:
        orchestrator.plot_metrics(directory=None)
    elif cfg.logging == cfg.LogType.local:
        directory = os.path.join(cfg.metric_directory, model_type.name, cfg.experiment_name, f'epoch_{epoch}')
        os.makedirs(directory, exist_ok=True)
        orchestrator.plot_metrics(directory=directory)
    elif cfg.logging == cfg.LogType.mlflow:
        print(f'Log type mlflow not implemented. Logs are not saved.')

    return result


if __name__ == '__main__':
    cfg.print_config()

    start_time = time.time()
    coops = run('main', cfg.ModelType.siam_mlp, debug=True)

    print(f'Time: {time.time() - start_time}')
    print('Coops:')
    for (key, value) in coops.items():
        print(f'{key}: {value}/{cfg.test_episodes} ({value / cfg.test_episodes})')
