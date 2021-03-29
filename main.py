from OffendAndDefend import OADEnvironment
from AgentOrchestrator import Orchestrator
import config as cfg
import time
import os


def run(epoch, model_type: cfg.ModelType, debug: bool = False):

    if model_type == cfg.ModelType.attention and not cfg.use_negotiation and not cfg.use_embeddings:
        raise Exception(f'You can not use attention model if negotiation ({cfg.use_negotiation}) and embedding '
                        f'({cfg.use_embeddings}) are disabled both.')

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
    result = 0
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

        if rewards[2] > rewards[0] and rewards[1] > rewards[0]:
            result += 1

    if cfg.logging == cfg.LogType.show:
        orchestrator.plot_metrics(directory=None)
    elif cfg.logging == cfg.LogType.local:
        if not os.path.exists(cfg.metric_directory):
            os.mkdir(cfg.metric_directory)
        directory = os.path.join(cfg.metric_directory, model_type.name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        directory = os.path.join(directory, f'{epoch}')
        if not os.path.exists(directory):
            os.mkdir(directory)
        orchestrator.plot_metrics(directory=directory)
    elif cfg.logging == cfg.LogType.mlflow:
        print(f'Log type mlflow not implemented. Logs are not saved.')

    return result


if __name__ == '__main__':
    cfg.print_config()

    start_time = time.time()
    coop_result = run('test', cfg.ModelType.attention)

    print(f'Time: {time.time() - start_time}')
    print(f'coop: {coop_result}/{cfg.test_episodes}')
