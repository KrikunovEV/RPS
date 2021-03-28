from AttackAndDefend import AADEnvironment
from AgentOrchestrator import Orchestrator
import config as cfg
import time
import os


def main(id, model_type: cfg.ModelType, log: cfg.LogType):
    env = AADEnvironment(players=cfg.players)
    orchestrator = Orchestrator(obs_space=env.get_obs_space(), action_space=env.get_action_space(),
                                model_type=model_type, cfg=cfg)

    orchestrator.set_eval(eval=False)
    obs = env.reset()
    epsilon = cfg.epsilon_upper
    for episode in range(cfg.train_episodes):
        #print(f'Train episode: {episode + 1}/{cfg.train_episodes}')
        if episode != 0 and episode % cfg.test_episodes == 0:
            obs = env.reset()
            if model_type == cfg.ModelType.baseline_rnn:
                orchestrator.reset_h()
        if cfg.shuffle:
            orchestrator.shuffle()
        if cfg.use_negotiation:
            orchestrator.negotiation(obs)
        choices = orchestrator.decisions(obs, epsilon)
        obs, rewards = env.play(choices)
        orchestrator.rewarding(rewards)
        orchestrator.train()
        epsilon -= cfg.epsilon_step

    orchestrator.set_eval(eval=True)
    obs = env.reset()
    orchestrator.reset_h()
    result = 0
    for episode in range(cfg.test_episodes):
        # print(f'Test episode: {episode + 1}/{cfg.test_episodes}')
        if cfg.shuffle:
            orchestrator.shuffle()
        if cfg.use_negotiation:
            orchestrator.negotiation(obs)
        choices = orchestrator.decisions(obs, epsilon)
        obs, rewards = env.play(choices)
        orchestrator.rewarding(rewards)

        if rewards[2] > rewards[0] and rewards[1] > rewards[0]:
            result += 1

    if log == cfg.LogType.show:
        orchestrator.plot_metrics(directory=None)
    elif log == cfg.LogType.local:
        directory = os.path.join(cfg.metric_directory, model_type.name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        directory = os.path.join(directory, f'{id}')
        if not os.path.exists(directory):
            os.mkdir(directory)
        orchestrator.plot_metrics(directory=directory)
    elif log == cfg.LogType.mlflow:
        pass

    return result


if __name__ == '__main__':
    if not cfg.Train:
        print(f'The value of cfg.Train is {cfg.Train}. Is this expected?')

    print(f'Use negotiation: {cfg.use_negotiation}')
    print(f'Use embeddings: {cfg.use_embeddings}')

    start_time = time.time()
    coop_result = main('test', cfg.ModelType.attention, cfg.LogType.show)

    print(f'Time: {time.time() - start_time}')
    print(f'coop: {coop_result}/{cfg.test_episodes}')
