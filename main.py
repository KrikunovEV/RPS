from AttackAndDefend import AADEnvironment
from AgentOrchestrator import Orchestrator
import config as cfg
import os
import time


def main(model_type: cfg.ModelType):
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
            if model_type == cfg.ModelType.rnn:
                orchestrator.reset_h()
        if cfg.shuffle:
            orchestrator.shuffle()
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
        orchestrator.negotiation(obs)
        choices = orchestrator.decisions(obs, epsilon)
        obs, rewards = env.play(choices)
        orchestrator.rewarding(rewards)

        if rewards[2] > rewards[0] and rewards[1] > rewards[0]:
            result += 1

    '''
    directory_path = os.path.join('attention' if attention else 'base', f'{epoch}')
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
    orchestrator.plot_metrics(directory=directory_path)  # cfg.metric_directory
    '''
    print(f'PID: {os.getpid()}, coops: {result}/{cfg.test_episodes}')

    return result


if __name__ == '__main__':

    start_time = time.time()
    coop_result = main(cfg.ModelType.rnn)

    print(f'Time: {time.time() - start_time}')
    print(f'coop: {coop_result}/{cfg.test_episodes}')
