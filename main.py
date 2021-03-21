from AttackAndDefend import AADEnvironment
from AgentOrchestrator import Orchestrator
import config as cfg
import numpy as np
import os
import time
import multiprocessing as mp


def main(attention):
    result = 0
    env = AADEnvironment(players=cfg.players)
    orchestrator = Orchestrator(obs_space=env.get_obs_space(), action_space=env.get_action_space(),
                                attention=attention, cfg=cfg)

    orchestrator.set_eval(eval=False)
    obs = env.reset()
    cfg.epsilon = cfg.epsilon_upper
    for episode in range(cfg.train_episodes):
        #print(f'Epoch: {epoch + 1}/{cfg.epochs}, Train episode: {episode + 1}/{cfg.train_episodes}')
        if episode != 0 and episode % cfg.test_episodes == 0:
            orchestrator.reset_h()
            obs = env.reset()
        #orchestrator.shuffle()
        orchestrator.negotiation(obs)
        choices = orchestrator.decisions(obs)
        obs, rewards = env.play(choices)
        orchestrator.rewarding(rewards)
        orchestrator.train()
        cfg.epsilon -= cfg.epsilon_step

    orchestrator.set_eval(eval=True)
    orchestrator.reset_h()
    obs = env.reset()
    rewards_counter = np.zeros(cfg.players)
    for episode in range(cfg.test_episodes):
        print(f'Test episode: {episode + 1}/{cfg.test_episodes}')
        #orchestrator.shuffle()
        orchestrator.negotiation(obs)
        choices = orchestrator.decisions(obs)
        obs, rewards = env.play(choices)
        orchestrator.rewarding(rewards)
        rewards_counter += rewards

    if rewards_counter[2] > rewards_counter[0] and rewards_counter[1] > rewards_counter[0]:
        result = 1

    '''
    directory_path = os.path.join('attention' if attention else 'base', f'{epoch}')
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
    orchestrator.plot_metrics(directory=directory_path)  # cfg.metric_directory
    '''

    print(f'{os.getpid()} result {result}')

    return result


if __name__ == '__main__':

    attentions = [False] * cfg.epochs + [True] * cfg.epochs
    chunksize = (len(attentions) // cfg.cores) + 1

    start_time = time.time()
    with mp.Pool(processes=cfg.cores) as pool:
        results = pool.map(main, attentions, chunksize)
        base_counts = np.sum(results[:cfg.epochs])
        att_counts = np.sum(results[cfg.epochs:])

    print(f'Time: {time.time() - start_time}')
    print(f'baseline: {base_counts}/{cfg.epochs}\nattention: {att_counts}/{cfg.epochs}')
