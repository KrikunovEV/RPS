import numpy as np
from AttackAndDefend import AADEnvironment
from AgentOrchestrator import Orchestrator
from config import cfg


env = AADEnvironment(players=cfg.players)
orchestrator = Orchestrator(obs_space=env.get_obs_space(), action_space=env.get_action_space(), cfg=cfg)

orchestrator.set_eval(eval=False)
obs = env.reset()
for episode in range(cfg.train_episodes):
    print(f'Train episode: {episode + 1}/{cfg.train_episodes}')
    #orchestrator.shuffle()
    #for r in range(cfg.rounds):
    orchestrator.negotiation(obs)
    choices = orchestrator.decisions(obs)
    obs, rewards = env.play(choices)
    orchestrator.rewarding(rewards)
    orchestrator.train()

orchestrator.set_eval(eval=True)
A_CM, D_CM = np.zeros((cfg.players, cfg.players), dtype=np.int), np.zeros((cfg.players, cfg.players), dtype=np.int)
obs = env.reset()
for episode in range(cfg.test_episodes):
    print(f'Test episode: {episode + 1}/{cfg.test_episodes}')
    #orchestrator.shuffle()
    #for r in range(cfg.rounds):
    orchestrator.negotiation(obs)
    choices = orchestrator.decisions(obs)
    obs, rewards = env.play(choices)
    orchestrator.rewarding(rewards)
    for a, choice in enumerate(choices):
        A_CM[a, choice[0]] += 1
        D_CM[a, choice[1]] += 1

orchestrator.plot_metrics(A_CM, D_CM, directory='test')
