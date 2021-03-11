from AttackAndDefend import AADEnvironment
from AgentOrchestrator import Orchestrator
import config as cfg


env = AADEnvironment(players=cfg.players)
orchestrator = Orchestrator(obs_space=env.get_obs_space(), action_space=env.get_action_space(), cfg=cfg)

orchestrator.set_eval(eval=False)
obs = env.reset()
for episode in range(cfg.train_episodes):
    print(f'Train episode: {episode + 1}/{cfg.train_episodes}')
    #orchestrator.shuffle()
    orchestrator.negotiation(obs)
    choices = orchestrator.decisions(obs)
    obs, rewards = env.play(choices)
    orchestrator.rewarding(rewards)
    orchestrator.train()

orchestrator.set_eval(eval=True)
obs = env.reset()
for episode in range(cfg.test_episodes):
    print(f'Test episode: {episode + 1}/{cfg.test_episodes}')
    #orchestrator.shuffle()
    orchestrator.negotiation(obs)
    choices = orchestrator.decisions(obs)
    obs, rewards = env.play(choices)
    orchestrator.rewarding(rewards)

orchestrator.plot_metrics(directory=cfg.metric_directory)
