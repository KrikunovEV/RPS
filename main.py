from AttackAndDefend import AADEnvironment
from AgentOrchestrator import Orchestrator
from config import cfg


env = AADEnvironment(players=cfg.players)
orchestrator = Orchestrator(obs_space=env.get_obs_space(), action_space=env.get_action_space(), cfg=cfg)

orchestrator.set_eval(eval=False)
for episode in range(cfg.train_episodes):
    print(f'Train episode: {episode}')
    obs = env.reset()
    for r in range(cfg.rounds):
        orchestrator.negotiation(obs)
        choices = orchestrator.decisions(obs)
        obs, rewards = env.play(choices)
        orchestrator.rewarding(rewards)
    orchestrator.train()

orchestrator.set_eval(eval=True)
for episode in range(cfg.test_episodes):
    print(f'Test episode: {episode}')
    obs = env.reset()
    for r in range(cfg.rounds):
        orchestrator.negotiation(obs)
        choices = orchestrator.decisions(obs)
        obs, rewards = env.play(choices)
        orchestrator.rewarding(rewards)

orchestrator.plot_metrics('test')
