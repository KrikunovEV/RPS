from easydict import EasyDict


cfg = EasyDict()
cfg.n_agents = 1  # number of agents who won't negotiate
cfg.lr = 0.01
cfg.train_episodes = 2000
cfg.test_episodes = 200
cfg.gamma = 0.99
cfg.rounds = 3

cfg.negot = EasyDict()
cfg.negot.n_agents = 4  # number of agents who will negotiate
cfg.negot.message_space = 3
cfg.negot.steps = 3

cfg.players = cfg.n_agents + cfg.negot.n_agents
