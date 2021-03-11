from easydict import EasyDict


cfg = EasyDict()
cfg.n_agents = 1  # number of agents who won't negotiate
cfg.train_episodes = 1000
cfg.test_episodes = 500
cfg.lr = 0.005
cfg.gamma = 0.9
cfg.entropy_coef = 0.001

cfg.negot = EasyDict()
cfg.negot.n_agents = 2  # number of agents who will negotiate
cfg.negot.steps = 2

cfg.players = cfg.n_agents + cfg.negot.n_agents
