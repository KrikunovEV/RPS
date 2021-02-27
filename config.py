from easydict import EasyDict


cfg = EasyDict()
cfg.n_agents = 1  # number of agents who won't negotiate
cfg.lr = 0.005
cfg.train_episodes = 2000
cfg.test_episodes = 200
cfg.gamma = 0.9
cfg.rounds = 2
cfg.entropy_coef = 0.1

cfg.negot = EasyDict()
cfg.negot.n_agents = 2  # number of agents who will negotiate
cfg.negot.message_space = 10
cfg.negot.steps = 2

cfg.players = cfg.n_agents + cfg.negot.n_agents
cfg.pe_steps = 0.25 / (cfg.negot.message_space * cfg.players)
