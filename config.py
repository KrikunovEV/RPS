from easydict import EasyDict
from Agent import MessageType


cfg = EasyDict()
cfg.n_agents = 2  # number of agents who won't negotiate
cfg.lr = 0.01
cfg.train_episodes = 1000
cfg.test_episodes = 200
cfg.gamma = 0.99
cfg.rounds = 50

cfg.negot = EasyDict()
cfg.negot.n_agents = 2  # number of agents who will negotiate
cfg.negot.message_type = MessageType.Numerical
cfg.negot.message_space = 16
cfg.negot.steps = 3

cfg.players = cfg.n_agents + cfg.negot.n_agents
