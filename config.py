from easydict import EasyDict
from Agent import MessageType
from functools import reduce


cfg = EasyDict()
cfg.n_agents = 1  # number of agents who won't negotiate
cfg.lr = 0.001
cfg.train_episodes = 1000
cfg.test_episodes = 200
cfg.gamma = 0.99

cfg.negot = EasyDict()
cfg.negot.teams = [2]  # list of number of agents who negotiate within a team
cfg.negot.message_type = MessageType.Numerical
cfg.negot.message_space = 10
cfg.negot.lr = 0.001
cfg.negot.steps = 10

cfg.players = cfg.n_agents + reduce(lambda a, b: a+b, cfg.negot.teams)
