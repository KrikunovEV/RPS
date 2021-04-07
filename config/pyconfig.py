from easydict import EasyDict


# CONFIG
config = EasyDict()

config.mp = EasyDict()
config.mp.cores = 7
config.mp.epochs = 1000
config.mp.stat_file = 'stat.pickle'
config.mp.model_list = ['siam_mlp']

config.common = EasyDict()
config.common.use_obs = True
config.common.shuffle = False
config.common.round_episodes = 20
config.common.players = 3
config.common.experiment_dir = '../experiments'
config.common.experiment_name = 'main'
config.common.logging = ['local']

config.train = EasyDict()
config.train.do_backward = True  # use False only for Random (non-trainable) Agents!
config.train.episodes = 3000
config.train.lr = 0.001
config.train.gamma = 1.
config.train.hidden_size = 32
config.train.value_loss_penalize = 0.5
config.train.entropy_penalize = 0.05
config.train.epsilon_upper = 0.5
config.train.epsilon_lower = 0.001

config.test = EasyDict()
config.test.episodes = 100

config.negotiation = EasyDict()
config.negotiation.use = True
config.negotiation.is_channel_open = True
config.negotiation.space = 3
config.negotiation.agents = 2
config.negotiation.steps = [2]

config.embeddings = EasyDict()
config.embeddings.use = False
config.embeddings.space = 64
