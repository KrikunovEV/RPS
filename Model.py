import torch
import torch.nn as nn
import torch.nn.functional as functional


class NegotiationModel(nn.Module):

    def __init__(self, obs_space: int, cfg):
        super(NegotiationModel, self).__init__()
        self.cfg = cfg
        obs_space *= cfg.common.players

        self.linear = nn.Sequential(
            nn.Linear(obs_space, obs_space // 2),
            nn.LeakyReLU()
        )
        self.policy = nn.Linear(obs_space // 2, cfg.negotiation.space)
        self.V = nn.Linear(obs_space // 2, 1)

    def forward(self, obs):
        obs = self.linear(obs.reshape(-1))
        return self.policy(obs), self.V(obs)


class AttentionNegotiationModel(nn.Module):

    def __init__(self, obs_space: int, cfg):
        super(AttentionNegotiationModel, self).__init__()
        self.cfg = cfg
        self.scale = torch.sqrt(torch.Tensor([cfg.train.hidden_size]))

        var = 1. / (2. * obs_space)
        self.WQ = nn.Parameter(torch.normal(0, var, (obs_space, cfg.train.hidden_size), requires_grad=True))
        self.WK = nn.Parameter(torch.normal(0, var, (obs_space, cfg.train.hidden_size), requires_grad=True))
        self.WV = nn.Parameter(torch.normal(0, var, (obs_space, cfg.train.hidden_size), requires_grad=True))
        self.policy = nn.Linear(cfg.train.hidden_size, cfg.negotiation.space)
        self.V = nn.Linear(cfg.train.hidden_size, 1)

    def forward(self, obs, ind):
        query = obs[ind]
        if ind == 0:
            keys_values = obs[1:]
        elif ind == self.cfg.common.players - 1:
            keys_values = obs[:-1]
        else:
            keys_values = torch.cat((obs[:ind], obs[ind + 1:]))

        Q = torch.matmul(query, self.WQ)
        K = torch.matmul(keys_values, self.WK)
        V = torch.matmul(keys_values, self.WV)

        O = torch.matmul(Q, K.T)
        O = torch.div(O, self.scale)
        O = functional.softmax(O, dim=-1)
        context = torch.matmul(O, V)

        return self.policy(context), self.V(context)


class AttentionLayer(nn.Module):

    def __init__(self, cfg):
        super(AttentionLayer, self).__init__()
        self.cfg = cfg
        self.scale = torch.sqrt(torch.Tensor([cfg.train.hidden_size]))

        space = cfg.negotiation.space + cfg.common.players - 1
        self.WQ = nn.Parameter(torch.randn((space, cfg.train.hidden_size), requires_grad=True))
        self.WK = nn.Parameter(torch.randn((space, cfg.train.hidden_size), requires_grad=True))
        self.WV = nn.Parameter(torch.randn((space, cfg.train.hidden_size), requires_grad=True))
        self.FF = nn.Sequential(
            nn.Linear(cfg.train.hidden_size, cfg.train.hidden_size * 2),
            nn.LeakyReLU(),
            nn.Linear(cfg.train.hidden_size * 2, cfg.negotiation.space),
        )

    def forward(self, kv, q):
        Q = torch.matmul(q, self.WQ)
        K = torch.matmul(kv, self.WK)
        V = torch.matmul(kv, self.WV)

        O = torch.matmul(Q, K.T)
        O = torch.div(O, self.scale)
        O = functional.softmax(O, dim=-1)
        context = torch.matmul(O, V)

        messages = self.FF(context)

        return messages


class SiamMLPModel(nn.Module):

    def __init__(self, obs_space: int, action_space: int, cfg):
        super(SiamMLPModel, self).__init__()
        self.cfg = cfg

        self.policies = nn.Sequential(
            nn.Linear(obs_space, obs_space // 2),
            nn.LeakyReLU(),
            nn.Linear(obs_space // 2, 2),
        )
        self.V = nn.Linear(2 * action_space, 1)

    def forward(self, obs):
        actions = self.policies(obs).reshape(-1)
        return actions[::2], actions[1::2], self.V(actions)
