import torch.nn as nn
import torch


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


class SiamRNNModel(nn.Module):

    def __init__(self, obs_space: int, action_space: int, cfg):
        super(SiamRNNModel, self).__init__()
        self.cfg = cfg

        if cfg.use_negotiation:
            obs_space += cfg.message_space + 1

        if cfg.use_embeddings:
            obs_space += cfg.embedding_space

        self.rnn = nn.GRUCell(obs_space, cfg.hidden_size)
        self.policies = nn.Linear(cfg.hidden_size, 2)
        self.V = nn.Linear(2 * action_space, 1)

    def get_h(self):
        return None

    def forward(self, obs, messages, embeddings, h):
        h = torch.zeros((1, self.cfg.hidden_size))
        actions = []

        for p in range(self.cfg.players):
            agent_obs = obs[p]
            if self.cfg.use_negotiation:
                agent_obs = torch.cat((agent_obs, messages[p]))
            if self.cfg.use_embeddings:
                agent_obs = torch.cat((agent_obs, embeddings[p]))
            h = self.rnn(agent_obs.unsqueeze(0), h)
            actions.append(self.policies(h[0]))

        actions = torch.cat(actions)
        return actions[::2], actions[1::2], self.V(actions), None


class AttentionModel(nn.Module):

    def __init__(self, obs_space: int, action_space: int, cfg):
        super(AttentionModel, self).__init__()
        self.cfg = cfg

        if cfg.use_negotiation:
            obs_space += cfg.message_space + 1

        if cfg.use_embeddings:
            obs_space += cfg.embedding_space

        self.rnn = nn.GRUCell(obs_space, cfg.hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden_size, 2),
            nn.LeakyReLU()
        )

        self.a_policy = nn.Linear(cfg.hidden_size, action_space)
        self.d_policy = nn.Linear(cfg.hidden_size, action_space)
        self.V = nn.Linear(cfg.hidden_size, 1)

    def get_h(self):
        return None

    def forward(self, obs, messages, embeddings, h):
        h = [torch.zeros((1, self.cfg.hidden_size))]
        actions = []
        for p in range(self.cfg.players):
            agent_obs = obs[p]
            if self.cfg.use_negotiation:
                agent_obs = torch.cat((agent_obs, messages[p]))
            if self.cfg.use_embeddings:
                agent_obs = torch.cat((agent_obs, embeddings[p]))
            h.append(self.rnn(agent_obs.unsqueeze(0), h[-1]))
            actions.append(self.ff(h[-1][0]))

        actions = torch.cat(actions)
        h = torch.stack(h[1:]).squeeze()
        a_attention = torch.sum(torch.softmax(actions[::2], -1).reshape(-1, 1) * h, dim=0)
        d_attention = torch.sum(torch.softmax(actions[1::2], -1).reshape(-1, 1) * h, dim=0)

        return self.a_policy(a_attention), self.d_policy(d_attention), self.V(a_attention + d_attention), None


class BaselineRNNModel(nn.Module):

    def __init__(self, obs_space: int, action_space: int, cfg):
        super(BaselineRNNModel, self).__init__()
        self.cfg = cfg

        obs_space = obs_space * cfg.players

        if cfg.use_negotiation:
            obs_space += (cfg.message_space + 1) * cfg.players

        if cfg.use_embeddings:
            obs_space += cfg.embedding_space * cfg.players

        self.rnn = nn.GRUCell(obs_space, cfg.hidden_size)
        self.a_policy = nn.Linear(cfg.hidden_size, action_space)
        self.d_policy = nn.Linear(cfg.hidden_size, action_space)
        self.V = nn.Linear(cfg.hidden_size, 1)

    def get_h(self):
        return torch.zeros((1, self.cfg.hidden_size))

    def forward(self, obs, messages, embeddings, h):
        obs = obs.reshape(-1)

        if self.cfg.use_negotiation:
            messages = messages.reshape(-1)
            obs = torch.cat((obs, messages))

        if self.cfg.use_embeddings:
            embeddings = embeddings.reshape(-1)
            obs = torch.cat((obs, embeddings))

        new_h = self.rnn(obs.unsqueeze(0), h)
        return self.a_policy(new_h[0]), self.d_policy(new_h[0]), self.V(new_h[0]), new_h.data


class BaselineMLPModel(nn.Module):

    def __init__(self, obs_space: int, action_space: int, cfg):
        super(BaselineMLPModel, self).__init__()
        self.cfg = cfg

        obs_space = obs_space * cfg.players

        if cfg.use_negotiation:
            obs_space += (cfg.message_space + 1) * cfg.players

        if cfg.use_embeddings:
            obs_space += cfg.embedding_space * cfg.players

        self.linear = nn.Sequential(
            nn.Linear(obs_space, obs_space // 2),
            nn.LeakyReLU()
        )
        self.a_policy = nn.Linear(obs_space // 2, action_space)
        self.d_policy = nn.Linear(obs_space // 2, action_space)
        self.V = nn.Linear(obs_space // 2, 1)

    def get_h(self):
        return None

    def forward(self, obs, messages, embeddings, h):
        obs = obs.reshape(-1)

        if self.cfg.use_negotiation:
            messages = messages.reshape(-1)
            obs = torch.cat((obs, messages))

        if self.cfg.use_embeddings:
            embeddings = embeddings.reshape(-1)
            obs = torch.cat((obs, embeddings))

        obs = self.linear(obs)
        return self.a_policy(obs), self.d_policy(obs), self.V(obs), None
