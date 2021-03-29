import torch.nn as nn
import torch


class NegotiationModel(nn.Module):

    def __init__(self, in_space: int, out_space: int):
        super(NegotiationModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_space, in_space // 2),
            nn.LeakyReLU()
        )
        self.policy = nn.Linear(in_space // 2, out_space)
        self.V = nn.Linear(in_space // 2, 1)

    def forward(self, obs):
        obs = self.linear(obs)
        return self.policy(obs), self.V(obs)


class AttentionModel(nn.Module):

    def __init__(self, obs_space: int, action_space: int, message_space: int, embedding_space: int, hidden_size: int,
                 cfg):
        super(AttentionModel, self).__init__()

        self.cfg = cfg
        self.message_space = message_space
        self.n_players = cfg.n_players
        self.hidden_size = hidden_size

        if cfg.use_negotiation:
            obs_space += message_space

        if cfg.use_embedding:
            obs_space += embedding_space

        self.rnn = nn.GRUCell(rnn_in_space, hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(obs_space + hidden_size, 2),
            nn.LeakyReLU()
        )

        self.a_policy = nn.Linear(hidden_size, n_players)
        self.d_policy = nn.Linear(hidden_size, n_players)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, obs, messages):
        h = [torch.zeros(self.hidden_size)]
        actions = []
        for p in range(self.n_players):
            h.append(self.rnn(messages[p * self.message_space:(p + 1) * self.message_space].unsqueeze(0),
                              h[-1].unsqueeze(0))[0])
            actions.append(self.ff(torch.cat((obs, h[-1]))))

        actions = torch.cat(actions)
        h = torch.stack(h[1:])
        attack_attention = torch.sum(torch.softmax(actions[::2], -1).reshape(-1, 1) * h, dim=0)
        defend_attention = torch.sum(torch.softmax(actions[1::2], -1).reshape(-1, 1) * h, dim=0)

        return self.a_policy(attack_attention), self.d_policy(defend_attention), self.V(attack_attention + defend_attention)


class BaselineRNNModel(nn.Module):

    def __init__(self, obs_space: int, action_space: int, message_space: int, embedding_space: int, hidden_size: int,
                 cfg):
        super(BaselineRNNModel, self).__init__()
        self.cfg = cfg

        if cfg.use_negotiation:
            obs_space += message_space

        if cfg.use_embedding:
            obs_space += embedding_space

        self.rnn = nn.GRUCell(obs_space, hidden_size)
        self.a_policy = nn.Linear(hidden_size, action_space)
        self.d_policy = nn.Linear(hidden_size, action_space)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, obs, messages, embeddings, h):

        if self.cfg.use_negotiation:
            obs = torch.cat((obs, torch.cat(messages)))

        if self.cfg.use_embedding:
            obs = torch.cat((obs, torch.cat(embeddings)))

        new_h = self.rnn(obs.unsqueeze(0), h)
        return self.a_policy(new_h[0]), self.d_policy(new_h[0]), self.V(new_h[0]), new_h


class BaselineMLPModel(nn.Module):

    def __init__(self, obs_space: int, action_space: int, message_space: int, embedding_space: int, cfg):
        super(BaselineMLPModel, self).__init__()
        self.cfg = cfg

        if cfg.use_negotiation:
            obs_space += message_space

        if cfg.use_embedding:
            obs_space += embedding_space

        self.linear = nn.Sequential(
            nn.Linear(obs_space, obs_space // 2),
            nn.LeakyReLU()
        )
        self.a_policy = nn.Linear(obs_space // 2, action_space)
        self.d_policy = nn.Linear(obs_space // 2, action_space)
        self.V = nn.Linear(obs_space // 2, 1)

    def forward(self, obs, messages, embeddings, h):

        if self.cfg.use_negotiation:
            obs = torch.cat((obs, torch.cat(messages)))

        if self.cfg.use_embedding:
            obs = torch.cat((obs, torch.cat(embeddings)))

        obs = self.linear(obs)
        return self.a_policy(obs), self.d_policy(obs), self.V(obs)


class DecisionModel(nn.Module):

    def __init__(self, obs_space: int, action_space: int, message_space: int, embedding_space: int, hidden_size: int,
                 cfg, model_type):
        super(DecisionModel, self).__init__()

        if model_type == cfg.ModelType.baseline_mlp:
            self.model = BaselineMLPModel(obs_space, action_space, message_space, embedding_space, cfg)
        else:
            raise Exception(f'Model type {model_type.name} has no implementation')

    def forward(self, obs, messages, embeddings, h):
        pass
