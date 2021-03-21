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


class AttDecisionModel(nn.Module):

    def __init__(self, message_space: int, obs_space: int, n_players: int):
        super(AttDecisionModel, self).__init__()

        self.message_space = message_space
        self.n_players = n_players
        self.attention = nn.GRUCell(message_space, 2)

        self.a_policy = nn.Linear(obs_space + n_players * 2, n_players)
        self.d_policy = nn.Linear(obs_space + n_players * 2, n_players)
        self.V = nn.Linear(obs_space + n_players * 2, 1)

    def forward(self, obs, messages, h):

        attention = []
        for p in range(self.n_players):
            attention.append(self.attention(messages[p * self.message_space:(p + 1) * self.message_space].unsqueeze(
                0), h)[0])
        attention = torch.cat(attention)
        obs = torch.cat((obs, attention))

        return self.a_policy(obs), self.d_policy(obs), self.V(obs)


class DecisionModel(nn.Module):

    def __init__(self, in_space: int, out_space: int):
        super(DecisionModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_space, in_space // 2),
            nn.LeakyReLU()
        )
        self.a_policy = nn.Linear(in_space // 2, out_space)
        self.d_policy = nn.Linear(in_space // 2, out_space)
        self.V = nn.Linear(in_space // 2, 1)

    def forward(self, obs):
        obs = self.linear(obs)
        return self.a_policy(obs), self.d_policy(obs), self.V(obs)

