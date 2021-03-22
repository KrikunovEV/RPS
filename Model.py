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

    def __init__(self, message_space: int, obs_space: int, n_players: int, hidden_size: int):
        super(AttDecisionModel, self).__init__()

        self.message_space = message_space
        self.n_players = n_players
        self.hidden_size = hidden_size

        self.rnn = nn.GRUCell(message_space, hidden_size)
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


class RNNDecisionModel(nn.Module):

    def __init__(self, in_space: int, out_space: int, hidden_size: int):
        super(RNNDecisionModel, self).__init__()
        self.rnn = nn.GRUCell(in_space, hidden_size)
        self.a_policy = nn.Linear(hidden_size, out_space)
        self.d_policy = nn.Linear(hidden_size, out_space)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, obs, h):
        new_h = self.rnn(obs.unsqueeze(0), h)
        return self.a_policy(new_h[0]), self.d_policy(new_h[0]), self.V(new_h[0]), new_h
