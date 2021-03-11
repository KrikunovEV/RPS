import torch.nn as nn


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


class DecisionModel(nn.Module):

    def __init__(self, in_space: int, out_space: int):
        super(DecisionModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_space, in_space // 2),
            nn.LeakyReLU()
        )
        self.policy = nn.Linear(in_space // 2, out_space)
        self.V = nn.Linear(in_space // 2, 1)

    def forward(self, obs):
        obs = self.linear(obs)
        return self.policy(obs), self.V(obs)
