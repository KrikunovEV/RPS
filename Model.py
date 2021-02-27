import torch.nn as nn


class NegotiationModel(nn.Module):

    def __init__(self, in_space: int, out_space: int):
        super(NegotiationModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_space, in_space),
            nn.LeakyReLU(),
            nn.Linear(in_space, out_space)
        )

    def forward(self, obs):
        return self.linear(obs)


class DecisionModel(nn.Module):

    def __init__(self, in_space: int, out_space: int):
        super(DecisionModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_space, in_space),
            nn.LeakyReLU()
        )
        self.a_policy = nn.Linear(in_space, out_space)
        self.d_policy = nn.Linear(in_space, out_space)

    def forward(self, obs):
        data = self.linear(obs)
        return self.a_policy(data), self.d_policy(data)
