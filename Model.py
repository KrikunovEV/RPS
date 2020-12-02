import torch.nn as nn


class PredictionModel(nn.Module):

    def __init__(self, in_space: int, out_space: int):
        super(PredictionModel, self).__init__()
        self.linear = nn.Linear(in_space, out_space)

    def forward(self, obs):
        return self.linear(obs)


class GenerationModel(nn.Module):

    def __init__(self, in_space: int, out_space: int, steps):
        super(GenerationModel, self).__init__()
        self.generators = nn.ModuleList([nn.Linear(in_space, out_space) for _ in range(steps)])

    def forward(self, obs, step):
        return self.generators[step](obs)


class DecisionModel(nn.Module):

    def __init__(self, in_space: int, out_space: int):
        super(DecisionModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_space, in_space),
            #nn.ReLU(),
        )
        self.attack_policy = nn.Linear(in_space, out_space)
        self.defend_policy = nn.Linear(in_space, out_space)

    def forward(self, obs):
        obs = self.linear(obs)
        attack_logits = self.attack_policy(obs)
        defend_logits = self.defend_policy(obs)
        return attack_logits, defend_logits
