import torch


class PredictionModel(torch.nn.Module):

    def __init__(self, n_agents: int, message_space: int):
        super(PredictionModel, self).__init__()
        self.linear = torch.nn.Linear(n_agents * message_space, (n_agents - 1) * message_space)

    def forward(self, obs):
        return self.linear(obs)


class GenerationModel(torch.nn.Module):

    def __init__(self, n_agents: int, message_space: int):
        super(GenerationModel, self).__init__()
        self.linear = torch.nn.Linear(n_agents * message_space, message_space)

    def forward(self, obs):
        return self.linear(obs)


class DecisionModel(torch.nn.Module):

    def __init__(self, obs_space: int, action_space: int):
        super(DecisionModel, self).__init__()
        self.mlp = torch.nn.Sequential(
            #torch.nn.Linear(obs_space, obs_space),
            #torch.nn.Sigmoid(),
            torch.nn.Linear(obs_space, action_space)
        )

    def forward(self, obs):
        return self.mlp(obs)
