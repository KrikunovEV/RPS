import torch


class PredictionModel(torch.nn.Module):
    """
    This model tries to predict message of other player
    """

    def __init__(self, obs_space: int, action_space: int):
        super(PredictionModel, self).__init__()
        self.linear = torch.nn.Linear(obs_space, action_space)

    def forward(self, obs):
        return self.linear(obs)


class GenerationModel(torch.nn.Module):
    """
    This model generates message
    """

    def __init__(self, obs_space, action_space):
        super(GenerationModel, self).__init__()
        self.linear = torch.nn.Linear(obs_space, action_space)

    def forward(self, obs):
        return self.linear(obs)
