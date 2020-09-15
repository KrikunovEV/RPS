import torch
import torch.nn as nn


class ModelCom(nn.Module):

    def __init__(self, obs_space, action_space):
        super(ModelCom, self).__init__()
        self.policy = nn.Linear(obs_space, action_space)

    def forward(self, obs):
        policy = self.policy(torch.Tensor(obs))
        return policy
