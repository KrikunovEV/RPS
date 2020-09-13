import torch
import torch.nn as nn


class ModelNeg(nn.Module):

    def __init__(self, obs_space, neg_space, action_space):
        super(ModelNeg, self).__init__()

        self.negotiation_nn = nn.Linear(obs_space + neg_space, 4)
        self.obs_nn = nn.Linear(obs_space, 4)
        self.policy = nn.Linear(8, action_space)

    def forward(self, obs, neg):
        obs = torch.Tensor(obs)
        neg = torch.Tensor(neg)

        obs_neg_state = torch.cat((obs, neg), dim=-1)
        neg_state = self.negotiation_nn(obs_neg_state)
        obs_state = self.obs_nn(obs)

        common_state = torch.cat((neg_state, obs_state), dim=-1)

        policy = self.policy(common_state)

        return policy


class Model(nn.Module):

    def __init__(self, obs_space, action_space):
        super(Model, self).__init__()

        self.obs_nn = nn.Linear(obs_space, 4)
        self.policy = nn.Linear(4, action_space)

    def forward(self, obs):
        obs = torch.Tensor(obs)
        obs_state = self.obs_nn(obs)
        policy = self.policy(obs_state)
        return policy
