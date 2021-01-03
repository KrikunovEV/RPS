import torch.nn as nn


class DecisionModel(nn.Module):

    def __init__(self, in_space: int, out_space: int, message_space: int):
        super(DecisionModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_space, in_space),
            nn.ReLU(),
        )
        self.attack_policy = nn.Linear(in_space, out_space)
        self.defend_policy = nn.Linear(in_space, out_space)
        self.negotiate_policy = nn.Linear(in_space, message_space)

    def forward(self, obs, negotiate: bool):
        obs = self.linear(obs)
        if negotiate:
            negotiate_logits = self.negotiate_policy(obs)
            return negotiate_logits
        else:
            attack_logits = self.attack_policy(obs)
            defend_logits = self.defend_policy(obs)
            return attack_logits, defend_logits
