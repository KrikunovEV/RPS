import torch
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
from Model import DecisionModel


class Agent:
    def __init__(self, id: int, obs_space: int, action_space: int, cfg):
        obs_space = obs_space + cfg.players * cfg.negot.message_space

        self.cfg = cfg
        self.id = id
        self.negotiable = True if id >= cfg.n_agents else False
        self.agent_label = f'{id}' + (' negotiable' if self.negotiable else '')
        self.eval = False

        self.model = DecisionModel(obs_space, action_space, cfg.negot.message_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)

        self.logs = []
        self.reward = []

        self.loss_metric = []
        self.reward_metric = []
        self.reward_eval_metric = []

    def set_eval(self, eval: bool):
        self.eval = eval

    def negotiate(self, messages, obs):
        message = torch.zeros(self.cfg.negot.message_space)
        if self.negotiable:
            obs = torch.cat((torch.Tensor(obs), torch.cat(messages)))
            negotiate_logits = self.model(obs, negotiate=True)
            negotiate_policy = functional.softmax(negotiate_logits, dim=-1)
            negotiate_action = np.random.choice(negotiate_policy.shape[0], p=negotiate_policy.detach().numpy())

            if not self.eval:
                self.logs.append(torch.log(negotiate_policy[negotiate_action]))
                self.reward.append(0)

            message[negotiate_action] = 1.

        return message

    def make_decision(self, obs, messages):
        data = torch.cat((torch.Tensor(obs), torch.cat(messages)))
        a_logits, d_logits = self.model(data, negotiate=False)
        a_policy = functional.softmax(a_logits, dim=-1)
        d_policy = functional.softmax(d_logits, dim=-1)
        a_action = np.random.choice(a_policy.shape[0], p=a_policy.detach().numpy())
        d_action = np.random.choice(d_policy.shape[0], p=d_policy.detach().numpy())

        if not self.eval:
            self.logs.append(torch.log(a_policy[a_action] * d_policy[d_action]))

        return [a_action, d_action]

    def rewarding(self, reward):
        if self.eval:
            self.reward_eval_metric.append(reward)
        else:
            self.reward.append(reward)
            self.reward_metric.append(reward)

    def train(self):
        G = 0
        loss = 0
        for i in reversed(range(len(self.reward))):
            G = self.reward[i] + self.cfg.gamma * G
            loss = loss - G * self.logs[i]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_metric.append(loss.item())

        self.reward = []
        self.logs = []

    def get_label(self):
        return self.agent_label
