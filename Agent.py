import torch
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
from Model import DecisionModel, NegotiationModel


class Agent:
    def __init__(self, id: int, obs_space: int, action_space: int, cfg):
        self.cfg = cfg
        self.id = id
        self.mask_id = id
        self.negotiable = True if id >= cfg.n_agents else False
        self.agent_label = f'{id}' + (' negotiable' if self.negotiable else '')
        self.eval = False

        obs_space = obs_space + cfg.players * cfg.negot.message_space
        self.model = DecisionModel(obs_space, action_space)
        self.negot_model = NegotiationModel(obs_space, cfg.negot.message_space)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.negot_model.parameters()), lr=cfg.lr)

        self.logs = []
        self.entropy = []
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
            negotiate_logits = self.negot_model(obs)
            negotiate_policy = functional.softmax(negotiate_logits, dim=-1)
            negotiate_action = np.random.choice(negotiate_policy.shape[0], p=negotiate_policy.detach().numpy())

            if not self.eval:
                self.logs.append(torch.log(negotiate_policy[negotiate_action]))
                self.entropy.append((negotiate_policy * torch.log_softmax(negotiate_logits, dim=-1)).sum())
                self.reward.append(0)

            message[negotiate_action] = 1.

        return message

    def make_decision(self, obs, messages):
        messages = torch.cat(messages)
        if not self.negotiable:
            messages = torch.zeros_like(messages)
        data = torch.cat((torch.Tensor(obs), messages))
        a_logits, d_logits = self.model(data)
        #a_logits[self.mask_id] = float('-inf')
        #d_logits[self.mask_id] = float('-inf')
        a_policy = functional.softmax(a_logits, dim=-1)
        d_policy = functional.softmax(d_logits, dim=-1)
        a_action = np.random.choice(a_policy.shape[0], p=a_policy.detach().numpy())
        d_action = np.random.choice(d_policy.shape[0], p=d_policy.detach().numpy())

        if not self.eval:
            self.logs.append(torch.log(a_policy[a_action] * d_policy[d_action]))
            a_entropy = (torch.softmax(a_logits, dim=-1) * torch.log_softmax(a_logits, dim=-1)).sum()
            d_entropy = (torch.softmax(d_logits, dim=-1) * torch.log_softmax(d_logits, dim=-1)).sum()
            self.entropy.append(a_entropy + d_entropy)

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
            loss = loss - G * self.logs[i] - self.cfg.entropy_coef * self.entropy[i]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_metric.append(loss.item())

        self.reward = []
        self.logs = []
        self.entropy = []

    def get_label(self):
        return self.agent_label
