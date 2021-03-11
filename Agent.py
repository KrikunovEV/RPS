import torch
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
from Model import DecisionModel, NegotiationModel


class Agent:
    def __init__(self, id: int, obs_space: int, action_space: int, message_space: int, cfg):
        self.cfg = cfg
        self.id = id
        self.negotiable = True if id >= cfg.n_agents else False
        self.agent_label = f'{id}' + (' negotiable' if self.negotiable else '')
        self.eval = False

        self.message_space = message_space
        obs_space = obs_space + cfg.players * message_space
        self.model = DecisionModel(obs_space, action_space)
        self.negot_model = NegotiationModel(obs_space, action_space)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.negot_model.parameters()), lr=cfg.lr)

        self.logs = []
        self.value = []
        self.entropy = []
        self.reward = []

        self.loss_metric = []
        self.reward_metric = []
        self.reward_eval_metric = []

    def set_eval(self, eval: bool):
        self.eval = eval

    def negotiate(self, obs_negot):
        message = torch.zeros(self.message_space)
        if self.negotiable:
            negotiate_logits, negotiate_V = self.negot_model(obs_negot)
            negotiate_policy = functional.softmax(negotiate_logits, dim=-1)
            negotiate_action = np.random.choice(negotiate_policy.shape[0], p=negotiate_policy.detach().numpy())

            if not self.eval:
                self.logs.append(torch.log(negotiate_policy[negotiate_action]))
                self.entropy.append((negotiate_policy * torch.log_softmax(negotiate_logits, dim=-1)).sum())
                self.reward.append(0)
                self.value.append(negotiate_V)

            message[negotiate_action] = 1.
        else:
            message[-1] = 1.

        return message

    def make_decision(self, obs, messages):
        if not self.cfg.is_channel_open:
            messages = torch.zeros_like(messages)
            messages[self.message_space - 1::self.message_space] = 1.  # empty messages
        data = torch.cat((obs, messages))
        logits, V = self.model(data)

        policy = functional.softmax(logits, dim=-1)
        action = np.random.choice(policy.shape[0], p=policy.detach().numpy())

        if not self.eval:
            self.logs.append(torch.log(policy[action]))
            self.value.append(V)
            entropy = (policy * torch.log_softmax(logits, dim=-1)).sum()
            self.entropy.append(entropy)

        return action

    def rewarding(self, reward):
        if self.eval:
            self.reward_eval_metric.append(reward)
        else:
            self.reward.append(reward)
            self.reward_metric.append(reward)

    def train(self):
        G = 0
        policy_loss, value_loss = 0, 0
        for i in reversed(range(len(self.reward))):
            G = self.reward[i] + self.cfg.gamma * G
            advantage = G - self.value[i]

            value_loss = value_loss + 0.5 * advantage.pow(2)
            policy_loss = policy_loss - (advantage.detach() * self.logs[i] + self.cfg.entropy_coef * self.entropy[i])

        loss = 0.5 * value_loss + policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #for g in self.optimizer.param_groups:
        #    g['lr'] = g['lr'] * 1.

        self.loss_metric.append(loss.item())

        self.reward = []
        self.logs = []
        self.entropy = []
        self.value = []

    def get_label(self):
        return self.agent_label
