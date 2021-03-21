import torch
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
from Model import DecisionModel, AttDecisionModel, NegotiationModel


class Agent:
    def __init__(self, id: int, obs_space: int, action_space: int, message_space: int, attention: bool, cfg):
        self.cfg = cfg
        self.id = id
        self.negotiable = True if id >= cfg.n_agents else False
        self.agent_label = f'{id}' + (' negotiable' if self.negotiable else '')
        self.eval = False

        self.message_space = message_space
        self.attention = attention
        self.h = torch.zeros((1, 2))
        new_obs_space = obs_space + cfg.players * message_space
        if attention:
            self.model = AttDecisionModel(message_space, obs_space, cfg.players)
        else:
            self.model = DecisionModel(new_obs_space, action_space)
        self.negot_model = NegotiationModel(new_obs_space, action_space)
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
        if not self.negotiable and not self.cfg.is_channel_open:
            messages = torch.zeros_like(messages)
            messages[self.message_space - 1::self.message_space] = 1.  # empty messages

        if self.attention:
            a_logits, d_logits, V = self.model(obs, messages, self.h)
        else:
            a_logits, d_logits, V = self.model(torch.cat((obs, messages)))

        a_policy = functional.softmax(a_logits, dim=-1)
        d_policy = functional.softmax(d_logits, dim=-1)
        strategy = np.random.choice(['random', 'policy'], p=[self.cfg.epsilon, 1 - self.cfg.epsilon])
        if not self.eval and strategy == 'random':
            a_action = np.random.randint(a_policy.shape[0])
            d_action = np.random.randint(d_policy.shape[0])
        else:
            a_action = np.random.choice(a_policy.shape[0], p=a_policy.detach().numpy())
            d_action = np.random.choice(d_policy.shape[0], p=d_policy.detach().numpy())

        if not self.eval:
            if a_policy[a_action] < 0.0000001 and d_policy[d_action] < 0.0000001:
                self.logs.append(torch.log(a_policy[a_action] * d_policy[d_action] + 0.0000001))
            else:
                self.logs.append(torch.log(a_policy[a_action] * d_policy[d_action]))
            self.value.append(V)
            a_entropy = (a_policy * torch.log_softmax(a_logits, dim=-1)).sum()
            d_entropy = (d_policy * torch.log_softmax(d_logits, dim=-1)).sum()
            self.entropy.append(a_entropy + d_entropy)

        return [a_action, d_action]

    def reset_h(self):
        self.h = torch.zeros((1, 2))

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
