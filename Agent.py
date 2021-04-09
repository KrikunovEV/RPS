import torch
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
from Model import SiamMLPModel, NegotiationModel


class Agent:
    def __init__(self, id: int, obs_space: int, action_space: int, model_type, negotiation_steps: int, cfg):
        self.cfg = cfg
        self.id = id
        self.negotiable = True if id < cfg.negotiation.players else False
        self.agent_label = f'{id + 1}' + (' negotiable' if self.negotiable else '')
        self.eval = False
        self.model_type = model_type
        self.negotiation_steps = negotiation_steps
        self.negotiate_action = -1

        self.model = SiamMLPModel(obs_space, action_space, cfg)
        list_params = list(self.model.parameters())

        self.negot_model = None
        if cfg.negotiation.use:
            self.negot_model = []
            for step in range(negotiation_steps):
                self.negot_model.append(NegotiationModel(obs_space, cfg))
                list_params = list_params + list(self.negot_model[-1].parameters())

        self.embeddings = None
        if cfg.embeddings.use:
            self.embeddings = torch.nn.Parameter(torch.ones((cfg.common.players, cfg.embeddings.space)),
                                                 requires_grad=True)
            list_params = list_params + [self.embeddings]

        self.optimizer = optim.Adam(list_params, lr=cfg.train.lr)

        self.logs = []
        self.value = []
        self.entropy = []
        self.reward = []

        # only in train
        self.loss_metric = []
        self.reward_metric = []
        # only in eval mode
        self.reward_eval_metric = []
        self.attacks_metric = np.zeros(cfg.common.players, dtype=np.int)
        self.defends_metric = np.zeros(cfg.common.players, dtype=np.int)
        self.messages_metric = np.zeros((negotiation_steps, cfg.negotiation.space + 1), dtype=np.int)

    def set_eval(self, eval: bool):
        self.eval = eval

    def reset_memory(self):
        raise Exception('There is no any model yet for resetting its hidden memory')

    def negotiate(self, obs, step, epsilon):
        message = torch.zeros(self.cfg.negotiation.space + 1)

        if self.negotiable and step < self.negotiation_steps:
            if self.cfg.embeddings.use:
                obs = torch.cat((obs, self.embeddings), dim=1)

            negotiate_logits, negotiate_V = self.negot_model[step](obs)
            negotiate_policy = functional.softmax(negotiate_logits, dim=-1)
            strategy = np.random.choice(['random', 'policy'], p=[epsilon, 1. - epsilon])
            if not self.eval and strategy == 'random':
                self.negotiate_action = np.random.randint(negotiate_policy.shape[0])
            else:
                self.negotiate_action = np.random.choice(negotiate_policy.shape[0],
                                                         p=negotiate_policy.detach().numpy())

            if not self.eval:
                if negotiate_policy[self.negotiate_action] < 0.00000001:
                    self.logs.append(torch.log(negotiate_policy[self.negotiate_action] + 0.00000001))
                else:
                    self.logs.append(torch.log(negotiate_policy[self.negotiate_action]))
                self.entropy.append((negotiate_policy * torch.log_softmax(negotiate_logits, dim=-1)).sum())
                self.reward.append(0)
                self.value.append(negotiate_V)

        message[self.negotiate_action] = 1.

        if self.eval and step < self.negotiation_steps:
            self.messages_metric[step, self.negotiate_action] += 1

        return message

    def make_decision(self, obs, epsilon):
        if self.cfg.negotiation.use and not self.negotiable and not self.cfg.negotiation.is_channel_open:
            obs[:, -(self.cfg.negotiation.space + 1):-1] = 0.
            obs[:, -1] = 1.

        if self.cfg.embeddings.use:
            obs = torch.cat((obs, self.embeddings), dim=1)

        a_logits, d_logits, V = self.model(obs)

        a_policy = functional.softmax(a_logits, dim=-1)
        d_policy = functional.softmax(d_logits, dim=-1)
        strategy = np.random.choice(['random', 'policy'], p=[epsilon, 1. - epsilon])
        if not self.eval and strategy == 'random':
            a_action = np.random.randint(a_policy.shape[0])
            d_action = np.random.randint(d_policy.shape[0])
        else:
            a_action = np.random.choice(a_policy.shape[0], p=a_policy.detach().numpy())
            d_action = np.random.choice(d_policy.shape[0], p=d_policy.detach().numpy())

        if not self.eval:
            if a_policy[a_action] * d_policy[d_action] < 0.00000001:
                self.logs.append(torch.log(a_policy[a_action] * d_policy[d_action] + 0.00000001))
            else:
                self.logs.append(torch.log(a_policy[a_action] * d_policy[d_action]))
            self.value.append(V)
            a_entropy = (a_policy * torch.log_softmax(a_logits, dim=-1)).sum()
            d_entropy = (d_policy * torch.log_softmax(d_logits, dim=-1)).sum()
            self.entropy.append(a_entropy + d_entropy)
        else:
            self.attacks_metric[a_action] += 1
            self.defends_metric[d_action] += 1

        return [a_action, d_action]

    def rewarding(self, reward):
        if self.eval:
            self.reward_eval_metric.append(reward)
        else:
            self.reward.append(reward)
            self.reward_metric.append(reward)

    def train(self):
        if self.cfg.train.do_backward:
            G = 0
            policy_loss, value_loss = 0, 0
            for i in reversed(range(len(self.reward))):
                G = self.reward[i] + self.cfg.train.gamma * G
                advantage = G - self.value[i]

                value_loss = value_loss + 0.5 * advantage.pow(2)
                policy_loss = policy_loss - (advantage.detach() * self.logs[i] +
                                             self.cfg.train.entropy_penalize * self.entropy[i])

            loss = self.cfg.train.value_loss_penalize * value_loss + policy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_metric.append(loss.item())
        else:
            self.loss_metric.append(0.)

        self.reward = []
        self.logs = []
        self.entropy = []
        self.value = []

    def get_label(self):
        return self.agent_label
