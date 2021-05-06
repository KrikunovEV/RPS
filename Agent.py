import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
from Model import SiamMLP, NegotiationLayer


class Agent:
    def __init__(self, id: int, obs_space: int, action_space: int, cfg):
        self.cfg = cfg
        self.id = id
        self.negotiable = True if id < cfg.negotiation.players else False
        self.agent_label = f'{id + 1}' + ('n' if self.negotiable else '')
        self.eval = False
        self.negotiation_steps = cfg.negotiation.steps[id]
        self.train_step = 1

        self.model = SiamMLP(obs_space, action_space, cfg)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.train.lr)

        self.transformer = None
        self.negotiation_optimizer = None
        self.start_kv = []
        self.kv = []
        if cfg.negotiation.use and self.negotiable:
            self.start_kv = nn.Parameter(torch.zeros((cfg.common.players - 1, cfg.negotiation.space)))
            list_params = [self.start_kv]
            self.kv = self.start_kv
            emb = torch.eye(len(self.kv))

            self.transformer = []
            for step in range(self.negotiation_steps):
                self.transformer.append(NegotiationLayer(cfg, emb))
                list_params = list_params + list(self.transformer[-1].parameters())
            self.negotiation_optimizer = optim.Adam(list_params, lr=0., betas=cfg.train.betas)
        else:
            self.start_kv = torch.normal(0, 0.25, (cfg.common.players - 1, cfg.negotiation.space))
        self.kv = self.start_kv

        self.logs = []
        self.value = []
        self.entropy = []
        self.reward = []

        # only in train
        self.loss_metric = []
        self.reward_metric = []
        self.lrs = []
        # only in eval mode
        self.reward_eval_metric = []
        self.attacks_metric = np.zeros(cfg.common.players, dtype=np.int)
        self.defends_metric = np.zeros(cfg.common.players, dtype=np.int)

    def set_eval(self, eval: bool):
        self.eval = eval

    def reset_memory(self):
        return

    def negotiate(self, q, step):
        if self.negotiable and step < self.negotiation_steps:

            if step == 0:
                self.kv = self.start_kv

            self.kv = self.transformer[step](self.kv, q)

    def make_decision(self, obs, epsilon):
        #if self.cfg.negotiation.use:
        #    obs = torch.cat((obs, self.kv), dim=1)

        a_logits, d_logits, V = self.model(self.kv)

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
            self.attacks_metric[a_action if a_action < self.id else (a_action + 1)] += 1
            self.defends_metric[d_action if d_action < self.id else (d_action + 1)] += 1

        return [a_action if a_action < self.id else (a_action + 1), d_action if d_action < self.id else (d_action + 1)]

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
            if self.negotiable:
                self.negotiation_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.negotiable:
                self.negotiation_optimizer.step()

            self.loss_metric.append(loss.item())
        else:
            self.loss_metric.append(0.)

        self.reward = []
        self.logs = []
        self.entropy = []
        self.value = []

        if self.negotiable:
            lrate = self.cfg.train.hidden_size ** (-0.5)
            lrate *= min(self.train_step ** (-0.5), self.train_step * self.cfg.train.warmup_episodes ** (-1.5))
            self.train_step += 1
            self.lrs.append(lrate)
            for param_group in self.negotiation_optimizer.param_groups:
                param_group['lr'] = lrate
