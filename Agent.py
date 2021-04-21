import torch
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
from Model import SiamMLPModel, AttentionLayer


class Agent:
    def __init__(self, id: int, obs_space: int, action_space: int, cfg):
        self.cfg = cfg
        self.id = id
        self.negotiable = True if id < cfg.negotiation.players else False
        self.agent_label = f'{id + 1}' + ('n' if self.negotiable else '')
        self.eval = False
        self.negotiation_steps = cfg.negotiation.steps[id]

        self.model = SiamMLPModel(obs_space, action_space, cfg)
        list_params = list(self.model.parameters())

        self.transformer = None
        self.start_messages = dict()
        if cfg.negotiation.use:
            self.transformer = []
            for step in range(self.negotiation_steps):
                self.transformer.append(AttentionLayer(cfg))
                list_params = list_params + list(self.transformer[-1].parameters())

            for p in range(cfg.common.players):
                if p != id:
                    #if self.negotiable:
                    #    self.start_messages[str(p)] = torch.nn.Parameter(torch.randn(cfg.negotiation.space,
                    #                                                                 requires_grad=True))
                    #    list_params = list_params + [self.start_messages[str(p)]]
                    #else:
                    self.start_messages[str(p)] = torch.zeros(cfg.negotiation.space)

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

    def set_eval(self, eval: bool):
        self.eval = eval

    def reset_memory(self):
        pass

    def negotiate(self, kv, q, step):
        new_messages = self.start_messages

        if self.negotiable and step < self.negotiation_steps:

            # concatenate one-hot vector to encode agent position
            emb = torch.eye(len(kv))
            kv = torch.cat((emb, kv), dim=1)
            q = torch.cat((emb, q), dim=1)

            messages = self.transformer[step](kv, q)
            ind = 0
            for p in range(self.cfg.common.players):
                if p != self.id:
                    new_messages[str(p)] = messages[ind]
                    ind += 1

        return new_messages

    def make_decision(self, obs, epsilon):
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
