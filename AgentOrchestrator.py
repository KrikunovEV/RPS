import numpy as np
import torch
from Agent import Agent


class Orchestrator:

    def __init__(self, obs_space: int, action_space: int, model_type, cfg):
        self.cfg = cfg
        self.messages = None
        self.ind = np.arange(cfg.common.players)
        self.eval = False
        self.negotiation_steps = np.max(cfg.negotiation.steps)
        self.Agents = np.array([Agent(id,
                                      obs_space,
                                      action_space,
                                      model_type,
                                      cfg.negotiation.steps[id],
                                      cfg) for id in range(cfg.common.players)])

        # only in eval mode
        self.pair_coops = dict()
        for p1 in range(cfg.common.players - 1):
            for p2 in range(p1 + 1, cfg.common.players):
                self.pair_coops[f'{p1 + 1}&{p2 + 1}'] = 0

    def shuffle(self, obs):
        np.random.shuffle(self.ind)
        return obs[self.ind]

    def reset_memory(self):
        for agent in self.Agents:
            agent.reset_memory()

    def set_eval(self, eval: bool):
        self.eval = eval
        for agent in self.Agents:
            agent.set_eval(eval)

    def negotiation(self, obs, epsilon):
        messages = []
        for a in range(self.cfg.common.players):
            tmp = torch.zeros(self.cfg.negotiation.space + 1)
            tmp[-1] = 1.
            messages.append(tmp)

        obs = torch.from_numpy(obs)
        for step in range(self.negotiation_steps):
            messages = torch.stack(messages)
            messages = [agent.negotiate(obs, messages, step, epsilon) for agent in self.Agents[self.ind]]
        self.messages = messages

    def decisions(self, obs, epsilon):
        obs = torch.from_numpy(obs)
        if self.messages is not None:
            messages = torch.stack(self.messages)
        else:
            messages = self.messages
        choices = np.array([agent.make_decision(obs, messages, epsilon) for agent in self.Agents[self.ind]])
        choices[self.ind] = choices[np.arange(self.cfg.common.players)]

        if self.eval:
            for p1 in range(self.cfg.common.players - 1):
                for p2 in range(p1 + 1, self.cfg.common.players):
                    if choices[p1, 0] == choices[p2, 0] and choices[p1, 1] == choices[p2, 1]:
                        self.pair_coops[f'{p1 + 1}&{p2 + 1}'] += 1

        return choices.tolist()

    def rewarding(self, rewards):
        for agent, reward in zip(self.Agents, rewards):
            agent.rewarding(reward)

    def train(self):
        for agent in self.Agents:
            agent.train()

    def get_metrics(self):
        return {'pair_coops': self.pair_coops,
                'agent_labels_list': [agent.get_label() for agent in self.Agents],
                'loss_list': [agent.loss_metric for agent in self.Agents],
                'reward_list': [agent.reward_metric for agent in self.Agents],
                'reward_eval_list': [agent.reward_eval_metric for agent in self.Agents],
                'attacks_list': [agent.attacks_metric for agent in self.Agents],
                'defends_list': [agent.defends_metric for agent in self.Agents],
                'messages_list': [agent.messages_metric for agent in self.Agents],
                'embeddings_list': [agent.embeddings for agent in self.Agents]}
