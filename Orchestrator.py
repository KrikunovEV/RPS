import numpy as np
import torch
from Agent import Agent


class Orchestrator:

    def __init__(self, obs_space: int, action_space: int, cfg):
        self.cfg = cfg
        self.eval = False
        self.negotiation_steps = np.max(cfg.negotiation.steps)
        self.shuffle_id = np.arange(cfg.common.players)
        self.kv_all, self.q_all = None, None

        if not cfg.common.use_obs:
            obs_space = 0
        if cfg.negotiation.use:
            obs_space += cfg.negotiation.space * 4
        if cfg.embeddings.use:
            obs_space += cfg.embeddings.space

        self.Agents = [Agent(id, obs_space, action_space, cfg) for id in range(cfg.common.players)]

        # only in eval mode
        self.pair_coops = dict()
        for p1 in range(cfg.common.players - 1):
            for p2 in range(p1 + 1, cfg.common.players):
                self.pair_coops[f'{self.Agents[p1].agent_label}&{self.Agents[p2].agent_label}'] = 0

    def shuffle(self):
        np.random.shuffle(self.shuffle_id)

    def reset_memory(self):
        for agent in self.Agents:
            agent.reset_memory()

    def set_eval(self, eval: bool):
        self.eval = eval
        for agent in self.Agents:
            agent.set_eval(eval)

    def negotiation(self):
        kv_all, q_all = self.__build_kv_q()

        for step in range(self.negotiation_steps):
            new_messages = [agent.negotiate(kv_all[agent.id], q_all[agent.id], step) for agent in self.Agents]
            kv_all, q_all = self.__build_kv_q(new_messages)

        self.kv_all, self.q_all = kv_all, q_all

    def __build_kv_q(self, messages=None):
        kv_all, q_all = [], []
        for p1 in range(self.cfg.common.players):
            kv_p, q_p = [], []
            for p2 in range(self.cfg.common.players):
                if p1 != p2:
                    if messages is None:
                        kv_p.append(self.Agents[p1].start_messages[str(p2)])
                        q_p.append(self.Agents[p2].start_messages[str(p1)])
                    else:
                        kv_p.append(messages[p1][str(p2)])
                        q_p.append(messages[p2][str(p1)])
            kv_all.append(torch.stack(kv_p))
            q_all.append(torch.stack(q_p).detach())
        return kv_all, q_all

    def decisions(self, obs, epsilon):
        if self.cfg.common.use_obs:
            obs = torch.from_numpy(obs)
        else:
            obs = torch.empty((0,))

        if self.cfg.negotiation.use:
            self.q_all = torch.stack([self.q_all[p].reshape(-1) for p in range(len(self.q_all))])

            choices = []
            for agent in self.Agents:
                kv_all_detached = []
                for p in range(self.cfg.common.players):
                    kv_all_detached.append(self.kv_all[p].reshape(-1))
                    if p != agent.id:
                        kv_all_detached[-1] = kv_all_detached[-1].detach()
                kv_all_detached = torch.stack(kv_all_detached)
                agent_obs = torch.cat((obs, kv_all_detached, self.q_all), dim=1)

                choices.append(agent.make_decision(agent_obs, epsilon))

        if self.eval:
            for p1 in range(self.cfg.common.players - 1):
                for p2 in range(p1 + 1, self.cfg.common.players):
                    if choices[p1][0] == choices[p2][0] and choices[p1][1] == choices[p2][1]:
                        self.pair_coops[f'{self.Agents[p1].agent_label}&{self.Agents[p2].agent_label}'] += 1

        return choices

    def rewarding(self, rewards):
        for agent, reward in zip(self.Agents, rewards):
            agent.rewarding(reward)

    def train(self):
        for agent in self.Agents:
            agent.train()

    def get_metrics(self):
        return {'pair_coops': self.pair_coops,
                'agent_labels_list': [agent.agent_label for agent in self.Agents],
                'loss_list': [agent.loss_metric for agent in self.Agents],
                'reward_list': [agent.reward_metric for agent in self.Agents],
                'reward_eval_list': [agent.reward_eval_metric for agent in self.Agents],
                'attacks_list': [agent.attacks_metric for agent in self.Agents],
                'defends_list': [agent.defends_metric for agent in self.Agents]}
