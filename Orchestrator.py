import numpy as np
import torch
from Agent import Agent


class Orchestrator:

    def __init__(self, obs_space: int, action_space: int, cfg):
        self.cfg = cfg
        self.eval = False
        self.negotiation_steps = np.max(cfg.negotiation.steps)
        self.shuffle_id = np.arange(cfg.common.players)
        self.q_all = None

        if not cfg.common.use_obs:
            obs_space = 0
        if cfg.negotiation.use:
            obs_space += cfg.negotiation.space  # * 2
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
        q_all = self.__build_q()
        for step in range(self.negotiation_steps):
            for agent in self.Agents:
                agent.negotiate(q_all[agent.id], step)
            q_all = self.__build_q(start=False)
        self.q_all = q_all

    def __build_q(self, start: bool = True):
        q_all = []
        ind = [0 for _ in range(self.cfg.common.players)]
        for p1 in range(self.cfg.common.players):
            q_p = []
            for p2 in range(self.cfg.common.players):
                if p1 == p2:
                    continue
                if start:
                    q_p.append(self.Agents[p2].start_kv[ind[p2]].detach())
                else:
                    q_p.append(self.Agents[p2].kv[ind[p2]].detach())
                ind[p2] += 1
            q_all.append(torch.stack(q_p))
        return q_all

    def decisions(self, obs, epsilon):
        if self.cfg.common.use_obs:
            obs = torch.from_numpy(obs)
        else:
            obs = torch.empty((0,))

        #if self.cfg.negotiation.use:
            #obs = torch.cat((obs, self.q_all), dim=1)

        choices = [agent.make_decision(self.q_all[agent.id], epsilon) for agent in self.Agents]

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
                'defends_list': [agent.defends_metric for agent in self.Agents],
                'lr': self.Agents[0].lrs}
