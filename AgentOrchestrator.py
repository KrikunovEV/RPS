import matplotlib.pyplot as plt
import numpy as np
import torch
from Agent import Agent


class Orchestrator:

    def __init__(self, obs_space: int, action_space: int, cfg):
        self.cfg = cfg
        self.messages = []
        self.Agents = [Agent(id, obs_space, action_space, cfg) for id in range(cfg.players)]

    def set_eval(self, eval: bool):
        for agent in self.Agents:
            agent.set_eval(eval)

    def negotiation(self, obs):
        obs = torch.Tensor(obs)
        messages = [agent.get_last_message() for agent in self.Agents]

        for step in range(self.cfg.negot.steps):
            messages = [
                agent.negotiate(messages[:i] + messages[i+1:], obs, step) for i, agent in enumerate(self.Agents)
            ]

        # [agents, agents, message_space]
        self.messages = []
        for a in range(len(self.Agents)):
            messages = []
            for a2, agent in enumerate(self.Agents):
                messages.append(agent.get_last_message(detach=(False if a == a2 else True)))
            self.messages.append(messages)

    def decisions(self, obs):
        choices = [agent.make_decision(obs, message) for agent, message in zip(self.Agents, self.messages)]
        return choices

    def rewarding(self, rewards):
        for agent, reward in zip(self.Agents, rewards):
            agent.rewarding(reward)

    def train(self):
        for agent in self.Agents:
            agent.train()

    def plot_metrics(self, directory: str = None, show: bool = False):
        fig, ax = plt.subplots(1, 3, figsize=(16, 9))
        ax[0].set_title(f'REINFORCE loss')
        ax[0].set_xlabel('# of episode')
        ax[0].set_ylabel('loss value')
        ax[1].set_title(f'Cumulative reward')
        ax[1].set_xlabel('# of episode')
        ax[1].set_ylabel('reward value')
        ax[2].set_title(f'Cumulative reward on test')
        ax[2].set_xlabel('# of episode')
        ax[2].set_ylabel('reward value')
        for agent in self.Agents:
            ax[0].plot(agent.loss_metric, label=agent.get_label())
            ax[1].plot(np.cumsum(np.array(agent.reward_metric).reshape(-1, self.cfg.rounds).sum(axis=1)),
                       label=agent.get_label())
            ax[2].plot(np.cumsum(np.array(agent.reward_eval_metric).reshape(-1, self.cfg.rounds).sum(axis=1)),
                       label=agent.get_label())
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        fig.tight_layout()
        if directory is not None:
            plt.savefig(f'{directory}/rl_metrics.png')

        if show:
            plt.show()


a = [0.0104, 0.1562, 0.1562]
print(a / np.sum(a))