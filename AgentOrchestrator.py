import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sn
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
        messages = [torch.zeros(self.cfg.negot.message_space) for a in range(self.cfg.players)]
        for step in range(self.cfg.negot.steps):
            messages = [agent.negotiate(messages, obs) for agent in self.Agents]
        self.messages = messages

    def decisions(self, obs):
        choices = [agent.make_decision(obs, self.messages) for agent in self.Agents]
        return choices

    def rewarding(self, rewards):
        for agent, reward in zip(self.Agents, rewards):
            agent.rewarding(reward)

    def train(self):
        for agent in self.Agents:
            agent.train()

    def plot_metrics(self, A_CM, D_CM, directory: str = None, show: bool = False):
        plt.title('REINFORCE loss')
        plt.xlabel('# of episode')
        plt.ylabel('loss value')
        for agent in self.Agents:
            plt.plot(agent.loss_metric, label=agent.get_label())
        plt.legend()
        plt.tight_layout()
        if directory is not None:
            plt.savefig(f'{directory}/loss.png')

        fig, ax = plt.subplots(2, 2, figsize=(16, 9))
        ax[0][0].set_title(f'Rewarding on train')
        ax[0][0].set_xlabel('# of episode')
        ax[0][0].set_ylabel('reward value')
        ax[0][1].set_title(f'Rewarding on test')
        ax[0][1].set_xlabel('# of episode')
        ax[0][1].set_ylabel('reward value')
        ax[1][0].set_title(f'Cumulative rewarding on train')
        ax[1][0].set_xlabel('# of episode')
        ax[1][0].set_ylabel('reward value')
        ax[1][1].set_title(f'Cumulative rewarding on test')
        ax[1][1].set_xlabel('# of episode')
        ax[1][1].set_ylabel('reward value')
        for agent in self.Agents:
            reward = np.array(agent.reward_metric).reshape(-1, self.cfg.rounds).sum(axis=1)
            eval_reward = np.array(agent.reward_eval_metric).reshape(-1, self.cfg.rounds).sum(axis=1)
            ax[0][0].plot(reward, label=agent.get_label())
            ax[0][1].plot(eval_reward, label=agent.get_label())
            ax[1][0].plot(np.cumsum(reward), label=agent.get_label())
            ax[1][1].plot(np.cumsum(eval_reward), label=agent.get_label())
        ax[0][0].legend()
        ax[0][1].legend()
        ax[1][0].legend()
        ax[1][1].legend()
        fig.tight_layout()
        if directory is not None:
            plt.savefig(f'{directory}/rewarding.png')

        fig, ax = plt.subplots(1, 2, figsize=(16, 9))
        ax[0].set_title('Attacks')
        ax[1].set_title('Defends')
        ticklabels = ['agent ' + agent.agent_label for agent in self.Agents]
        sn.heatmap(A_CM, annot=True, cmap='Reds',
                   xticklabels=ticklabels, yticklabels=ticklabels, ax=ax[0], square=True)
        sn.heatmap(D_CM, annot=True, cmap='Blues',
                   xticklabels=ticklabels, yticklabels=ticklabels, ax=ax[1], square=True)
        if directory is not None:
            plt.savefig(f'{directory}/CM_metrics.png')

        if show:
            plt.show()
