import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sn
from Agent import Agent


class Orchestrator:

    def __init__(self, obs_space: int, action_space: int, attention: bool, cfg):
        self.cfg = cfg
        self.messages = []
        self.message_space = action_space + 1
        self.Agents = [Agent(id, obs_space, action_space, self.message_space, attention, cfg) for id in range(
            cfg.players)]
        self.eval = False
        self.AM = np.zeros((cfg.players, cfg.players), dtype=np.int)
        self.DM = np.zeros((cfg.players, cfg.players), dtype=np.int)

    #def shuffle(self):
    #    np.random.shuffle(self.Agents)
    #    for i in range(len(self.Agents)):
    #        self.Agents[i].mask_id = i

    def set_eval(self, eval: bool):
        self.eval = eval
        for agent in self.Agents:
            agent.set_eval(eval)

    def negotiation(self, obs):
        messages = []
        for a in range(self.cfg.players):
            tmp = torch.zeros(self.message_space)
            tmp[-1] = 1.
            messages.append(tmp)
        obs = torch.Tensor(obs)

        for step in range(self.cfg.negot_steps):
            obs_negot = torch.cat((obs, torch.cat(messages)))
            messages = [agent.negotiate(obs_negot) for agent in self.Agents]
        self.messages = messages

    def decisions(self, obs):
        obs = torch.Tensor(obs)
        messages = torch.cat(self.messages)
        choices = [agent.make_decision(obs, messages) for agent in self.Agents]
        if self.eval:
            for a, choice in enumerate(choices):
                self.AM[a, choice[0]] += 1
                self.DM[a, choice[1]] += 1
        return choices

    def reset_h(self):
        for agent in self.Agents:
            agent.reset_h()

    def rewarding(self, rewards):
        for agent, reward in zip(self.Agents, rewards):
            agent.rewarding(reward)

    def train(self):
        for agent in self.Agents:
            agent.train()

    def plot_metrics(self, directory: str = None):
        plt.close('all')
        plt.title('A2C loss')
        plt.xlabel('# of episode')
        plt.ylabel('loss value')
        for agent in self.Agents:
            plt.plot(agent.loss_metric, label=agent.get_label())
        plt.legend()
        plt.tight_layout()
        if directory is not None:
            plt.savefig(f'{directory}/loss.png')

        fig, ax = plt.subplots(1, 2, figsize=(16, 9))
        ax[0].set_title(f'Cumulative rewarding on train')
        ax[0].set_xlabel('# of episode')
        ax[0].set_ylabel('reward value')
        ax[1].set_title(f'Cumulative rewarding on test')
        ax[1].set_xlabel('# of episode')
        ax[1].set_ylabel('reward value')
        for agent in self.Agents:
            reward = np.array(agent.reward_metric)
            eval_reward = np.array(agent.reward_eval_metric)
            ax[0].plot(np.cumsum(reward), label=agent.get_label())
            ax[1].plot(np.cumsum(eval_reward), label=agent.get_label())
        ax[0].legend()
        ax[1].legend()
        fig.tight_layout()
        if directory is not None:
            plt.savefig(f'{directory}/rewarding.png')

        fig, ax = plt.subplots(1, 2, figsize=(16, 9))
        ax[0].set_title('Attacks')
        ax[1].set_title('Defends')
        labels = [agent.agent_label for agent in self.Agents]
        sn.heatmap(self.AM, annot=True, cmap='Reds', xticklabels=labels, yticklabels=labels, ax=ax[0], square=True,
                   cbar=False, fmt='g')
        sn.heatmap(self.DM, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax[1],  square=True,
                   cbar=False, fmt='g')
        for (a, d) in zip(ax[0].yaxis.get_ticklabels(), ax[1].yaxis.get_ticklabels()):
            a.set_verticalalignment('center')
            d.set_verticalalignment('center')
        if directory is not None:
            plt.savefig(f'{directory}/action_matrix.png')

        if directory is None:
            plt.show()
