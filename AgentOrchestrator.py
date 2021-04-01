import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sn
from Agent import Agent


class Orchestrator:

    def __init__(self, obs_space: int, action_space: int, model_type, cfg):
        self.cfg = cfg
        self.messages = None
        self.Agents = np.array([Agent(id,
                                      obs_space,
                                      action_space,
                                      model_type,
                                      cfg) for id in range(cfg.players)])
        self.ind = np.arange(cfg.players)
        self.eval = False
        self.AM = np.zeros((cfg.players, cfg.players), dtype=np.int)
        self.DM = np.zeros((cfg.players, cfg.players), dtype=np.int)

    def shuffle(self, obs):
        np.random.shuffle(self.ind)
        return obs[self.ind]

    def reset_h(self):
        for agent in self.Agents:
            agent.reset_h()

    def set_eval(self, eval: bool):
        self.eval = eval
        for agent in self.Agents:
            agent.set_eval(eval)

    def negotiation(self, obs):
        messages = []
        for a in range(self.cfg.players):
            tmp = torch.zeros(self.cfg.message_space + 1)
            tmp[-1] = 1.
            messages.append(tmp)

        obs = torch.from_numpy(obs).reshape(-1)
        for step in range(self.cfg.negotiation_steps):
            obs_negot = torch.cat((obs, torch.cat(messages)))
            messages = [agent.negotiate(obs_negot, step) for agent in self.Agents[self.ind]]
        self.messages = messages

    def decisions(self, obs, epsilon):
        obs = torch.from_numpy(obs)
        if self.messages is not None:
            messages = torch.stack(self.messages)
        else:
            messages = self.messages
        choices = np.array([agent.make_decision(obs, messages, epsilon) for agent in self.Agents[self.ind]])
        choices[self.ind] = choices[np.arange(self.cfg.players)]
        if self.eval:
            for a, choice in enumerate(choices):
                self.AM[a, choice[0]] += 1
                self.DM[a, choice[1]] += 1
        return choices.tolist()

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
