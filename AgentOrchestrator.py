import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sn
from sklearn.decomposition import PCA
from Agent import Agent


class Orchestrator:

    def __init__(self, obs_space: int, action_space: int, model_type, cfg):
        self.cfg = cfg
        self.messages = None
        self.ind = np.arange(cfg.players)
        self.eval = False
        self.negotiation_steps = np.max(cfg.negotiation_steps)

        negotiation_steps = cfg.negotiation_steps
        if isinstance(negotiation_steps, int):
            negotiation_steps = np.full(cfg.players, negotiation_steps)
        self.Agents = np.array([Agent(id,
                                      obs_space,
                                      action_space,
                                      model_type,
                                      negotiation_steps[id],
                                      cfg) for id in range(cfg.players)])

        # only in test episodes
        self.AM = np.zeros((cfg.players, cfg.players), dtype=np.int)
        self.DM = np.zeros((cfg.players, cfg.players), dtype=np.int)
        self.messages_distr = np.zeros((cfg.players, self.negotiation_steps, cfg.message_space + 1), np.int)

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
        for step in range(self.negotiation_steps):
            obs_negot = torch.cat((obs, torch.cat(messages)))
            messages = [agent.negotiate(obs_negot, step) for agent in self.Agents[self.ind]]
            if self.eval:
                for i in range(self.cfg.players):
                    self.messages_distr[i, step] += messages[i].detach().numpy().astype(np.int)
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
        ax[0].set_title('Offends', fontsize=24)
        ax[0].tick_params(labelsize=13)
        ax[1].set_title('Defends', fontsize=24)
        ax[1].tick_params(labelsize=13)
        xlabels = [agent.id + 1 for agent in self.Agents]
        ylabels = [agent.agent_label for agent in self.Agents]
        sn.heatmap(self.AM, annot=True, cmap='Reds', xticklabels=xlabels, yticklabels=ylabels, ax=ax[0], square=True,
                   cbar=False, fmt='g', annot_kws={"size": 15})
        sn.heatmap(self.DM, annot=True, cmap='Blues', xticklabels=xlabels, yticklabels=ylabels, ax=ax[1], square=True,
                   cbar=False, fmt='g', annot_kws={"size": 15})
        for (a, d) in zip(ax[0].yaxis.get_ticklabels(), ax[1].yaxis.get_ticklabels()):
            a.set_verticalalignment('center')
            a.set_rotation('horizontal')
            d.set_verticalalignment('center')
            d.set_rotation('horizontal')

        fig.tight_layout()
        if directory is not None:
            plt.savefig(f'{directory}/action_matrix.png')

        if self.cfg.use_negotiation:
            fig, ax = plt.subplots(1, self.negotiation_steps, figsize=(16, 9))
            labels = np.arange(1, self.cfg.message_space + 1).tolist() + ['empty']
            width = 0.2
            locations = np.arange(self.cfg.message_space + 1)
            player_loc = np.linspace(-width, width, self.cfg.players)
            for step in range(self.negotiation_steps):
                for i in range(self.cfg.players):
                    ax[step].bar(locations + player_loc[i], self.messages_distr[i, step], width,
                                 label=self.Agents[i].get_label())
                ax[step].set_ylabel('message count', fontsize=16)
                ax[step].set_xlabel('message category', fontsize=16)
                ax[step].set_title(f'Negotiation step {step + 1}', fontsize=18)
                ax[step].set_xticks(locations)
                ax[step].set_xticklabels(labels)
                ax[step].tick_params(labelsize=13)
                ax[step].legend()

            fig.tight_layout()
            if directory is not None:
                plt.savefig(f'{directory}/messages.png')

        if self.cfg.use_embeddings:
            fig, ax = plt.subplots(1, 1, figsize=(16, 9))
            ax.set_title('embeddings PCA', fontsize=24)
            embeddings = []
            for agent in self.Agents:
                embeddings.append(agent.embeddings.data)
            embeddings = torch.stack(embeddings).reshape(-1, self.cfg.embedding_space)
            pca = PCA(n_components=2)
            embeddings = pca.fit_transform(embeddings.detach().numpy())
            embeddings = embeddings.reshape(self.cfg.players, self.cfg.players, 2)
            for i, agent in enumerate(self.Agents):
                ax.scatter(embeddings[i, :, 0], embeddings[i, :, 1], label=agent.get_label(), s=150)
                for agent, emb in zip(self.Agents, embeddings[i]):
                    ax.annotate(f'{agent.id + 1}', emb + np.array([0, 0.075]), fontsize=14, ha='center')
            ax.legend()
            fig.tight_layout()
            if directory is not None:
                plt.savefig(f'{directory}/embeddings.png')

        if directory is None:
            plt.show()
