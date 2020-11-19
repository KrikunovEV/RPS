import matplotlib.pyplot as plt
import numpy as np
from Agent import Agent, MessageType


class Orchestrator:

    def __init__(self, obs_space: int, action_space: int, cfg):
        self.cfg = cfg
        self.teams_messages = []
        self.Agents = []
        self.solo_ids = []
        self.teams_ids = []

        counter = 0
        for a in range(cfg.n_agents):
            self.solo_ids.append(counter)
            self.Agents.append(Agent(id=counter, obs_space=obs_space, action_space=action_space, cfg=cfg))
            counter += 1

        for team, agents in enumerate(cfg.negot.teams):
            self.teams_ids.append([])
            space = obs_space + agents * cfg.negot.message_space
            for a in range(agents):
                self.teams_ids[-1].append(counter)
                self.Agents.append(Agent(id=counter, team=team, obs_space=space, action_space=action_space, cfg=cfg))
                counter += 1

    def set_eval(self, eval: bool):
        for agent in self.Agents:
            agent.set_eval(eval)

    def negotiation(self):
        self.teams_messages = []
        for team in self.teams_ids:
            agents = [self.Agents[a] for a in team]
            messages = [agent.get_last_message() for agent in agents]
            for step in range(self.cfg.negot.steps):
                messages = [agent.negotiate(messages[:i] + messages[i+1:]) for i, agent in enumerate(agents)]
            self.teams_messages.append([agent.get_last_message(detach=False) for agent in agents])
            for agent in agents:
                agent.negotiation_train()

    def decisions(self, obs):
        agents = [self.Agents[a] for a in self.solo_ids]
        choices = [agent.make_decision(obs, []) for agent in agents]

        for id, team in enumerate(self.teams_ids):
            agents = [self.Agents[a] for a in team]
            for i, agent in enumerate(agents):
                message = self.teams_messages[id].copy()
                for j in range(len(agents)):
                    if i != j:
                        message[j] = message[j].detach()
                choices = choices + [agent.make_decision(obs, message)]

        return choices

    def rewarding(self, rewards):
        for agent, reward in zip(self.Agents, rewards):
            agent.rewarding(reward)

    def train(self):
        for agent in self.Agents:
            agent.train()

    def plot_metrics(self, test_negotiations: int, directory: str = None, show: bool = False):

        self.__plot_metrics(directory)
        self.__plot_negot_metrics(test_negotiations, directory)

        if show:
            plt.show()

    def __plot_metrics(self, directory: str = None):
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
            ax[0].plot(agent.loss_metric + np.random.normal(0.0, 0.01, len(agent.loss_metric)), label=agent.get_label())
            ax[1].plot(np.cumsum(agent.reward_metric), label=agent.get_label())
            ax[2].plot(np.cumsum(agent.reward_eval_metric), label=agent.get_label())
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.tight_layout()
        if directory is not None:
            plt.savefig(f'{directory}/rl_metrics.png')

    def __plot_negot_metrics(self, test_negotiations: int, directory: str = None):
        if self.cfg.negot.message_type == MessageType.Categorical:
            for id, team in enumerate(self.teams_ids):
                agents = [self.Agents[a] for a in team]
                fig, ax = plt.subplots(1, 2, figsize=(16, 9))
                #fig.suptitle(f'Team {id}')
                ax[0].set_title(f'CE loss')
                ax[0].set_xlabel('# of negotiation')
                ax[0].set_ylabel('loss value')
                ax[1].set_title(f'Accuracy')
                ax[1].set_xlabel('# of negotiation')
                ax[1].set_ylabel('accuracy value')
                for agent in agents:
                    ax[0].plot(agent.negot_loss_metric, label=agent.get_label())
                    ax[1].plot(agent.negot_accuracy_metric, label=agent.get_label())
                ax[0].legend()
                ax[1].legend()
                plt.tight_layout()
                if directory is not None:
                    plt.savefig(f'{directory}/negotiation_train_team_{id}.png')

                fig, ax = plt.subplots(1, len(team), figsize=(16, 9))
                #fig.suptitle(f'Team {id}')
                for i, agent in enumerate(agents):
                    ax[i].set_title(f'Agent {team[i]} step-wise accuracy')
                    ax[i].set_xlabel('# of step')
                    ax[i].set_ylabel('accuracy value')
                    agent.negot_step_accuracy_metric = agent.negot_step_accuracy_metric / test_negotiations
                    for j in range(len(agents) - 1):
                        j_ind = j if j < i else j + 1
                        ax[i].plot(agent.negot_step_accuracy_metric[:, j], label=agents[j_ind].get_label())
                    ax[i].legend()
                plt.tight_layout()
                if directory is not None:
                    plt.savefig(f'{directory}/negotiation_test_team_{id}.png')

        elif self.cfg.negot.message_type == MessageType.Numerical:
            for id, team in enumerate(self.teams_ids):
                agents = [self.Agents[a] for a in team]
                fig, ax = plt.subplots(1, 1, figsize=(16, 9))
                #fig.suptitle(f'Team {id}')
                ax.set_title(f'MSE loss')
                ax.set_yscale('log')
                ax.set_xlabel('# of negotiation')
                ax.set_ylabel('loss value')
                for agent in agents:
                    ax.plot(agent.negot_loss_metric, label=agent.get_label())
                ax.legend()
                plt.tight_layout()
                if directory is not None:
                    plt.savefig(f'{directory}/negotiation_train_team_{id}.png')

                fig, ax = plt.subplots(1, len(team), figsize=(16, 9))
                #fig.suptitle(f'Team {id}')
                for i, agent in enumerate(agents):
                    ax[i].set_title(f'Agent {team[i]} step-wise distance')
                    ax[i].set_xlabel('# of step')
                    ax[i].set_ylabel('distance value')
                    for j in range(len(agents) - 1):
                        j_ind = j if j < i else j + 1
                        ax[i].plot(agent.negot_step_distance_metric[:, j], label=agents[j_ind].get_label())
                    ax[i].legend()
                plt.tight_layout()
                if directory is not None:
                    plt.savefig(f'{directory}/negotiation_test_team_{id}.png')
