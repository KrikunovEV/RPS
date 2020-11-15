import matplotlib.pyplot as plt
import numpy as np
from Agent import NegotiationAgent, MessageType


class AgentOrchestrator:

    def __init__(self, lr: float, message_space: int, steps: int, n_agents: int, message_type: MessageType):
        self.n_agents = n_agents
        self.message_type = message_type
        self.Agents = [NegotiationAgent(f'{a} {message_type.name}', message_type, lr, message_space, n_agents, steps)
                       for a in range(n_agents)]

        #np.random.shuffle(self.Agents)

    def get_last_messages(self):
        return [agent.get_last_message() for agent in self.Agents]

    def generate_messages(self, messages: list):
        return [agent.generate_message(messages[:i] + messages[i+1:]) for i, agent in enumerate(self.Agents)]

    def train(self):
        for agent in self.Agents:
            agent.train()

    def eval(self):
        for agent in self.Agents:
            agent.eval()

    def plot_metrics(self, test_negotiations: int, directory: str = None, show: bool = False):

        if self.message_type == MessageType.Categorical:
            fig, ax = plt.subplots(1, 2, figsize=(16, 9))
            ax[0].set_title(f'CE loss')
            ax[0].set_xlabel('# of negotiation')
            ax[0].set_ylabel('loss value')
            ax[1].set_title(f'Accuracy')
            ax[1].set_xlabel('# of negotiation')
            ax[1].set_ylabel('accuracy value')
            for agent in self.Agents:
                ax[0].plot(agent.loss_metric, label=agent.get_label())
                ax[1].plot(agent.accuracy_metric, label=agent.get_label())
            ax[0].legend()
            ax[1].legend()
            plt.tight_layout()
            if directory is not None:
                plt.savefig(f'{directory}/categorical_train.png')

            fig, ax = plt.subplots(1, self.n_agents, figsize=(16, 9))
            for i, agent in enumerate(self.Agents):
                ax[i].set_title(f'Agent {i} step-wise accuracy')
                ax[i].set_xlabel('# of step')
                ax[i].set_ylabel('accuracy value')
                agent.step_accuracy_metric = agent.step_accuracy_metric / test_negotiations
                for j in range(self.n_agents - 1):
                    j_ind = j if j < i else j + 1
                    ax[i].plot(agent.step_accuracy_metric[:, j], label=self.Agents[j_ind].get_label())
                ax[i].legend()
            plt.tight_layout()
            if directory is not None:
                plt.savefig(f'{directory}/categorical_test.png')

        elif self.message_type == MessageType.Numerical:
            fig, ax = plt.subplots(1, 1, figsize=(16, 9))
            ax.set_title(f'MSE loss')
            ax.set_yscale('log')
            ax.set_xlabel('# of negotiation')
            ax.set_ylabel('loss value')
            for agent in self.Agents:
                ax.plot(agent.loss_metric, label=agent.get_label())
            ax.legend()
            plt.tight_layout()
            if directory is not None:
                plt.savefig(f'{directory}/numerical_train.png')

            fig, ax = plt.subplots(1, self.n_agents, figsize=(16, 9))
            for i, agent in enumerate(self.Agents):
                ax[i].set_title(f'Agent {i} step-wise distance')
                ax[i].set_xlabel('# of step')
                ax[i].set_ylabel('distance value')
                for j in range(self.n_agents - 1):
                    j_ind = j if j < i else j + 1
                    ax[i].plot(agent.step_distance_metric[:, j], label=self.Agents[j_ind].get_label())
                ax[i].legend()
            plt.tight_layout()
            if directory is not None:
                plt.savefig(f'{directory}/numerical_test.png')

        if show:
            plt.show()
