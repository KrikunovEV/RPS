import matplotlib.pyplot as plt
import numpy as np
from Agent import NegotiationAgent, MessageType


class AgentOrchestrator:

    def __init__(self, lr: float, message_space: int, steps: int, categorical_agents: int, numerical_agents: int):
        agents = categorical_agents + numerical_agents
        self.categorical_agents = categorical_agents
        self.numerical_agents = numerical_agents
        self.Agents = []

        for a in range(categorical_agents):
            self.Agents.append(NegotiationAgent(f'{a} cat', MessageType.Categorical, lr, message_space, agents, steps))

        for a in range(categorical_agents, categorical_agents + numerical_agents):
            self.Agents.append(NegotiationAgent(f'{a} num', MessageType.Numerical, lr, message_space, agents, steps))

        np.random.shuffle(self.Agents)

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

    def plot_metrics(self, test_negotiations: int, directory: str = None):

        if self.categorical_agents > 0:
            fig, ax = plt.subplots(1, 3, figsize=(16, 9))

            ax[0].set_title(f'CE loss')
            ax[0].set_xlabel('# of negotiation')
            ax[0].set_ylabel('loss value')

            ax[1].set_title(f'Accuracy')
            ax[1].set_xlabel('# of negotiation')
            ax[1].set_ylabel('accuracy value')

            ax[2].set_title(f'Step-wise accuracy')
            ax[2].set_xlabel('# of step')
            ax[2].set_ylabel('accuracy value')

            for agent in self.Agents:
                if agent.message_type == MessageType.Categorical:
                    ax[0].plot(agent.loss_metric, label=agent.get_label())
                    ax[1].plot(agent.accuracy_metric, label=agent.get_label())
                    ax[2].plot(agent.level_accuracy_metric / test_negotiations, label=agent.get_label())

            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            plt.tight_layout()

            if directory is not None:
                plt.savefig(f'{directory}/categorical.png')

        if self.numerical_agents > 0:
            fig, ax = plt.subplots(1, 2, figsize=(16, 9))

            ax[0].set_title(f'MSE loss')
            ax[0].set_yscale('log')
            ax[0].set_xlabel('# of negotiation')
            ax[0].set_ylabel('loss value')

            ax[1].set_title(f'Step-wise distance')
            ax[1].set_xlabel('# of step')
            ax[1].set_ylabel('value')

            for agent in self.Agents:
                if agent.message_type == MessageType.Numerical:
                    ax[0].plot(agent.loss_metric, label=agent.get_label())
                    ax[1].plot(agent.distance_metric.detach().numpy(), label=agent.get_label())

            ax[0].legend()
            ax[1].legend()
            plt.tight_layout()

            if directory is not None:
                plt.savefig(f'{directory}/numerical.png')

        plt.show()
