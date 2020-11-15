import matplotlib.pyplot as plt
from Agent import NegotiationAgent, MessageType


class AgentOrchestrator:

    def __init__(self, lr: float, message_space: int, n_agents: int, steps: int):
        self.Agents = [
            NegotiationAgent('A', MessageType.Categorical, lr, message_space, n_agents, steps),
            NegotiationAgent('B', MessageType.Categorical, lr, message_space, n_agents, steps),
            NegotiationAgent('С', MessageType.Categorical, lr, message_space, n_agents, steps)
        ]

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

    def plot_metrics(self, test_negotiations: int):
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))

        ax[1][0].set_title(f'MSE loss')
        ax[1][0].set_yscale('log')
        ax[1][0].set_xlabel('# of negotiation')
        ax[1][0].set_ylabel('loss value')

        ax[1][1].axis("off")

        ax[1][2].set_title(f'Level-wise distance')
        ax[1][2].set_xlabel('# of step')
        ax[1][2].set_ylabel('value')

        ax[0][0].set_title(f'CE loss')
        # ax[0][0].set_yscale('log')
        ax[0][0].set_xlabel('# of negotiation')
        ax[0][0].set_ylabel('loss value')

        ax[0][1].set_title(f'Accuracy')
        ax[0][1].set_xlabel('# of negotiation')
        ax[0][1].set_ylabel('accuracy value')

        ax[0][2].set_title(f'Level-wise accuracy')
        ax[0][2].set_xlabel('# of step')
        ax[0][2].set_ylabel('accuracy value')

        for agent in self.Agents:
            if agent.message_type == MessageType.Categorical:
                ax[0][0].plot(agent.loss_metric, label=agent.agent_label)
                ax[0][1].plot(agent.accuracy_metric, label=agent.agent_label)
                ax[0][2].plot(agent.level_accuracy_metric / test_negotiations, label=agent.agent_label)
            else:
                ax[1][0].plot(agent.loss_metric, label=agent.agent_label)
                ax[1][2].plot(agent.distance_metric.detach().numpy(), label=agent.agent_label)

        ax[0][0].legend()
        ax[0][1].legend()
        ax[0][2].legend()
        ax[1][0].legend()
        ax[1][2].legend()

        plt.tight_layout()
        plt.show()
