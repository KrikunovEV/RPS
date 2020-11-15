import torch
from Model import PredictionModel, GenerationModel
from enum import Enum


class MessageType(Enum):
    Categorical = 1
    Numerical = 2


def make_one_hot(message):
    message_ = torch.zeros(len(message))
    message_[torch.argmax(message)] = 1.
    return message_


class NegotiationAgent:

    def __init__(self, label: str, message_type: MessageType, lr: float, message_space: int, n_agents: int, steps: int):
        self.agent_label = label
        self.message_type = message_type
        self.message_space = message_space
        self.n_agents = n_agents

        self.generator = GenerationModel(obs_space=message_space * n_agents, action_space=message_space)
        self.predictor = PredictionModel(obs_space=message_space * n_agents, action_space=message_space * (n_agents -
                                                                                                           1))
        self.optimizer = torch.optim.Adam(list(self.generator.parameters()) + list(self.predictor.parameters()), lr=lr)
        self.loss_fn = torch.nn.MSELoss() if message_type == MessageType.Numerical else torch.nn.CrossEntropyLoss()

        self.__init_messages()

        self.loss_metric = []
        self.accuracy_metric = []
        self.step_accuracy_metric = torch.zeros((steps - 1, n_agents - 1))
        self.step_distance_metric = torch.zeros((steps - 1, n_agents - 1))

    def get_last_message(self):
        return self.generated_messages[-1].detach()  # do detach to make sure that gradient won't be corrupted

    def generate_message(self, messages: list):

        # make categorical values
        if self.message_type == MessageType.Categorical:
            messages = [make_one_hot(message) for message in messages]

        # flat messages
        messages = torch.cat(messages)

        # store B_i
        self.received_messages.append(messages)

        # form a pair
        message = torch.cat((self.generated_messages[-1], self.received_messages[-1]))  # (A_i, B_i)

        # generate new message A_i+1
        self.generated_messages.append(self.generator(message))

        return self.get_last_message()

    def train(self):
        """
        generated_messages : received_messages
        A_0 : B_0
        A_1 : B_1
        ...
        A_n : B_n
        A_n+1 : absent

        We need to form pairs:
        x predicted -> target
        A_0 B_1' -> B_1
        A_1 B_2' -> B_2
        ...
        A_n-1 B_n' -> B_n

        How you can see B_0, A_n+1 and A_n can be not taken into account.
        """

        # convert to torch and remove B_0, A_n+1 and A_n
        self.generated_messages = torch.stack(self.generated_messages[:-2])
        self.received_messages = torch.stack(self.received_messages[1:])

        # form samples
        x = torch.cat((self.generated_messages, self.received_messages), dim=1)
        predicted = self.predictor(x)
        target = self.received_messages  # already converted to categorical if there is a need

        # reshape agent-wise
        target = target.reshape(-1, self.message_space)
        predicted = predicted.reshape(-1, self.message_space)

        # find classes
        if self.message_type == MessageType.Categorical:
            target = target.argmax(dim=1)

        # compute loss
        loss = self.loss_fn(predicted, target)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # clear messages
        self.__init_messages()

        # compute metrics
        self.loss_metric.append(loss.item())
        if self.message_type == MessageType.Categorical:
            accuracy = torch.mean(predicted.argmax(dim=1) == target, dtype=torch.float)
            self.accuracy_metric.append(accuracy.item())

    def eval(self):
        self.generated_messages = torch.stack(self.generated_messages[:-2])
        self.received_messages = torch.stack(self.received_messages[1:])

        x = torch.cat((self.generated_messages, self.received_messages), dim=1)
        predicted = self.predictor(x)
        target = self.received_messages  # already converted to categorical

        # step, agent, message
        target = target.reshape(-1, self.n_agents - 1, self.message_space)
        predicted = predicted.reshape(-1, self.n_agents - 1, self.message_space).detach()

        if self.message_type == MessageType.Categorical:
            self.step_accuracy_metric += predicted.argmax(dim=2) == target.argmax(dim=2)
        elif self.message_type == MessageType.Numerical:
            self.step_distance_metric += torch.sqrt(torch.sum((predicted - target) ** 2, dim=2))

        self.__init_messages()

    def get_label(self):
        return self.agent_label

    def __init_messages(self):
        self.generated_messages = [torch.zeros(self.message_space)]
        self.received_messages = []
