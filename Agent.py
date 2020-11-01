from Model import PredictionModel, GenerationModel
import torch
from enum import Enum


class Agent:
    class MessageType(Enum):
        Categorical = 1
        Numerical = 2

    def __init__(self, label: str, message_type: MessageType, lr: float, obs_space: int, action_space: int):
        self.agent_label = label + (' categorical' if message_type == self.MessageType.Categorical else ' numerical')
        self.message_type = message_type

        self.generator = GenerationModel(obs_space=obs_space, action_space=action_space)
        self.predictor = PredictionModel(obs_space=obs_space, action_space=action_space)
        self.optimizer = torch.optim.SGD(list(self.generator.parameters()) + list(self.predictor.parameters()), lr=lr)
        self.loss_fn = torch.nn.MSELoss() if message_type == self.MessageType.Numerical else torch.nn.CrossEntropyLoss()

        self.generated_messages = [torch.zeros(20)]
        self.received_messages = []

        self.loss_metric = []
        self.accuracy_metric = []

    def get_last_message(self):
        return self.generated_messages[-1].detach()  # do detach to make sure that gradient won't be corrupted

    def generate_message(self, message):
        if self.message_type == self.MessageType.Categorical:
            message = torch.Tensor([1 if i == torch.argmax(message) else 0 for i in range(len(message))])
        self.generated_messages.append(self.generator(torch.Tensor(message)))
        self.received_messages.append(message)
        return self.generated_messages[-1].detach()

    def train(self):
        self.generated_messages = self.generated_messages[:-2]  # nothing to predict for the last messages
        self.received_messages = self.received_messages[1:]  # first message is zero tensor

        x = torch.stack(self.generated_messages)
        predicted = self.predictor(x)
        target = torch.stack(self.received_messages)
        if self.message_type == self.MessageType.Categorical:
            target = target.argmax(dim=1)

        loss = self.loss_fn(predicted, target)
        if self.message_type == self.MessageType.Categorical:
            accuracy = torch.mean(predicted.argmax(dim=1) == target, dtype=torch.float)
        else:
            accuracy = torch.mean(predicted.argmax(dim=1) == target.argmax(dim=1), dtype=torch.float)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_metric.append(loss.item())
        self.accuracy_metric.append(accuracy.item())

        self.generated_messages = [torch.zeros(20)]
        self.received_messages = []

    '''
    def save_agent_state(self, directory: str):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.losses,
            'parties_won': self.parties_won,
            'reward_cum': self.reward_cum,
            'id': self.original_id,
        }
        torch.save(state, directory + str(self.original_id) + '.pt')

    def load_agent_state(self, path: str):
        state = torch.load(path)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.losses = state['losses']
        self.parties_won = state['parties_won']
        self.reward_cum = state['reward_cum']
        self.original_id = state['id']
    '''
