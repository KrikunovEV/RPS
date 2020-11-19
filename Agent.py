import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from Model import PredictionModel, GenerationModel, DecisionModel
from RockPaperScissors import Choice
from enum import Enum


class MessageType(Enum):
    Categorical = 1
    Numerical = 2


def make_one_hot(message):
    message_ = torch.zeros(len(message))
    message_[torch.argmax(message)] = 1.
    return message_


class Agent:
    def __init__(self, id: int, obs_space: int, action_space: int, cfg, team: int = -1):
        # common part
        self.cfg = cfg
        self.agent_label = f'{id} ' + (f'team {team}' if team != -1 else 'solo')
        self.id = id
        self.team = team
        self.decision_maker = DecisionModel(obs_space=obs_space, action_space=action_space)
        self.optimizer1 = optim.SGD(self.decision_maker.parameters(), lr=cfg.lr)
        self.prob = []
        self.reward = []
        self.loss_metric = []
        self.reward_metric = []
        self.reward_eval_metric = []
        self.eval = False

        # negotiation part
        self.generator = GenerationModel(n_agents=cfg.negot.teams[team], message_space=cfg.negot.message_space)
        self.predictor = PredictionModel(n_agents=cfg.negot.teams[team], message_space=cfg.negot.message_space)
        self.optimizer2 = optim.Adam(list(self.generator.parameters()) + list(self.predictor.parameters()),
                                     lr=cfg.negot.lr)
        self.negot_loss_fn = nn.MSELoss() if cfg.negot.message_type == MessageType.Numerical else nn.CrossEntropyLoss()
        self.negot_loss = 0
        self.__init_messages()
        self.negot_loss_metric = []
        self.negot_accuracy_metric = []
        self.negot_step_accuracy_metric = torch.zeros((cfg.negot.steps - 1, cfg.negot.teams[team] - 1))
        self.negot_step_distance_metric = torch.zeros((cfg.negot.steps - 1, cfg.negot.teams[team] - 1))

    def set_eval(self, eval: bool):
        self.eval = eval

    def get_last_message(self, detach: bool = True):
        message = self.generated_messages[-1]
        if detach:
            message = message.detach()
        return message

    def negotiate(self, messages: list):
        """
        Данная функция на основе своего сообщения и полученных сообщений messages генерирует следующее сообщение.
        Градиент идёт ТОЛЬКО по всем сгенерированным сообщениям, т.к. messages являются detached.
        """

        if self.cfg.negot.message_type == MessageType.Categorical:
            messages = [make_one_hot(message) for message in messages]

        self.received_messages.append(torch.cat(messages))
        message = torch.cat((self.generated_messages[-1], self.received_messages[-1]))
        self.generated_messages.append(self.generator(message))

        return self.get_last_message()

    def negotiate_train(self):
        """
        Имеем (generated_messages : received_messages):
        A_0 : B_0
        A_1 : B_1
        ...
        A_n : B_n
        A_n+1 : -

        Пытаемся предсказать:
        A_0 B_0 -> B_1' vs B_1
        A_1 B_1 -> B_2' vs B_2
        ...
        A_n-1 B_n-1 -> B_n' vs B_n
        """

        generated_messages = torch.stack(self.generated_messages)
        received_messages = torch.stack(self.received_messages)

        x = torch.cat((generated_messages[:-2], received_messages[:-1]), dim=1)
        predicted = self.predictor(x)
        target = received_messages[1:]

        if self.eval:
            # step, agent, message
            target = target.reshape(-1, self.cfg.negot.teams[self.team] - 1, self.cfg.negot.message_space)
            predicted = predicted.reshape(-1, self.cfg.negot.teams[self.team] - 1, self.cfg.negot.message_space).detach()

            if self.cfg.negot.message_type == MessageType.Categorical:
                self.negot_step_accuracy_metric += predicted.argmax(dim=2) == target.argmax(dim=2)
            elif self.cfg.negot.message_type == MessageType.Numerical:
                self.negot_step_distance_metric += torch.sqrt(torch.sum((predicted - target) ** 2, dim=2))
        else:
            target = target.reshape(-1, self.cfg.negot.message_space)
            predicted = predicted.reshape(-1, self.cfg.negot.message_space)

            if self.cfg.negot.message_type == MessageType.Categorical:
                target = target.argmax(dim=1)

            self.negot_loss = self.negot_loss_fn(predicted, target)

            self.negot_loss_metric.append(self.negot_loss.item())
            if self.cfg.negot.message_type == MessageType.Categorical:
                accuracy = torch.mean(predicted.argmax(dim=1) == target, dtype=torch.float)
                self.negot_accuracy_metric.append(accuracy.item())

        self.__init_messages()

    def make_decision(self, obs, messages):
        """
        Данная функция отвечает за выбор действия на основе состояния среды и последних сгенерированных сообщений.
        Только сообщение данного агента должно пропускать градиент.
        """
        if len(messages) == 0:
            data = torch.Tensor(obs)
        else:
            data = torch.cat((torch.Tensor(obs), torch.cat(messages)))

        logits = self.decision_maker(data)

        if self.eval:
            action = logits.argmax().detach()
        else:
            policy = functional.softmax(logits, dim=-1)
            action = policy.argmax()
            self.prob.append(torch.log(policy[action]))
        return Choice(action.item())

    def rewarding(self, reward):
        if self.eval:
            self.reward_eval_metric.append(reward)
        else:
            self.reward.append(reward)
            self.reward_metric.append(reward)

    def train(self):
        """
        Применяем REINFORCE функцию и обновляем веса для переговоров и среды
        """
        rl_loss = (self.prob[1] * self.reward[1] + self.cfg.gamma * self.prob[0] * self.reward[0]) * (-1)
        loss = rl_loss + self.negot_loss

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        loss.backward()  # due to negot tensors
        self.optimizer1.step()
        self.optimizer2.step()

        self.loss_metric.append(rl_loss.item())

        self.prob = []
        self.reward = []

    def get_label(self):
        return self.agent_label

    def __init_messages(self):
        self.generated_messages = [torch.zeros(self.cfg.negot.message_space)]
        self.received_messages = []
