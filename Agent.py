import torch
import torch.optim as optim
import torch.nn.functional as functional
import numpy as np
from Model import PredictionModel, GenerationModel, DecisionModel
from enum import Enum


class MessageType(Enum):
    Categorical = 1
    Numerical = 2


def make_one_hot(message):
    message_ = torch.zeros(len(message))
    message_[torch.argmax(message)] = 1.
    return message_


class Agent:
    def __init__(self, id: int, obs_space: int, action_space: int, cfg):
        obs_space = obs_space + cfg.players * cfg.negot.message_space

        self.cfg = cfg
        self.id = id
        self.negotiable = True if id >= cfg.n_agents else False
        self.agent_label = f'{id}' + (' negotiable' if self.negotiable else '')
        self.eval = False

        self.decision_maker = DecisionModel(obs_space, action_space)
        self.generator = GenerationModel(obs_space, cfg.negot.message_space, cfg.negot.steps)
        self.optimizer = optim.SGD(
            list(self.decision_maker.parameters()) + list(self.generator.parameters()),
            lr=cfg.lr
        )

        self.prob = []
        self.entropy = []
        self.reward = []

        self.loss_metric = []
        self.reward_metric = []
        self.reward_eval_metric = []

        self.__init_messages()

    def set_eval(self, eval: bool):
        self.eval = eval

    def get_last_message(self, detach: bool = True):
        message = self.generated_messages[-1]
        if detach:
            message = message.detach()
        return message

    def negotiate(self, messages: list, obs, step: int):
        """
        Данная функция на основе своего сообщения и полученных сообщений messages генерирует следующее сообщение.
        Градиент идёт ТОЛЬКО по всем сгенерированным сообщениям, т.к. messages являются detached.
        """

        if self.negotiable:
            if self.cfg.negot.message_type == MessageType.Categorical:
                messages = [make_one_hot(message) for message in messages]

            self.received_messages.append(torch.cat(messages))
            message = torch.cat((self.generated_messages[-1], self.received_messages[-1]))
            obs = torch.cat((obs, message))
            self.generated_messages.append(self.generator(obs, step))

        return self.get_last_message()

    def make_decision(self, obs, messages):
        """
        Данная функция отвечает за выбор действия на основе состояния среды и последних сгенерированных сообщений.
        Только сообщение данного агента должно пропускать градиент.
        """

        data = torch.cat((torch.Tensor(obs), torch.cat(messages)))
        a_logits, d_logits = self.decision_maker(data)

        if self.eval:
            a_policy = functional.softmax(a_logits.detach(), dim=-1)
            d_policy = functional.softmax(d_logits.detach(), dim=-1)
            a_action = np.random.choice(len(a_policy), p=a_policy.numpy())
            d_action = np.random.choice(len(d_policy), p=d_policy.numpy())
            #a_action = a_logits.argmax().detach()
            #d_action = d_logits.argmax().detach()
        else:
            a_policy = functional.softmax(a_logits, dim=-1)
            d_policy = functional.softmax(d_logits, dim=-1)

            '''
            if np.random.randint(5) == 0:
                a_action = np.random.randint(len(a_policy))
            else:
                a_action = a_policy.argmax().detach()

            if np.random.randint(5) == 0:
                d_action = np.random.randint(len(d_policy))
            else:
                d_action = d_policy.argmax().detach()
            '''

            a_action = np.random.choice(len(a_policy), p=a_policy.detach().numpy())
            d_action = np.random.choice(len(d_policy), p=d_policy.detach().numpy())
            #a_action = a_policy.argmax().detach()
            #d_action = d_policy.argmax().detach()
            self.prob.append(torch.log(a_policy[a_action] * d_policy[d_action]))
            self.entropy.append(torch.sum(a_policy * torch.log(a_policy)) + torch.sum(d_policy * torch.log(d_policy)))

        return [a_action, d_action]

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
        G = 0
        loss = 0
        for i in reversed(range(len(self.reward))):
            G = self.reward[i] + self.cfg.gamma * G
            loss = loss - G * self.prob[i]# + 0.001 * self.entropy[i]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_metric.append(loss.item())

        self.prob = []
        self.reward = []
        self.entropy = []
        self.__init_messages()

    def get_label(self):
        return self.agent_label

    def __init_messages(self):
        self.generated_messages = [torch.zeros(self.cfg.negot.message_space)]
        self.received_messages = []

    '''
    def negotiation_train(self):
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

            self.optimizer2.zero_grad()
            self.negot_loss.backward()
            self.optimizer2.step()

            self.negot_loss_metric.append(self.negot_loss.item())
            if self.cfg.negot.message_type == MessageType.Categorical:
                accuracy = torch.mean(predicted.argmax(dim=1) == target, dtype=torch.float)
                self.negot_accuracy_metric.append(accuracy.item())

        self.__init_messages()
    '''
