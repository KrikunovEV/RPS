from Model import ModelNeg, Model
import numpy as np
import torch
import torch.optim as optim


class Agent:

    def __init__(self, id: int, obs_space, action_space, negotiate: bool):
        self.id = id
        self.original_id = id
        self.negotiate = negotiate

        self.rewards, self.logs = [], []

        self.losses, self.reward_cum, self.parties_won = [], [0], 0

        if negotiate:
            self.model = ModelNeg(obs_space, action_space, action_space)
        else:
            self.model = Model(obs_space, action_space)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.00001)

    def __call__(self, obs, eval: bool = False):
        if self.negotiate:
            logits = self.model(obs[0], obs[1])
        else:
            logits = self.model(obs)
        policy = torch.softmax(logits, dim=-1)
        if eval:
            choice = policy.argmax()
        else:
            choice = np.random.choice(policy.shape[0], 1, p=policy.detach().numpy())[0]
        self.logs.append(torch.log(policy[choice]))
        return choice

    def give_reward(self, reward):
        self.rewards.append(reward)
        self.reward_cum.append(self.reward_cum[-1] + reward)
        if reward != 0:
            self.parties_won += 1

    def train(self):
        G = 0
        policy_loss = 0

        for i in reversed(range(len(self.rewards))):
            G = self.rewards[i] + 0.99 * G
            policy_loss = policy_loss - self.logs[i] * G

        self.losses.append(policy_loss.item())

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.logs, self.rewards = [], []

    def make_guess(self, obs, eval: bool = False):
        guess = self.model(obs)
        guess = guess.detach()
        policy = torch.softmax(guess, dim=-1)
        if eval:
            choice = policy.argmax()
        else:
            choice = np.random.choice(policy.shape[0], 1, p=policy.detach().numpy())[0]

        guess = np.zeros(guess.shape)
        guess[choice] = 1
        return guess

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
