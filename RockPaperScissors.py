from enum import IntEnum
import numpy as np


class Choice(IntEnum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


class Environment3p:

    def __init__(self, steps: int, debug: bool = False):
        self.steps = steps
        self.players = 3
        self.debug = debug
        self.obs_space = self.players * len(Choice) * (steps - 1)
        self.reset()

    def action(self, choices):

        if self.step == self.steps:
            print('Please, reset the environment.')
            return np.zeros(self.players)

        self.__print(f'\nStep: {self.step}')

        choice1 = choices[0]
        choice2 = choices[1]
        choice3 = choices[2]

        if choice1 == choice2 == choice3:
            reward = np.zeros(self.players)

        elif choice1 == choice2:
            diff = choice3 - choice1
            reward = np.array([0., 0., 1.]) if diff == 1 or diff == -2 else np.array([0.5, 0.5, 0.])

        elif choice2 == choice3:
            diff = choice1 - choice3
            reward = np.array([1., 0., 0.]) if diff == 1 or diff == -2 else np.array([0., 0.5, 0.5])

        elif choice1 == choice3:
            diff = choice2 - choice3
            reward = np.array([0., 1., 0.]) if diff == 1 or diff == -2 else np.array([0.5, 0., 0.5])

        else:
            reward = np.zeros(self.players)

        for p in range(self.players):
            self.__print(f'Player {p}: {Choice(int(choices[p])).name}' + ('(winner)' if reward[p] != 0 else ''))

        offset = 3 * 3 * self.step
        if self.step < self.steps - 1:
            self.obs[offset + choice1] = 1
            self.obs[offset + 3 + choice2] = 1
            self.obs[offset + 6 + choice3] = 1
        self.step += 1
        return self.obs, reward

    def reset(self):
        self.step = 0
        self.obs = np.zeros(self.obs_space)
        return self.obs

    def get_obs_space(self):
        return self.obs_space

    def get_action_space(self):
        return len(Choice)

    def __print(self, str):
        if self.debug:
            print(str)
