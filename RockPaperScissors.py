from enum import IntEnum
import numpy as np


class Choice(IntEnum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


class Environment3p:

    def __init__(self, debug: bool = False):
        self.players = 3
        self.debug = debug
        self.obs_space = self.players * len(Choice)
        self.obs = np.zeros(self.obs_space)

    def action(self, choices):
        choice1 = choices[0]
        choice2 = choices[1]
        choice3 = choices[2]

        if choice1 == choice2 == choice3:
            reward = np.zeros(self.players)

        elif choice1 == choice2:
            diff = choice3 - choice1
            reward = np.array([0., 0., 1.]) if diff == 1 or diff == -2 else np.array([1., 1., 0.])

        elif choice2 == choice3:
            diff = choice1 - choice3
            reward = np.array([1., 0., 0.]) if diff == 1 or diff == -2 else np.array([0., 1., 1.])

        elif choice1 == choice3:
            diff = choice2 - choice3
            reward = np.array([0., 1., 0.]) if diff == 1 or diff == -2 else np.array([1., 0., 1.])

        else:
            reward = np.zeros(self.players)

        for p in range(self.players):
            self.__print(f'Player {p}: {Choice(int(choices[p])).name}' + ('(winner)' if reward[p] != 0 else ''))

        self.obs[choice1] = 1
        self.obs[3 + choice2] = 1
        self.obs[6 + choice3] = 1
        return self.obs, reward

    def get_obs_space(self):
        return self.obs_space

    def get_action_space(self):
        return len(Choice)

    def __print(self, str):
        if self.debug:
            print(str)
