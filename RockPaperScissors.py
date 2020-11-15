from enum import IntEnum
import numpy as np


class Choice(IntEnum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


class RPSEnvironment:

    def __init__(self, players: int, debug: bool = False):
        self.players = players
        self.debug = debug
        self.obs_space = self.players * len(Choice)
        self.obs = np.zeros(self.obs_space)
        self.action_space = len(Choice)

    def step(self, choices: list):
        unique_choices = np.unique(choices)
        rewards = np.zeros(self.players)
        if len(unique_choices) == 2:
            option1 = unique_choices[0]
            option2 = unique_choices[1]

            if option1 > option2:
                option1, option2 = option2, option1

            # PAPERs VS ROCKs and SCISSORs vs PAPERs
            if option2 - option1 == 1:
                rewards[choices == option2] = 1.

            # ROCKs vs SCISSORs
            else:
                rewards[choices == option1] = 1.

        for p in range(len(choices)):
            self.__print(f'Player {p}: {choices[p].name}' + ('(winner)' if rewards[p] != 0 else ''))

        self.obs[choice1] = 1
        self.obs[3 + choice2] = 1
        self.obs[6 + choice3] = 1
        return self.obs, rewards

    def get_obs_space(self):
        return self.obs_space

    def get_action_space(self):
        return self.action_space

    def __print(self, str):
        if self.debug:
            print(str)
