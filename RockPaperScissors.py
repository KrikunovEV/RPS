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
        self.action_space = len(Choice)
        self.obs_space = self.players * self.action_space
        self.obs = np.zeros(self.obs_space)

    def play(self, choices: list):
        unique_choices = np.unique(choices)
        rewards = np.zeros(self.players)
        '''
        Победители существуют, если уникальных значений два.
        Если уникальных значений одно, то все игроки выбрали одно действие.
        Если уникальных значений три, то победитель не может быть определён.
        '''
        if len(unique_choices) == 2:
            option1 = unique_choices[0]
            option2 = unique_choices[1]

            '''
            Для определённости делаем так, что option1 содержит наименьшее значениe из класса Choice
            Тогда разница между option2 и option1 может гарантированно быть либо 1, либо 2:
            1: ROCK vs PAPER (winners)
               PAPER vs SCISSOR (winners)
               Победители те, кто выбрал option2
            2: ROCK (winner) vs SCISSOR
               Победители те, кто выбрал option1
            
            Вознаграждение делим между всеми победителями!
            '''
            if option1 > option2:
                option1, option2 = option2, option1

            if option2 - option1 == 1:
                winners = np.where(np.array(choices) == option2)[0]
                rewards[winners] = 1. / winners.shape[0]
            else:
                winners = np.where(np.array(choices) == option1)[0]
                rewards[winners] = 1. / winners.shape[0]

        for p in range(len(choices)):
            self.__print(f'Player {p}: {choices[p].name}' + ('(winner)' if rewards[p] != 0 else ''))

        '''
        Состоянием среды является выбранные действия, закодированные в one-hot вектора 
        '''
        self.obs = np.zeros(self.obs_space)
        for i, choice in enumerate(choices):
            self.obs[int(choice) + i * self.action_space] = 1

        return self.obs, rewards

    def reset(self):
        self.obs = np.zeros(self.obs_space)
        return self.obs

    def get_obs_space(self):
        return self.obs_space

    def get_action_space(self):
        return self.action_space

    def __print(self, text):
        if self.debug:
            print(text)
