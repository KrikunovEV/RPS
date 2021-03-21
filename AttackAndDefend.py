import numpy as np


class AADEnvironment:
    """
    Игроки должны выбрать кого атаковать и от кого защищаться.
    """

    def __init__(self, players: int, debug: bool = False):
        self.players = players
        self.debug = debug
        self.obs_space = players * (2 * (players + 1))  # +1 for empty
        self.action_space = players

    def reset(self):
        return np.zeros(self.obs_space)

    def play(self, choices: list):
        """
        We just need to check that all players have defended.
        If someone not, do reward[someone] = 0
        """

        rewards = np.ones(self.players)
        for p in range(self.players):
            attacked = choices[p][0]
            if choices[attacked][1] != p:
                rewards[attacked] = -1.
                self.__print(f'{p} defeated {attacked}')
            else:
                self.__print(f'{p} did not defeat {attacked}')

        obs = np.zeros(self.obs_space)
        for p in range(self.players):
            obs[p * 2 * (self.players + 1) + choices[p][0]] = 1.
            obs[p * 2 * (self.players + 1) + (self.players + 1) + choices[p][1]] = 1.

        #winners = np.sum(rewards == 1.)
        #if winners != 0:
        #    rewards[rewards == 1.] = 1. / winners
        self.__print(rewards)

        return obs, rewards

    def get_obs_space(self):
        return self.obs_space

    def get_action_space(self):
        return self.action_space

    def __print(self, text):
        if self.debug:
            print(text)
