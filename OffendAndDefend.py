import numpy as np


class OADEnvironment:
    """
    Игроки должны выбрать кого атаковать и от кого защищаться.
    """
    OFFEND_ID: int = 0
    DEFEND_ID: int = 1
    NEGATIVE_REWARD: float = -1.

    def __init__(self, players: int, debug: bool = False):
        self.players = players
        self.debug = debug
        self.obs_space = 2 * (players + 1)

    def reset(self):
        obs = np.zeros((self.players, self.obs_space), dtype=np.float32)
        obs[:, (self.players, self.obs_space - 1)] = 1.
        return obs

    def play(self, choices: list):
        """
        Нам нужно лишь проверить, что на агента не было совершенно успешное нападение.
        Если он не смог защититься, то reward[agent_id] = -1
        """

        self.__print('\nНовый раунд: ')
        rewards = np.ones(self.players)
        for offender in range(self.players):
            defender = choices[offender][self.OFFEND_ID]
            if choices[defender][self.DEFEND_ID] != offender:
                rewards[defender] = self.NEGATIVE_REWARD
                self.__print(f'{offender} напал на {defender} (не защитился)')
            else:
                self.__print(f'{offender} напал на {defender} (защитился)')

        obs = np.zeros((self.players, self.obs_space), dtype=np.float32)
        for p in range(self.players):
            obs[p][choices[p][self.OFFEND_ID]] = 1.
            obs[p][self.players + 1 + choices[p][self.DEFEND_ID]] = 1.

        winners_mask = rewards == 1.
        total_win_reward = np.sum(winners_mask)
        if total_win_reward != 0:
            rewards[winners_mask] = 1. / total_win_reward
        self.__print(f'Награды {rewards}')

        return obs, rewards

    def get_obs_space(self):
        return self.obs_space

    def get_action_space(self):
        return self.players

    def __print(self, text):
        if self.debug:
            print(text)


if __name__ == '__main__':
    env = OADEnvironment(players=3, debug=True)
    print(env.reset())
    print(env.play([[1, 2], [0, 0], [0, 0]]))
    print(env.play([[0, 0], [1, 1], [2, 2]]))
