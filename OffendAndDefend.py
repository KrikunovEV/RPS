import numpy as np


class OADEnvironment:
    """
    Игроки должны выбрать кого атаковать и от кого защищаться.
    """
    OFFEND_ID: int = 0
    DEFEND_ID: int = 1
    WIN_REWARD: float = 1.
    LOOSE_REWARD: float = -1.
    ELO_K: float = 20.
    ELO_BASE: float = 10.
    ELO_DEL: float = 400.

    def __init__(self, players: int, debug: bool = False):
        self.players = players
        self.debug = debug
        self.obs_space = 2 * (players + 1)
        self.elo = np.full(players, 1000, dtype=np.int)

    def reset(self):
        obs = np.zeros((self.players, self.obs_space), dtype=np.float32)
        obs[:, (self.players, self.obs_space - 1)] = 1.
        return obs

    def reset_elo(self):
        self.elo = np.full(self.players, 1000, dtype=np.int)

    def play(self, choices: list):
        """
        Нам нужно лишь проверить, что на агента не было совершенно успешное нападение.
        Если он не смог защититься, то reward[agent_id] = -1
        """

        self.__print('\nНовый раунд: ')
        rewards = np.full(self.players, self.WIN_REWARD)
        for offender in range(self.players):
            defender = choices[offender][self.OFFEND_ID]
            if choices[defender][self.DEFEND_ID] != offender:
                rewards[defender] = self.LOOSE_REWARD
                self.__print(f'{offender} напал на {defender} (не защитился)')
            else:
                self.__print(f'{offender} напал на {defender} (защитился)')

        obs = np.zeros((self.players, self.obs_space), dtype=np.float32)
        for p in range(self.players):
            obs[p][choices[p][self.OFFEND_ID]] = 1.
            obs[p][self.players + 1 + choices[p][self.DEFEND_ID]] = 1.

        # zero sum
        winners_mask = rewards == self.WIN_REWARD
        total_win_reward = np.sum(winners_mask)
        if total_win_reward != 0:
            rewards[winners_mask] = 1. / total_win_reward

        loosers_mask = rewards == self.LOOSE_REWARD
        total_loose_reward = np.sum(loosers_mask)
        if total_loose_reward != 0:
            rewards[loosers_mask] = -1. / total_loose_reward

        # http://www.tckerrigan.com/Misc/Multiplayer_Elo/
        # Simple Multiplayer Elo (SME)
        '''
        # Use right or left player to correct your elo
        argind = np.argsort(self.elo)
        estimates = -self.elo
        for i in range(len(argind)):
            cur_player = argind[i]
            if rewards[cur_player] > 0.:
                vs_player = cur_player if i == 0 else argind[i - 1]
            else:
                vs_player = cur_player if i == (len(argind) - 1) else argind[i + 1]
            estimates[cur_player] += estimates[vs_player]
        '''

        # Use mean elo of other players to correct your elo
        elo_sum = np.sum(self.elo)
        estimates = ((elo_sum - self.elo) / (self.players - 1)) - self.elo  # mean R other - R current

        estimates = 1. / (1. + self.ELO_BASE**(estimates / self.ELO_DEL))  # sigmoid

        # S - E
        scores = np.zeros(self.players)
        scores[rewards > 0.] = 1.
        estimates = scores - estimates

        # compute rating
        self.elo = self.elo + (self.ELO_K * estimates).astype(np.int)

        self.__print(f'Награды {rewards}')
        self.__print(f'ELO {self.elo}')

        return obs, rewards, self.elo

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
