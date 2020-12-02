import numpy as np


class AADEnvironment:
    """
    Игроки должны выбрать кого атаковать и от кого защищаться.
    Возможны 4 исхода для игроков i и j:
    - (Ничья)     Если игрок i атаковал игрока j,           а j защитился    от i,     то j выживет.
    - (Выживание) Если игрок i атаковал игрока j,           а j не защитился от i,     то j проиграет.
    - (Выживание) Если игрок i защитился от игрока j,       а j атаковал i,            то i выживет
    - (Поражение) Если игрок i не защитился от игрока j,    а j атаковал i,            то i проиграет

    action (2)
    Действие представляет из себя два числа

    obs (players * (players * action))
    Кодируем прошедший раунд one-hot векторами.

    choices (players, action)
    Каждый игрок делает 2 действия

    reward (players)
    Награда делится между выжившими игроками
    """

    def __init__(self, players: int, debug: bool = False):
        self.players = players
        self.debug = debug
        self.obs_space = players * (players * 2)
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
                rewards[attacked] = 0.
                self.__print(f'{p} defeated {attacked}')
            else:
                self.__print(f'{p} did not defeat {attacked}')

        obs = np.zeros(self.obs_space)
        for p in range(self.players):
            obs[p * 2 * self.players + choices[p][0]] = 1.
            obs[p * 2 * self.players + self.players + choices[p][1]] = 1.

        total_reward = 1#rewards.sum()
        if total_reward != 0:
            rewards = rewards / total_reward
        self.__print(rewards)

        return obs, rewards

    def get_obs_space(self):
        return self.obs_space

    def get_action_space(self):
        return self.action_space

    def __print(self, text):
        if self.debug:
            print(text)


if __name__ == '__main__':
    env = AADEnvironment(3)

    reward = []
    for e in range(10000):
        choices = [
            [a0, a1],
            [b0, b1],
            [c0, c1]
        ]
        obs, rewards = env.play(choices)
        reward.append(rewards[0])

    '''
    for a0 in [1, 2]:
        for a1 in [1, 2]:
            for b0 in [0, 2]:
                for b1 in [0, 2]:
                    for c0 in [0, 1]:
                        for c1 in [0, 1]:
                            choices = [
                                [a0, a1],
                                [b0, b1],
                                [c0, c1]
                            ]
                            obs, rewards = env.play(choices)
                            reward.append(rewards[0])

    reward = np.array(reward)
    uniques = np.unique(reward)
    print(f'parties: {reward.shape[0]}\n')

    for u in uniques:
        print(f'reward: {np.around(u, 3)}, '
              f'rate: {np.around(np.sum(reward == u) / reward.shape[0], 3)}, '
              f'benefit: {np.around(u * np.sum(reward == u) / reward.shape[0], 4)}')
    '''
