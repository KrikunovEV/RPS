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
        self.action_space = 2

    def get_initial_obs(self):
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

        total_reward = rewards.sum()
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
    env = AADEnvironment(3, True)
    reward = []

    choices = [
        [1, 1],
        [0, 0],
        [0, 0]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [1, 1],
        [2, 0],
        [0, 0]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [1, 1],
        [0, 2],
        [0, 0]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [1, 1],
        [0, 0],
        [1, 0]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
            [1, 1],
            [0, 0],
            [0, 1]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [1, 1],
        [2, 2],
        [0, 0]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [1, 1],
        [0, 0],
        [1, 1]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [1, 1],
        [2, 0],
        [1, 0]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [1, 1],
        [0, 2],
        [0, 1]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [1, 1],
        [2, 0],
        [0, 1]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [1, 1],
        [0, 2],
        [1, 0]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    ''' ********** '''

    choices = [
        [2, 1],
        [0, 0],
        [0, 0]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [2, 1],
        [2, 0],
        [0, 0]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [2, 1],
        [0, 2],
        [0, 0]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [2, 1],
        [0, 0],
        [1, 0]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [2, 1],
        [0, 0],
        [0, 1]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [2, 1],
        [2, 2],
        [0, 0]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [2, 1],
        [0, 0],
        [1, 1]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [2, 1],
        [2, 0],
        [1, 0]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [2, 1],
        [0, 2],
        [0, 1]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [2, 1],
        [2, 0],
        [0, 1]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    choices = [
        [2, 1],
        [0, 2],
        [1, 0]
    ]
    obs, rewards = env.play(choices)
    reward.append(rewards[0])

    reward = reward + reward
    reward = np.array(reward)
    print(reward[reward != 0.].sum() / len(reward))
    print(reward)
