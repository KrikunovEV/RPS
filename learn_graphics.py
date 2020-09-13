import torch
import matplotlib.pyplot as plt
import numpy as np


path = 'test/'
players = 3

fig, ax = plt.subplots(players, figsize=(16, 9))
for i in range(players):
    state = torch.load(path + str(i) + '.pt')
    data = state['losses']
    parties_won = state['parties_won']
    reward_cum = state['reward_cum']
    ax[i].set_title(f'wins: {parties_won}, score = {reward_cum[-1]}')
    ax[i].plot(np.convolve(data, np.full(31, 1. / 31), mode='valid'))
    ax[i].set_xlabel('episode')
    ax[i].set_ylabel('loss')
fig.tight_layout()
plt.savefig(f'{path}loss.png')

fig, ax = plt.subplots(players, figsize=(16, 9))
for i in range(players):
    state = torch.load(path + str(i) + '.pt')
    parties_won = state['parties_won']
    reward_cum = state['reward_cum']
    ax[i].set_title(f'wins: {parties_won}, score = {reward_cum[-1]}')
    ax[i].plot(reward_cum)
    ax[i].set_xlabel('episode')
    ax[i].set_ylabel('cumulative reward')
fig.tight_layout()
plt.savefig(f'{path}reward.png')
