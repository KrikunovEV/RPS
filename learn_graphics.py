import torch
import matplotlib.pyplot as plt
import numpy as np


path = 'test2/'
players = 1

state = torch.load(path + '0' + '.pt')
data = state['losses']
parties_won = state['parties_won']
reward_cum = state['reward_cum']
plt.title(f'wins: {parties_won}, score = {reward_cum[-1]}')
plt.plot(np.convolve(data, np.full(31, 1. / 31), mode='valid'))
plt.xlabel('episode')
plt.ylabel('loss')
plt.tight_layout()
plt.savefig(f'{path}loss.png')

plt.title(f'wins: {parties_won}, score = {reward_cum[-1]}')
plt.plot(reward_cum)
plt.xlabel('episode')
plt.ylabel('cumulative reward')
plt.tight_layout()
plt.savefig(f'{path}reward.png')
