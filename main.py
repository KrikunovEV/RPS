import matplotlib.pyplot as plt
from Agent import Agent
import numpy as np


negotiations = 1000
steps = 10
lr = 0.001

A = Agent(label='A', message_type=Agent.MessageType.Categorical, lr=lr, obs_space=20, action_space=20, steps=steps)
B = Agent(label='B', message_type=Agent.MessageType.Categorical, lr=lr, obs_space=20, action_space=20, steps=steps)

for negotiation in range(negotiations):
    print(f'negotiation: {negotiation + 1}/{negotiations}')

    Am = A.get_last_message()
    Bm = B.get_last_message()

    for step in range(steps):
        Am_prev = Am
        Bm_prev = Bm
        Am = A.generate_message(message=Bm_prev)
        Bm = B.generate_message(message=Am_prev)

    A.train()
    B.train()

test_negotiations = 1000
for negotiation in range(test_negotiations):
    print(f'test negotiation: {negotiation + 1}/{test_negotiations}')

    Am = A.get_last_message()
    Bm = B.get_last_message()

    for step in range(steps):
        Am_prev = Am
        Bm_prev = Bm
        Am = A.generate_message(message=Bm_prev)
        Bm = B.generate_message(message=Am_prev)

    A.eval()
    B.eval()

fig, ax = plt.subplots(1, 3, figsize=(16, 9))

ax[0].set_title(f'Loss')
ax[0].plot(np.convolve(A.loss_metric, np.full(15, 1./15.)), label=A.agent_label)
ax[0].plot(np.convolve(B.loss_metric, np.full(15, 1./15.)), label=B.agent_label)
ax[0].set_xlabel('# of negotiation')
ax[0].set_ylabel('value')
ax[0].legend()

ax[1].set_title(f'Accuracy')
if A.message_type == Agent.MessageType.Categorical:
    ax[1].plot(np.convolve(A.accuracy_metric, np.full(15, 1./15.)), label=A.agent_label)
if B.message_type == Agent.MessageType.Categorical:
    ax[1].plot(np.convolve(B.accuracy_metric, np.full(15, 1./15.)), label=B.agent_label)
ax[1].set_xlabel('# of negotiation')
ax[1].set_ylabel('value')
ax[1].legend()

ax[2].set_title(f'Accuracy by step')
if A.message_type == Agent.MessageType.Categorical:
    ax[2].plot(np.convolve(A.accuracy_by_step_metric / test_negotiations, np.full(15, 1./15.)), label=A.agent_label)
if B.message_type == Agent.MessageType.Categorical:
    ax[2].plot(np.convolve(B.accuracy_by_step_metric / test_negotiations, np.full(15, 1./15.)), label=B.agent_label)
ax[2].set_xlabel('negotiation step')
ax[2].set_ylabel('value')
ax[2].legend()


plt.tight_layout()
plt.show()
