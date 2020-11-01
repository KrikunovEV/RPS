import matplotlib.pyplot as plt
from Agent import Agent


negotiations = 5000
steps = 10
lr = 0.001

A = Agent(label='A', message_type=Agent.MessageType.Categorical, lr=lr, obs_space=20, action_space=20)
B = Agent(label='B', message_type=Agent.MessageType.Categorical, lr=lr, obs_space=20, action_space=20)

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

fig, ax = plt.subplots(1, 2, figsize=(16, 9))

ax[0].set_title(f'Loss')
ax[0].plot(A.loss_metric, label=A.agent_label)
ax[0].plot(B.loss_metric, label=B.agent_label)
ax[0].set_xlabel('# of negotiation')
ax[0].set_ylabel('value')
ax[0].legend()

ax[1].set_title(f'Accuracy')
ax[1].plot(A.accuracy_metric, label=A.agent_label)
ax[1].plot(B.accuracy_metric, label=B.agent_label)
ax[1].set_xlabel('# of negotiation')
ax[1].set_ylabel('value')
ax[1].legend()

plt.tight_layout()
plt.show()
