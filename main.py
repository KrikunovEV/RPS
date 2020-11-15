import matplotlib.pyplot as plt
from Agent import Agent, MessageType
import numpy as np


negotiations = 1000
steps = 10
lr = 0.001
n_agents = 4
space = 20

Agents = [
    Agent(label='A', message_type=MessageType.Categorical, lr=lr, message_space=space, n_agents=n_agents, steps=steps),
    Agent(label='B', message_type=MessageType.Categorical, lr=lr, message_space=space, n_agents=n_agents, steps=steps),
    Agent(label='С', message_type=MessageType.Categorical, lr=lr, message_space=space, n_agents=n_agents, steps=steps)
]


# Training
for negotiation in range(negotiations):
    print(f'negotiation: {negotiation + 1}/{negotiations}')

    messages = [agent.get_last_message() for agent in Agents]

    for step in range(steps):
        messages = [agent.generate_message(messages=messages[:i] + messages[i+1:]) for i, agent in enumerate(Agents)]

    for agent in Agents:
        agent.train()


# Testing
test_negotiations = 200
for negotiation in range(test_negotiations):
    print(f'test negotiation: {negotiation + 1}/{test_negotiations}')

    messages = [agent.get_last_message() for agent in Agents]

    for step in range(steps):
        messages = [agent.generate_message(messages=messages[:i] + messages[i+1:]) for i, agent in enumerate(Agents)]

    for agent in Agents:
        agent.eval()

fig, ax = plt.subplots(2, 3, figsize=(16, 9))

ax[1][0].set_title(f'MSE loss')
ax[1][0].set_yscale('log')
ax[1][0].set_xlabel('# of negotiation')
ax[1][0].set_ylabel('loss value')

ax[1][1].axis("off")

ax[1][2].set_title(f'Level-wise distance')
ax[1][2].set_xlabel('# of step')
ax[1][2].set_ylabel('value')

ax[0][0].set_title(f'CE loss')
#ax[0][0].set_yscale('log')
ax[0][0].set_xlabel('# of negotiation')
ax[0][0].set_ylabel('loss value')

ax[0][1].set_title(f'Accuracy')
ax[0][1].set_xlabel('# of negotiation')
ax[0][1].set_ylabel('accuracy value')

ax[0][2].set_title(f'Level-wise accuracy')
ax[0][2].set_xlabel('# of step')
ax[0][2].set_ylabel('accuracy value')

for agent in Agents:
    if agent.message_type == MessageType.Categorical:
        ax[0][0].plot(agent.loss_metric, label=agent.agent_label)
        ax[0][1].plot(agent.accuracy_metric, label=agent.agent_label)
        ax[0][2].plot(agent.level_accuracy_metric / test_negotiations, label=agent.agent_label)
    else:
        ax[1][0].plot(agent.loss_metric, label=agent.agent_label)
        ax[1][2].plot(agent.distance_metric.detach().numpy(), label=agent.agent_label)

ax[0][0].legend()
ax[0][1].legend()
ax[0][2].legend()
ax[1][0].legend()
ax[1][2].legend()

plt.tight_layout()
plt.show()
