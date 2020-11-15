from AgentOrchestrator import AgentOrchestrator, MessageType


train_negotiations = 1000
test_negotiations = 200
steps = 10
n_agents = 3
message_space = 20
lr = 0.001

agent_orchestrator = AgentOrchestrator(lr, message_space, steps, n_agents, MessageType.Numerical)


# Training
for negotiation in range(train_negotiations):
    print(f'negotiation: {negotiation + 1}/{train_negotiations}')

    messages = agent_orchestrator.get_last_messages()

    for step in range(steps):
        messages = agent_orchestrator.generate_messages(messages=messages)

    agent_orchestrator.train()


# Testing
for negotiation in range(test_negotiations):
    print(f'test negotiation: {negotiation + 1}/{test_negotiations}')

    messages = agent_orchestrator.get_last_messages()

    for step in range(steps):
        messages = agent_orchestrator.generate_messages(messages=messages)

    agent_orchestrator.eval()


# Metrics
agent_orchestrator.plot_metrics(test_negotiations, directory='test')
