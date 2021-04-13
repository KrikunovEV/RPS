import torch
import torch.nn as nn
import torch.nn.functional as functional


class NegotiationModel(nn.Module):

    def __init__(self, obs_space: int, cfg):
        super(NegotiationModel, self).__init__()
        self.cfg = cfg
        obs_space *= cfg.common.players

        self.linear = nn.Sequential(
            nn.Linear(obs_space, obs_space // 2),
            nn.LeakyReLU()
        )
        self.policy = nn.Linear(obs_space // 2, cfg.negotiation.space)
        self.V = nn.Linear(obs_space // 2, 1)

    def forward(self, obs):
        obs = self.linear(obs.reshape(-1))
        return self.policy(obs), self.V(obs)


class AttentionNegotiationModel(nn.Module):

    def __init__(self, obs_space: int, cfg):
        super(AttentionNegotiationModel, self).__init__()
        self.cfg = cfg
        self.scale = torch.sqrt(torch.Tensor([cfg.train.hidden_size]))

        var = 1. / (2. * obs_space)
        #self.WQ = nn.Parameter(torch.normal(0, var, (obs_space, cfg.train.hidden_size), requires_grad=True))
        #self.WK = nn.Parameter(torch.normal(0, var, (obs_space, cfg.train.hidden_size), requires_grad=True))
        #self.WV = nn.Parameter(torch.normal(0, var, (obs_space, cfg.negotiation.space), requires_grad=True))
        self.WQ = nn.Linear(obs_space, cfg.train.hidden_size)
        self.WK = nn.Linear(obs_space, cfg.train.hidden_size)
        self.WV = nn.Linear(obs_space, cfg.negotiation.space)
        self.V = nn.Linear(obs_space, 1)

    def forward(self, obs, ind):

        query = obs[ind]
        if ind == 0:
            keys_values = obs[1:]
        elif ind == self.cfg.common.players - 1:
            keys_values = obs[:-1]
        else:
            keys_values = torch.cat((obs[:ind], obs[ind + 1:]))

        '''
        Q = torch.matmul(query, self.WQ)
        K = torch.matmul(keys_values, self.WK)
        V = torch.matmul(keys_values, self.WV)
        '''

        Q = self.WQ(query)
        K = self.WK(keys_values)
        V = self.WV(keys_values)

        O = torch.matmul(Q, K.T)
        O = torch.div(O, self.scale)
        O = functional.softmax(O, dim=-1)
        O = torch.matmul(O, V)

        return O, self.V(query)


class SiamMLPModel(nn.Module):

    def __init__(self, obs_space: int, action_space: int, cfg):
        super(SiamMLPModel, self).__init__()
        self.cfg = cfg

        self.policies = nn.Sequential(
            nn.Linear(obs_space, obs_space // 2),
            nn.LeakyReLU(),
            nn.Linear(obs_space // 2, 2),
        )
        self.V = nn.Linear(2 * action_space, 1)

    def forward(self, obs):
        actions = self.policies(obs).reshape(-1)
        return actions[::2], actions[1::2], self.V(actions)


class SiamRNNModel(nn.Module):

    def __init__(self, obs_space: int, action_space: int, cfg):
        super(SiamRNNModel, self).__init__()
        self.cfg = cfg

        if cfg.use_negotiation:
            obs_space += cfg.message_space + 1

        if cfg.use_embeddings:
            obs_space += cfg.embedding_space

        self.rnn = nn.GRUCell(obs_space, cfg.hidden_size)
        self.policies = nn.Linear(cfg.hidden_size, 2)
        self.V = nn.Linear(2 * action_space, 1)

    def get_h(self):
        return None

    def forward(self, obs, messages, embeddings, h):
        h = torch.zeros((1, self.cfg.hidden_size))
        actions = []

        for p in range(self.cfg.players):
            agent_obs = obs[p]
            if self.cfg.use_negotiation:
                agent_obs = torch.cat((agent_obs, messages[p]))
            if self.cfg.use_embeddings:
                agent_obs = torch.cat((agent_obs, embeddings[p]))
            h = self.rnn(agent_obs.unsqueeze(0), h)
            actions.append(self.policies(h[0]))

        actions = torch.cat(actions)
        return actions[::2], actions[1::2], self.V(actions), None


class AttentionModel(nn.Module):

    def __init__(self, obs_space: int, action_space: int, cfg):
        super(AttentionModel, self).__init__()
        self.cfg = cfg

        if cfg.use_negotiation:
            obs_space += cfg.message_space + 1

        if cfg.use_embeddings:
            obs_space += cfg.embedding_space

        self.rnn = nn.GRUCell(obs_space, cfg.hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden_size, 2),
            nn.LeakyReLU()
        )

        self.a_policy = nn.Linear(cfg.hidden_size, action_space)
        self.d_policy = nn.Linear(cfg.hidden_size, action_space)
        self.V = nn.Linear(cfg.hidden_size, 1)

    def get_h(self):
        return None

    def forward(self, obs, messages, embeddings, h):
        h = [torch.zeros((1, self.cfg.hidden_size))]
        actions = []
        for p in range(self.cfg.players):
            agent_obs = obs[p]
            if self.cfg.use_negotiation:
                agent_obs = torch.cat((agent_obs, messages[p]))
            if self.cfg.use_embeddings:
                agent_obs = torch.cat((agent_obs, embeddings[p]))
            h.append(self.rnn(agent_obs.unsqueeze(0), h[-1]))
            actions.append(self.ff(h[-1][0]))

        actions = torch.cat(actions)
        h = torch.stack(h[1:]).squeeze()
        a_attention = torch.sum(torch.softmax(actions[::2], -1).reshape(-1, 1) * h, dim=0)
        d_attention = torch.sum(torch.softmax(actions[1::2], -1).reshape(-1, 1) * h, dim=0)

        return self.a_policy(a_attention), self.d_policy(d_attention), self.V(a_attention + d_attention), None


class BaselineRNNModel(nn.Module):

    def __init__(self, obs_space: int, action_space: int, cfg):
        super(BaselineRNNModel, self).__init__()
        self.cfg = cfg

        obs_space = obs_space * cfg.players

        if cfg.use_negotiation:
            obs_space += (cfg.message_space + 1) * cfg.players

        if cfg.use_embeddings:
            obs_space += cfg.embedding_space * cfg.players

        self.rnn = nn.GRUCell(obs_space, cfg.hidden_size)
        self.a_policy = nn.Linear(cfg.hidden_size, action_space)
        self.d_policy = nn.Linear(cfg.hidden_size, action_space)
        self.V = nn.Linear(cfg.hidden_size, 1)

    def get_h(self):
        return torch.zeros((1, self.cfg.hidden_size))

    def forward(self, obs, messages, embeddings, h):
        obs = obs.reshape(-1)

        if self.cfg.use_negotiation:
            messages = messages.reshape(-1)
            obs = torch.cat((obs, messages))

        if self.cfg.use_embeddings:
            embeddings = embeddings.reshape(-1)
            obs = torch.cat((obs, embeddings))

        new_h = self.rnn(obs.unsqueeze(0), h)
        return self.a_policy(new_h[0]), self.d_policy(new_h[0]), self.V(new_h[0]), new_h.data


class BaselineMLPModel(nn.Module):

    def __init__(self, obs_space: int, action_space: int, cfg):
        super(BaselineMLPModel, self).__init__()
        self.cfg = cfg

        obs_space = obs_space * cfg.players

        if cfg.use_negotiation:
            obs_space += (cfg.message_space + 1) * cfg.players

        if cfg.use_embeddings:
            obs_space += cfg.embedding_space * cfg.players

        self.linear = nn.Sequential(
            nn.Linear(obs_space, obs_space // 2),
            nn.LeakyReLU()
        )
        self.a_policy = nn.Linear(obs_space // 2, action_space)
        self.d_policy = nn.Linear(obs_space // 2, action_space)
        self.V = nn.Linear(obs_space // 2, 1)

    def get_h(self):
        return None

    def forward(self, obs, messages, embeddings, h):
        obs = obs.reshape(-1)

        if self.cfg.use_negotiation:
            messages = messages.reshape(-1)
            obs = torch.cat((obs, messages))

        if self.cfg.use_embeddings:
            embeddings = embeddings.reshape(-1)
            obs = torch.cat((obs, embeddings))

        obs = self.linear(obs)
        return self.a_policy(obs), self.d_policy(obs), self.V(obs), None


if __name__ == '__main__':
    import utility as util
    cfg = util.load_config('config/default.yaml')
    model = AttentionNegotiationModel(3, cfg)

    obs = torch.normal(0, 0.1, (3,), dtype=torch.float)
    obs = torch.stack((obs, torch.full((3,), 2, dtype=torch.float)))
    obs = torch.cat((obs, torch.full((1, 3), 3, dtype=torch.float)))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    print(obs)

    for i in range(5):
        output = model(obs, 0)[0]
        loss = -torch.log(functional.softmax(output, dim=-1).max())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(output.data)
        print(model.WQ.grad)

    ind = [1, 0, 2]

    print()
    print(ind)
    print()

    #print(output20.data, output21.data, output22.data)
