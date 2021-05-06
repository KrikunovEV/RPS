import torch
import torch.nn as nn
import torch.nn.functional as functional


class Attention(nn.Module):
    def __init__(self, cfg):
        super(Attention, self).__init__()
        self.cfg = cfg
        self.scale = torch.sqrt(torch.Tensor([cfg.train.hidden_size]))

        space = cfg.negotiation.space + cfg.common.players - 1
        var = 2. / (5. * space)
        self.WQ = nn.Parameter(torch.normal(0, var, (space, cfg.train.hidden_size)))
        self.WK = nn.Parameter(torch.normal(0, var, (space, cfg.train.hidden_size)))
        self.WV = nn.Parameter(torch.normal(0, var, (space, cfg.negotiation.space)))

    def forward(self, kv, q):
        Q = torch.matmul(q, self.WQ)
        K = torch.matmul(kv, self.WK)
        V = torch.matmul(kv, self.WV)

        O = torch.matmul(Q, K.T)
        O = torch.div(O, self.scale)
        O = functional.softmax(O, dim=-1)
        messages = torch.matmul(O, V)

        return messages


class NegotiationLayer(nn.Module):

    def __init__(self, cfg, emb):
        super(NegotiationLayer, self).__init__()
        self.cfg = cfg
        self.emb = emb

        self.ln_kv = nn.LayerNorm(cfg.negotiation.space)
        self.ln_q = nn.LayerNorm(cfg.negotiation.space)
        self.ln_m = nn.LayerNorm(cfg.negotiation.space)
        self.ln_ff = nn.LayerNorm(cfg.negotiation.space)

        self.attention = Attention(cfg)
        self.self_attention = Attention(cfg)
        self.FF = nn.Sequential(
            nn.Linear(cfg.negotiation.space, cfg.train.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.train.hidden_size, cfg.negotiation.space)
        )

    def forward(self, kv, q):
        kv_emb = torch.cat((self.emb, self.ln_kv(kv)), dim=1)
        q_emb = torch.cat((self.emb, self.ln_q(q)), dim=1)
        m = self.ln_m(self.attention(kv_emb, q_emb))

        kv_emb = torch.cat((self.emb, m + kv), dim=1)
        q_emb = torch.cat((self.emb, m + q), dim=1)
        m = self.ln_ff(self.self_attention(kv_emb, q_emb))

        m = self.FF(m)
        return m


class SiamMLP(nn.Module):

    def __init__(self, obs_space: int, action_space: int, cfg):
        super(SiamMLP, self).__init__()
        self.cfg = cfg

        self.policies = nn.Sequential(
            nn.Linear(obs_space, obs_space // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(obs_space // 2, 2),
        )
        self.V = nn.Linear(2 * (action_space - 1), 1)

    def forward(self, obs):
        actions = self.policies(obs).reshape(-1)
        return actions[::2], actions[1::2], self.V(actions)
