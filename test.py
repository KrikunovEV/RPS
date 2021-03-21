import torch
import torch.nn as nn

print([False] * 5 + [True] * 5)
gru = nn.GRUCell(10, 2)
m = torch.randn((5, 10))
h = torch.randn((1, 2))

at = []
for mi in m:
    at.append(gru(mi.unsqueeze(0), h)[0])

print(torch.cat(at))