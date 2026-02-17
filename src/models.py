import torch
import torch.nn as nn


class VNet(nn.Module):
    def __init__(self, hidden=256, depth=4):
        super().__init__()
        layers = []
        in_dim = 4

        for k in range(depth):
            layers.append(nn.Linear(in_dim if k == 0 else hidden, hidden))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x, s, t):
        if s.dim() == 1:
            s = s[:, None]
        if t.dim() == 1:
            t = t[:, None]

        inp = torch.cat([x, s, t], dim=1)
        return self.net(inp)