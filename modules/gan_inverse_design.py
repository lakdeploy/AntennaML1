import torch
import torch.nn as nn

class AntennaGenerator(nn.Module):
    def __init__(self, z_dim=10, cond_dim=1, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, z, cond):
        return self.net(torch.cat((z, cond), dim=1))

def generate_design(frequency, num=1):
    gen = AntennaGenerator()
    z = torch.randn((num, 10))
    cond = torch.full((num,1), float(frequency))
    return gen(z, cond).detach().numpy()

