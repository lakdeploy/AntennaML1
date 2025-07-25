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

if __name__ == "__main__":
    gen = AntennaGenerator()
    noise = torch.randn((16, 10))
    freq = torch.tensor([[60.0]] * 16)
    out = gen(noise, freq)
    print(out)
