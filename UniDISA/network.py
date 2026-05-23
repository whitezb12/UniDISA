import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple
    

class VAEEncoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_latent: int
    ) -> None:
        super().__init__()
        n_hidden = 512
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc_mu = nn.Linear(n_hidden, n_latent)
        self.fc_logvar = nn.Linear(n_hidden, n_latent)

    def forward(self, x: torch.Tensor):
        h = self.fc1(x)
        h = F.relu(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

class Generator(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_input: int
    ) -> None:
        super().__init__()        
        n_hidden = 512
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_input)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc1(z)
        h = F.relu(h)
        out = self.fc2(h)
        return out


class BinaryDiscriminator(nn.Module):
    def __init__(self, n_input: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.net(x)
        return torch.clamp(score, min=-50.0, max=50.0)


class MultiClassDiscriminator(nn.Module):
    def __init__(self, n_input: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.net(x)
        return torch.clamp(score, min=-50.0, max=50.0)
