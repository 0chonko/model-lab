"""Fit the same rank-2 residual using different adapter ranks."""

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False).to(base.weight)
        self.b = nn.Linear(rank, base.out_features, bias=False).to(base.weight)
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * self.b(self.a(x))


def rank_sweep():
    torch.manual_seed(7)
    x = torch.randn(128, 8)
    base = nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        y = base(x) + x @ torch.randn(8, 2) @ torch.randn(2, 4)
    for rank in (1, 2, 4):
        torch.manual_seed(7)
        layer = LoRALinear(base, rank, alpha=rank)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.03)
        for _ in range(300):
            optimizer.zero_grad()
            loss = (layer(x) - y).square().mean()
            loss.backward()
            optimizer.step()
        yield rank, (layer(x) - y).square().mean().item()


if __name__ == "__main__":
    for rank, error in rank_sweep():
        print(f"rank={rank} mse={error:.6f}")
