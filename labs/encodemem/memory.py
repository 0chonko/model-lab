from contextlib import contextmanager

from torch import Tensor, nn


class LowRankMemory(nn.Module):
    """Wraps a Linear as y = W x + sum_i v_i (u_i . x)."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.terms: list[tuple[Tensor, Tensor]] = []
        self.enabled = True

    @property
    def d_in(self) -> int:
        return self.base.in_features

    @property
    def d_out(self) -> int:
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        if self.enabled:
            for u, v in self.terms:
                coeff = x.to(u.dtype) @ u
                y = y + (coeff.unsqueeze(-1) * v).to(y.dtype)
        return y

    def add(self, u: Tensor, v: Tensor) -> int:
        """Append a rank-1 term. Tensors are stored by reference, so a `v` with
        requires_grad stays trainable in place."""
        self.terms.append((u, v))
        return len(self.terms) - 1

    def clear(self) -> None:
        self.terms.clear()

    def __repr__(self) -> str:
        return (f"LowRankMemory({self.d_in} -> {self.d_out}, terms={len(self.terms)}, "
                f"enabled={self.enabled})")


def install(model, layer: int) -> LowRankMemory:
    mlp = model.model.layers[layer].mlp
    if not isinstance(mlp.down_proj, LowRankMemory):
        mlp.down_proj = LowRankMemory(mlp.down_proj)
    return mlp.down_proj


def uninstall(model, layer: int) -> None:
    mlp = model.model.layers[layer].mlp
    if isinstance(mlp.down_proj, LowRankMemory):
        mlp.down_proj = mlp.down_proj.base


@contextmanager
def disabled(memory: LowRankMemory):
    """Temporarily bypass the appended terms."""
    prev = memory.enabled
    memory.enabled = False
    try:
        yield
    finally:
        memory.enabled = prev
