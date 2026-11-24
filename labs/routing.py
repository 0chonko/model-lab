"""Top-k expert routing and expert-major token dispatch."""

import torch
from torch import Tensor


def route_from_logits(logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
    topk_logits, topk_idx = torch.topk(logits, k, dim=-1)
    topk_w = torch.softmax(topk_logits, dim=-1)
    return topk_idx, topk_w


def route(x: Tensor, W_gate: Tensor, k: int) -> tuple[Tensor, Tensor]:
    return route_from_logits(x @ W_gate, k)


def dispatch(x: Tensor, topk_idx: Tensor, topk_w: Tensor, num_experts: int) -> list[tuple[Tensor, Tensor, Tensor]]:
    T, K = topk_idx.shape
    device = x.device
    token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, K)  # [T, K]

    flat_idx = topk_idx.reshape(-1)
    flat_w = topk_w.reshape(-1)
    flat_tok = token_ids.reshape(-1)

    out = []
    for expert in range(num_experts):
        mask = flat_idx == expert
        ids = flat_tok[mask]
        out.append((x[ids], flat_w[mask], ids))
    return out


def route_and_dispatch(
    x: Tensor, W_gate: Tensor, k: int
) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor, Tensor]]]:
    topk_idx, topk_w = route(x, W_gate, k)
    packed = dispatch(x, topk_idx, topk_w, W_gate.shape[1])
    return topk_idx, topk_w, packed


if __name__ == "__main__":
    torch.manual_seed(7)
    tokens = torch.randn(32, 8)
    gate = torch.randn(8, 4)
    for k in (1, 2, 4):
        ids, _, packed = route_and_dispatch(tokens, gate, k)
        print(f"k={k}: {ids.numel()} assignments, expert counts={[len(rows) for rows, _, _ in packed]}")
