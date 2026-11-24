"""Dual-Input Gated MoE: router on embeddings, frozen base h_base, 20 experts."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualInputExpert(nn.Module):
    def __init__(self, d_model: int, d_expert_hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_expert_hidden, bias=False)
        self.k_proj = nn.Linear(d_model, d_expert_hidden, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h_base: torch.Tensor,
        gate: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        h_base: (B, L, D)
        gate:   (B, L, 1) per-token gate for this expert (0 where not selected)
        attn_mask: (B, L) 1=keep, 0=pad
        """
        Q = self.q_proj(h_base)
        K = self.k_proj(h_base)
        V = self.v_proj(h_base)
        scale = Q.shape[-1] ** -0.5
        logits = (Q @ K.transpose(-2, -1)) * scale
        if attn_mask is not None:
            # mask keys: (B, 1, L)
            key_mask = attn_mask[:, None, :].bool()
            logits = logits.masked_fill(~key_mask, torch.finfo(logits.dtype).min)
        attn = F.softmax(logits, dim=-1)
        out = attn @ V
        y = self.layer_norm(self.out_proj(out) + h_base)
        return y * gate


class DualInputMoE(nn.Module):
    def __init__(
        self,
        num_experts: int = 20,
        top_k: int = 2,
        d_model: int = 896,
        d_expert_hidden: int = 256,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [DualInputExpert(d_model, d_expert_hidden) for _ in range(num_experts)]
        )
        self.schema_head = nn.Linear(d_model, num_experts)

    def route(
        self,
        emb: torch.Tensor,
        force_labels: torch.Tensor | None = None,
        ignore_index: int = -100,
    ):
        """
        emb: (B, L, D)
        force_labels: (B, L) class ids for phase-1 supervised routing; ignore_index skipped.
        Returns gates (B, L, E) dense after softmax-then-topk, router_logits, aux dict.
        """
        logits = self.router(emb)  # (B, L, E)
        probs = F.softmax(logits, dim=-1)

        if force_labels is not None:
            gates = torch.zeros_like(probs)
            valid = force_labels != ignore_index
            # one-hot on labeled tokens; unlabeled get uniform over top-k from probs
            oh = F.one_hot(
                force_labels.clamp(min=0), num_classes=self.num_experts
            ).float()
            gates = torch.where(valid.unsqueeze(-1), oh, gates)
            # for unlabeled: use learned top-k so graph still flows a bit
            topk_p, topk_i = torch.topk(probs, self.top_k, dim=-1)
            topk_w = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            sparse = torch.zeros_like(probs)
            sparse.scatter_(-1, topk_i, topk_w)
            gates = torch.where(valid.unsqueeze(-1), gates, sparse)
        else:
            topk_p, topk_i = torch.topk(probs, self.top_k, dim=-1)
            topk_w = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            gates = torch.zeros_like(probs)
            gates.scatter_(-1, topk_i, topk_w)

        # Load-balance: fraction of tokens per expert vs mean gate mass
        with torch.no_grad():
            dispatch = (gates > 0).float().mean(dim=(0, 1))
        importance = gates.mean(dim=(0, 1))
        load_loss = self.num_experts * (dispatch * importance).sum()

        return gates, logits, {"load_loss": load_loss, "probs": probs}

    def forward(
        self,
        emb: torch.Tensor,
        h_base: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        force_labels: torch.Tensor | None = None,
        ignore_index: int = -100,
    ):
        gates, router_logits, aux = self.route(emb, force_labels, ignore_index)
        # Dense expert mix: each expert sees full sequence; gate zeros inactive tokens.
        combined = torch.zeros_like(h_base)
        for e, expert in enumerate(self.experts):
            g = gates[:, :, e : e + 1]
            if not (g > 0).any():
                continue
            combined = combined + expert(h_base, g, attn_mask)
        logits = self.schema_head(combined)
        return logits, router_logits, aux
