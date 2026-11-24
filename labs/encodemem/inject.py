import torch
import torch.nn.functional as F
from torch import Tensor

from .memory import LowRankMemory


def target_token_id(tok, target: str) -> int:
    ids = tok.encode(target, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{target!r} is {len(ids)} tokens: {tok.convert_ids_to_tokens(ids)}")
    return ids[0]


def unembed_value(model, target_id: int, scale: float = 1.0) -> Tensor:
    """Closed-form sanity value: push the residual stream along the target's output
    embedding. Moves the logit, but it is a blunt instrument compared to fitting."""
    w = model.get_output_embeddings().weight[target_id].detach().float()
    return scale * w / w.norm()


def fit_value(model, tok, memory: LowRankMemory, u: Tensor, prompts: list[str], target: str,
              steps: int = 200, lr: float = 0.1, l2: float = 5e-4, stop_ce: float = 0.01,
              log_every: int = 10) -> Tensor:
    """Gradient-fit v so that (u, v) makes the model emit `target` after `prompts`.

    The model is frozen; v is the only leaf with requires_grad, so backprop through
    the whole network lands on a single 1536-dim vector.
    """
    target_id = target_token_id(tok, target)
    batch = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    last = batch["attention_mask"].sum(1) - 1
    labels = torch.full((len(prompts),), target_id, device=model.device)

    u = u.detach().to(model.device)
    v = torch.zeros(memory.d_out, device=model.device, dtype=torch.float32, requires_grad=True)
    index = memory.add(u, v)
    opt = torch.optim.Adam([v], lr=lr)

    for step in range(1, steps + 1):
        logits = model(**batch).logits[torch.arange(len(prompts)), last].float()
        ce = F.cross_entropy(logits, labels)
        loss = ce + l2 * v.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == 1):
            print(f"  step {step:4d}  ce {ce.item():.4f}  |v| {v.norm().item():.2f}")
        if ce.item() < stop_ce:
            print(f"  converged at step {step}, ce {ce.item():.4f}")
            break

    memory.terms[index] = (u, v.detach())
    return v.detach()
