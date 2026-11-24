"""Compare checkpointing on the same transformer block and input."""

from copy import deepcopy
from time import perf_counter

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def train_step(model, x, checkpointed):
    saved_bytes = 0

    def pack(tensor):
        nonlocal saved_bytes
        saved_bytes += tensor.numel() * tensor.element_size()
        return tensor.detach()

    start = perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = checkpoint(model, x, use_reentrant=False) if checkpointed else model(x)
        loss = output.square().mean()
    loss.backward()
    return loss.detach(), saved_bytes, perf_counter() - start


def compare():
    torch.manual_seed(7)
    original = nn.TransformerEncoderLayer(32, 4, 64, dropout=0, batch_first=True)
    x = torch.randn(2, 24, 32)
    regular, recomputed = deepcopy(original), deepcopy(original)
    baseline = train_step(regular, x, False)
    checkpointed = train_step(recomputed, x, True)
    torch.testing.assert_close(baseline[0], checkpointed[0])
    for a, b in zip(regular.parameters(), recomputed.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    return baseline, checkpointed


if __name__ == "__main__":
    torch.set_num_threads(1)
    for use_checkpoint, (loss, size, seconds) in zip((False, True), compare()):
        print(f"checkpoint={use_checkpoint} loss={loss:.6f} saved={size} bytes time={seconds:.4f}s")
