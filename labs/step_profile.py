"""Break down a transformer training step by phase and tensor ownership."""

import argparse

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function


def storage_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors if torch.is_tensor(t))


def profile_step(device):
    torch.manual_seed(7)
    model = nn.TransformerEncoderLayer(32, 4, 64, dropout=0, batch_first=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    x = torch.randn(2, 24, 32, device=device)
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats()
    with profile(activities=activities, record_shapes=True, profile_memory=True) as trace:
        with record_function("forward"):
            loss = model(x).square().mean()
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
    buckets = {
        "parameters": storage_bytes(model.parameters()),
        "gradients": storage_bytes(p.grad for p in model.parameters()),
        "optimizer state": storage_bytes(
            value for state in optimizer.state.values() for value in state.values()
        ),
    }
    if device == "cuda":
        buckets["peak CUDA allocation"] = torch.cuda.max_memory_allocated()
    return trace, buckets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    trace, buckets = profile_step(args.device)
    print(trace.key_averages().table(sort_by="cpu_time_total", row_limit=12))
    for name, size in buckets.items():
        print(f"{name}: {size} bytes")
