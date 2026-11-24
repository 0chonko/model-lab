# Training-step profiling

How do time and memory break down across the forward, backward, and Adam optimizer phases?

`python -m systems.step_profile.step_profile`

Compare parameter, gradient, and Adam state bytes. Add `--device cuda` to report peak CUDA allocation. Treat single-run times as illustrative.

## Synthetic run

On an RTX 3060 Ti with PyTorch 2.13.0+cu130, the seed-7 synthetic step allocated 17,302,528 bytes at peak. Parameters and gradients each occupied 34,176 bytes, and AdamW state occupied 68,400 bytes. This is one profiled step on a batch of two sequences of 24 tokens through a single width-32 encoder layer.

## Recorded Qwen profiles

These exports are from the original CUDA training profiler for Qwen2.5-0.5B-Instruct. The source configuration uses one 256-token chunk per step and AdamW. The CSVs describe two profiler runs.

- [Operator profile](qwen_operator_profile.csv) reports total and self CPU and CUDA times in microseconds, along with call counts and profiler memory deltas in bytes. Total times can include nested operations.
- [Memory timeline](qwen_memory_timeline.csv) reports elapsed milliseconds and memory in decimal MB. Rows identify forward (`fwd`), backward (`bwd`), and optimizer (`opt`) phases. `activ_mb` is the allocation remaining after subtracting parameters, gradients, and optimizer state. `autograd_mb` counts tensors saved for backward. These columns describe overlapping accounting views.
