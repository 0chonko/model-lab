# Training-step profiling

How do time and memory break down across the forward, backward, and Adam optimizer phases?

`python -m systems.step_profile.step_profile`

Compare parameter, gradient, and Adam state bytes. Add `--device cuda` to report peak CUDA allocation. Treat single-run times as illustrative.
