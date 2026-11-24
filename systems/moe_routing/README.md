# Mixture-of-experts routing

How does top-k affect total expert assignments and the load on the busiest expert?

`python -m systems.moe_routing.routing`

Sweep top-k on identical logits. The [dual-input ablation model](dual_input_moe.py) routes from embeddings and sends contextual states to experts. Its expert forward pass uses dense sequence attention, so assignment sparsity alone cannot establish a speedup. To compare learned top-k with label-forced routing, track task quality, load by expert, and total assignments.

## Synthetic run

The routing demo reuses the same 32 tokens and four experts for each fan-out, with seed 7.

| Top-k | Assignments | Expert counts |
| ---: | ---: | --- |
| 1 | 32 | 4, 16, 7, 5 |
| 2 | 64 | 15, 20, 13, 16 |
| 4 | 128 | 32, 32, 32, 32 |
