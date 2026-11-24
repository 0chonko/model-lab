# Mixture-of-experts routing

How does top-k affect total expert assignments and the load on the busiest expert?

`python -m systems.moe_routing.routing`

Sweep top-k on identical logits. The [dual-input ablation model](dual_input_moe.py) routes from embeddings and sends contextual states to experts. Its expert forward pass uses dense sequence attention, so assignment sparsity alone cannot establish a speedup. To compare learned top-k with label-forced routing, track task quality, load by expert, and total assignments.
