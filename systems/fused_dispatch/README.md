# Fused dispatch timing

How did histogram, scan, and expert-major packing compare with expert-by-expert gather?

The RTX 3060 Ti log reports the median of three trials of 50 CUDA-event timings. All rows used top‑2 routing. Times are milliseconds.

| Tokens × width × experts | PyTorch gather | Global histogram + scalar pack | Privatized histogram | Vectorized pack | Warp top-2 + vectorized pack |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 × 4,096 × 32 | 5.595 | 3.043 | 3.038 | 1.984 | 1.979 |
| 4,096 × 4,096 × 256 | 40.798 | 13.026 | 13.031 | 12.000 | 11.941 |
| 16,384 × 16,384 × 32 | 18.080 | 47.307 | 47.406 | 30.932 | 30.903 |
| 1 × 4,096 × 32 | 3.758 | 0.205 | 0.205 | 0.098 | 0.097 |

Vectorizing the pack step improved the fused path at these shapes. Privatizing the histogram and then changing top‑2 had little effect in the recorded comparisons. The largest shape still favored the PyTorch reference. These are single-device results from the original CUDA benchmark. The [PyTorch dispatch example](../moe_routing/routing.py) provides a reference for checking equivalent inputs and outputs. Record GPU, shapes, warmup, and precision when comparing speed.
