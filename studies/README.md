# Experiment notes

These labs grew out of small investigations into how models store, route, and extract information. Each study records a baseline and limitations so the next run can test a specific change.

## Reproduced in the course

| Question | Control and change | Observation |
| --- | --- | --- |
| Does the rank of a LoRA adapter limit what it can fit? | Same frozen linear map, rank-2 residual, seed, and 300 Adam steps. Compare ranks 1, 2, and 4. | CPU MSE was 14.8470, below 0.000001, and 0.000082 respectively. Rank 2 had enough capacity on this synthetic target. The rank-4 optimization trace can vary with initialization and training budget. |
| What does activation checkpointing save? | Same transformer block, inputs, and loss. Enable non-reentrant checkpointing. | Saved-tensor bytes dropped from 108,544 to 12,288 while loss and parameter gradients matched. This counts saved tensors on a toy block. |

### LFM2 rank sweep on synthetic catalogs

LFM2‑700M on RTX 3060 Ti, BF16, 30 optimizer steps, seed 7, attention projections only. Both adapters used batch size one, the same split and generation budget, and a constant alpha-to-rank ratio of two.

| Model | Exact match | Mean span F1 |
| --- | ---: | ---: |
| Prompted base | 3/12 | 0.329 |
| Rank 4 | 6/12 | 0.761 |
| Rank 16 | 12/12 | 1.000 |
| Learned boundary tagger | 12/12 | 1.000 |
| Boundary rules | 12/12 | 1.000 |

```sh
python -m labs.lfm_lora --steps 30 --rank 4 --out runs/lfm-r4
python -m labs.lfm_lora --steps 30 --rank 16 --out runs/lfm-r16
python -m labs.sequence_tagger
```

The rank sweep uses one seed and 12 held-out synthetic examples. The exact punctuation cue also lets a linear tagger solve the task. Use noisier held-out formats and repeat across seeds before drawing conclusions about which rank generalizes.

## Recorded PMMP experiments

The measurements below come from the original research logs. The course uses independent synthetic fixtures for public reproduction of the methods.

### LFM2 catalog segmentation

On the same 93-row clean evaluation set, the prompted LFM2-700M baseline scored 0 exact matches. The first adapter scored 47, extending training context to 1,536 scored 50, and raising rank to 16 scored 55. The hard-case oversampling and epoch change scored 53. A subsequent run targeting extra MLP projections scored 56. The same recipe trained on clean data scored 71.

A later 700M run on an RTX 3060 Ti mixed clean and reconstructed pages and scored 82/93 in one clean evaluation dump. A separate dump scored 71/93 on the clean slice and 0/8 on the reconstructed slice it covered. The reconstructed slice covered only 8 of 23 prepared evaluation rows. A separate LFM3-1.7B run reached 81/93, while its longer-context variant reached 75/93. These are local task results with several co-varying settings, so use a fixed split and equal budgets to attribute changes in a new run.

Use the learned tagger and punctuation rules as controls. The synthetic corpus has a consistent boundary cue, so perfect scores there show the mechanics of the evaluation. Real OCR can merge punctuation with tokens, drop separators, and introduce unseen formats.

### EncodeMemory model editing

Treat the Qwen MLP down-projection as a map from an activation key to a residual-stream value. A rank‑1 term adds a learned value weighted by its overlap with the current activation. In the original run, capturing the final prompt token picked up shared wording about a city and made an unrelated subject respond to the edit. Capturing the last subject token preserved the unrelated control. Fitting across several sentence frames helped the edit transfer to a held-out phrasing. The lab compares the base response, fitted and held-out prompts, a control, and the disabled state.

### MoE and systems follow-ups

- **Routing:** compare learned top-k with label-forced routing. Track task quality, load by expert, and total assignments. The dual-input example retains the model from the ablation. Its expert forward pass uses dense sequence attention, so assignment sparsity alone cannot establish a speedup.
- **Fused dispatch:** the original CUDA progression compared expert-by-expert gather with histogram, scan, and expert-major packing. Use the PyTorch dispatch lesson as a reference when checking equivalent inputs and outputs. Record GPU, shapes, warmup, and precision before comparing speed.
- **Vision-language segmentation:** the local VLM splitter has fixture preparation and CLI stubs. The experiment would compare a layout detector and a LoRA-adapted VLM on entry-level page boxes using held-out pages and one-to-one IoU matching. Training and evaluation remain proposed.

### Fused dispatch timing

The RTX 3060 Ti log reports the median of three trials of 50 CUDA-event timings. All rows used top‑2 routing. Times are milliseconds.

| Tokens × width × experts | PyTorch gather | Global histogram + scalar pack | Privatized histogram | Vectorized pack | Warp top-2 + vectorized pack |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 × 4,096 × 32 | 5.595 | 3.043 | 3.038 | 1.984 | 1.979 |
| 4,096 × 4,096 × 256 | 40.798 | 13.026 | 13.031 | 12.000 | 11.941 |
| 16,384 × 16,384 × 32 | 18.080 | 47.307 | 47.406 | 30.932 | 30.903 |
| 1 × 4,096 × 32 | 3.758 | 0.205 | 0.205 | 0.098 | 0.097 |

Vectorizing the pack step improved the fused path at these shapes. Privatizing the histogram and then changing top‑2 had little effect in the recorded comparisons. The largest shape still favored the PyTorch reference. These are single-device results and the course does not bundle the original CUDA benchmark suite.
