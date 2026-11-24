# Model Lab

Recorded experiments in OCR catalog segmentation and GPU performance. The reports include measurement setups, results, and limitations.

| Experiment | What was measured | Data |
| --- | --- | --- |
| [OCR catalog segmentation with LoRA](adaptation/lfm_lora/) | Checkpoint scores on 93 clean catalog rows and exact span reproduction on two long-page evaluation sets | [Checkpoint scores](adaptation/lfm_lora/clean_checkpoint_scores.csv) |
| [Fused MoE dispatch](systems/fused_dispatch/) | CUDA-event timings for PyTorch gather and four CUDA variants across four input shapes on an RTX 3060 Ti | [Dispatch timings](systems/fused_dispatch/timings.csv) |
| [Qwen training-step profiling](systems/step_profile/) | Operator costs and sampled GPU memory across forward, backward, and AdamW phases | [Operator profile](systems/step_profile/qwen_operator_profile.csv), [Memory timeline](systems/step_profile/qwen_memory_timeline.csv) |

## Run the local checks

The catalog and profiling reports also include small runnable examples. Python 3.12 is the reference environment. From the repository root, install the test dependencies and run the CPU suite.

```sh
python -m pip install -r requirements-cpu.txt
python -m pytest -q
```

The synthetic LFM2 adapter run also needs the model dependencies in `requirements-models.txt`. It downloads model weights on first use and benefits from a GPU.
