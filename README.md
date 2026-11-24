# model-lab

Small experiments on model behavior, weights, and GPU performance.

- [Embeddings](embeddings/) — [EmbeddingGemma retrieval](embeddings/embedding_gemma/).
- [Model editing](model_editing/) — [fact injection](model_editing/fact_injection/).
- [Adaptation](adaptation/) — [LoRA from scratch](adaptation/low_rank/) and [LoRA on LFM2](adaptation/lfm_lora/).
- [Systems](systems/) — [KV cache](systems/kv_cache/), [activation checkpointing](systems/training_memory/), [training-step profiling](systems/step_profile/), [MoE routing](systems/moe_routing/), and [fused dispatch timings](systems/fused_dispatch/).
- [Structured outputs](structured_outputs/) — [catalog segmentation](structured_outputs/catalog_segmentation/), [extraction with evidence](structured_outputs/extraction/), and [box evaluation](structured_outputs/box_evaluation/).
- [Learn](learn/) — tutorials on tokenization, PyTorch, attention, Transformer architecture, and distributed training.

## Setup

Run commands from the repository root. Python 3.12 is the reference environment.

```sh
python -m pip install -r requirements-course.txt
```

For EmbeddingGemma, Qwen, and LFM2, also install the [model dependencies](requirements-models.txt) with `python -m pip install -r requirements-models.txt`. Use a torchvision build compatible with your torch installation. Model runs download weights on first use and benefit from a GPU.

## Future idea

Change a single model weight and compare tone on fixed prompts before and after the edit. Include unrelated prompts to check for broader changes.
