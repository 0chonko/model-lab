# model-lab

Small experiments on model behavior, weights, and GPU performance.

## [Embeddings](embeddings/)

- [EmbeddingGemma retrieval](embeddings/embedding_gemma/)

  Start with a tiny corpus to see whether a question retrieves the passage that answers it before tuning chunk size and overlap on technical docs.

## [Model editing](model_editing/)

- [Fact injection](model_editing/fact_injection/)

  Test whether a reversible rank-1 edit to Qwen can teach one association across paraphrases while preserving the answer to an unrelated prompt.

## [Adaptation](adaptation/)

- [LoRA from scratch](adaptation/low_rank/)

  Build a LoRA layer in isolation to see how rank limits what it can learn with the base weights frozen.

- [LoRA on LFM2](adaptation/lfm_lora/)

  Adapt a small model to split OCR catalog text into exact source spans, and measure how much adapter rank matters on a fixed training budget.

## [Systems](systems/)

- [KV cache](systems/kv_cache/)

  Estimate what a million-token context costs in GPU memory and how grouped-query attention, cache precision, and sliding windows change that budget.

- [Activation checkpointing](systems/training_memory/)

  Measure how much activation memory recomputation saves on a small Transformer block and check that the gradients still agree.

- [Training-step profiling](systems/step_profile/)

  Break a training step into forward, backward, and optimizer phases to see where time and memory go before choosing what to optimize.

- [MoE routing](systems/moe_routing/)

  Measure how top-k changes expert load, then explore a catalog-tagging design in which the router sees input embeddings and experts see contextual states.

- [Fused dispatch timings](systems/fused_dispatch/)

  Find which token shapes benefit from replacing per-expert gathers with CUDA packing, and which still favor the PyTorch baseline.

## [Structured outputs](structured_outputs/)

- [Catalog segmentation](structured_outputs/catalog_segmentation/)

  Compare a learned boundary tagger with punctuation rules before tackling OCR where separators may be missing or merged.

- [Extraction with evidence](structured_outputs/extraction/)

  Make structured manuscript records auditable by checking that extracted claims point to unambiguous source text and that their dependencies are valid.

- [Box evaluation](structured_outputs/box_evaluation/)

  Measure whether layout predictions recover individual catalog entries when nearby entries can be merged into one box or an entry can be split across boxes.

[Learn](learn/) contains tutorials on tokenization, PyTorch, attention, Transformer architecture, and distributed training.

## Run data

Download the [fused-dispatch timings](systems/fused_dispatch/timings.csv), [Qwen operator profile](systems/step_profile/qwen_operator_profile.csv), [Qwen memory timeline](systems/step_profile/qwen_memory_timeline.csv), and [catalog checkpoint scores](adaptation/lfm_lora/clean_checkpoint_scores.csv). The experiment READMEs describe the original runs and give commands and results for the small examples in this repo.

## Setup

Run commands from the repository root. Python 3.12 is the reference environment.

```sh
python -m pip install -r requirements-course.txt
```

For EmbeddingGemma, Qwen, and LFM2, also install the [model dependencies](requirements-models.txt) with `python -m pip install -r requirements-models.txt`. Use a torchvision build compatible with your torch installation. Model runs download weights on first use and benefit from a GPU.

## Future idea

Change a single model weight and compare tone on fixed prompts before and after the edit. Include unrelated prompts to check for broader changes.
