# Lab guide

Run these commands from the repository root. Python 3.12 is the CI reference environment.

| Lesson | Run | Exercise |
| --- | --- | --- |
| 1. Attention memory | `python -m labs.kv_cache` | Change KV head count, element size, and sliding-window length one at a time. Explain the slope before and after the window. |
| 2. Activation checkpointing | `python -m labs.training_memory` | Compare saved tensor bytes and confirm equal gradients. Increase sequence length while holding model width fixed. |
| 3. Training-step profiling | `python -m labs.step_profile` | Inspect forward, backward, and optimizer phases. Compare parameter, gradient, and Adam state bytes. Add `--device cuda` for peak CUDA allocation. |
| 4. Mixture of experts | `python -m labs.routing` | Sweep top-k on identical logits. Compare assignment count and busiest expert. Then inspect [dual-input gates](dual_input_moe.py), where the router sees embeddings and experts see contextual states. |
| 5. LoRA from scratch | `python -m labs.low_rank` | Fit the same target with ranks 1, 2, and 4. Explain the zero-initialized output projection and inspect which weights get gradients. |
| 6. Tagging vs. generation | `python -m labs.sequence_tagger` | Compare a learned boundary tagger and a rule baseline. Break the synthetic delimiter convention and measure both again. |
| 7. Extraction with evidence | `python -m labs.extraction` | Change a quote, repeat it in the source, or introduce a cycle. Check how the compiler responds. |
| 8. Layout box evaluation | `python -m labs.box_evaluation` | Compare merged and split boxes on a synthetic page. Sweep the IoU threshold and add a duplicate prediction. |

Saved-tensor bytes count tensors submitted to autograd's save hook. Views, repeated saves, and allocator behavior make this a different measure from peak allocation. Treat the displayed single-run times as illustrative.

The KV-cache example counts keys and values across 16 full and 46 sliding layers. At one million tokens, each new token adds 32 KiB across the full layers, or 4 KiB per device under ideal eight-way head sharding. The sliding layers contribute a bounded allocation. The formula excludes padding and allocator overhead.

## Model runs

Install the extra dependencies with `python -m pip install -r requirements-models.txt`. Use a torchvision build compatible with your torch installation.

| Lesson | Run | Exercise |
| --- | --- | --- |
| 9. EncodeMemory | `python -m labs.model_editing.wrap_noop`, then `python -m labs.model_editing.inject_fact` | Add a rank-1 association to Qwen2.5-1.5B. Compare paraphrases, an unrelated control, and logits with the edit disabled. |
| 10. LoRA on LFM2 | `python -m labs.lfm_lora --steps 2 --out runs/lfm-smoke` | Smoke-test adapter training, saving, reloading, and scoring. For an ablation use equal step counts and separate output paths while varying only `--rank` or `--targets`. |
| 11. Embeddings and retrieval | `python -m labs.embeddings` | Rank a tiny text corpus by cosine similarity. Add a lexical distractor and write down the ranking before running it. |

For LFM2, the default training budget is 60 steps with batch size one. GPU execution is preferable for the model labs. The tagger and LFM2 use the same seeded synthetic evaluation data and share the span F1 definition. Exact match also checks span order. Report both metrics, the model ID, random seed, and training budget when adding results.
