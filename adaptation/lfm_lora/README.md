# LoRA on LFM2

## Motivation

The manuscript migration needs each citation preserved as an exact source span before a later classifier can label it and attach it to a manuscript or chapter. The post-LoRA notes flag long fields because an autoregressive splitter can omit or merge entries as the list grows. They propose a boundary tagger that selects spans directly from the input. The adapter results give that follow-up a comparison point.

LFM2-700M was fine-tuned to split OCR catalog text into exact source spans. This report covers checkpoint selection on hand-checked catalog rows, reproduction of long reconstructed pages, and a runnable synthetic rank comparison.

## Recorded catalog runs

The library migration needs one record per reference. These runs trained on about 840 hand-checked catalog rows or their paired OCR-noisy variants. The mixed-data run added 131 reconstructed long catalogs. The same 93 clean validation rows were reused across configurations and checkpoint selection.

| Training recipe | Final exact | Best exact | Final exact for 5–7 spans |
| --- | ---: | ---: | ---: |
| Prompted LFM2-700M | 0/93 | — | 0/32 |
| Noisy LFM2 baseline | 64/93 | 66/93 | 16/32 |
| Noisy, longer context | 61/93 | 62/93 | 15/32 |
| Noisy, rank 16 | 66/93 | 66/93 | 17/32 |
| Noisy, oversampled hard rows | 66/93 | 73/93 | 19/32 |
| Noisy, extra MLP projections | 69/93 | 74/93 | 20/32 |
| Clean, same MLP recipe | 71/93 | 75/93 | 20/32 |
| Clean LFM3-1.7B | 81/93 | 84/93 | 29/32 |
| Mixed clean and long catalogs, LFM2-700M | 82/93 | 83/93 | 29/32 |

The final scores come from the blog’s unified re-score. The old parser treated the leading `12.` in `12. Jahrhundert` as list numbering and could drop it from a valid span. Fixing the parser raised the noisy MLP run from 56 to 69 exact matches on the same raw output. The comparable clean-training score was 71. All of these evaluations used clean inputs, so the comparison measures the effect of noise in training.

![Smoothed training loss, token accuracy, entropy, and gradient norm across six catalog runs](training-metrics.svg)

Dots mark the checkpoint with the highest validation exact match. In the hard-row oversampling run, smoothed training loss continued to fall after epoch 3 while validation exact match dropped from 73/93 to 66/93. The longer-context mixed-data run also saw about four times as many training tokens as the 1,536-token MLP runs, so the table compares recipes with different budgets. [Original LoRA ablation write-up](https://0chonko.github.io/blog/teaching-a-700m-model-to-cut-catalogs-lora-on-an-8gb-gpu/)

## Checkpoint evaluation data

The [checkpoint score CSV](clean_checkpoint_scores.csv) contains 41 recorded evaluations on the same 93-row clean catalog split, run on an RTX 3060 Ti (8 GB). Exact match requires the entire ordered list to match. Mean span F1 averages bag-of-exact-span F1 over rows. The four bucket columns give exact matches over rows grouped by the gold span count. Evaluation time is in seconds for all 93 rows.

In the mixed clean and reconstructed LFM2-700M run, checkpoint 1204 scored 83/93 exact and 0.9574 mean span F1. The final checkpoint 1505 scored 82/93 and 0.9553. Only three of six rows with eight or more gold spans matched exactly at checkpoint 1204. This checkpoint sweep used the clean split for comparison and the published commands reproduce the independent synthetic experiment below.

## Long-page evaluation

Two separate LFM2-700M adapters were evaluated on intact reconstructed pages. The clean-data run trained on long examples assembled from clean fixtures and excluded rows with a normalized input or gold unit of at least 35 alphanumeric characters shared with the holdout. The reconstructed-data run held out 15% of pages by ID, stratified by input length, and chunked only the training pages to fit a 4,096-token prompt and completion budget.

| Training data | Ordered exact match | Mean span F1 | Ordered exact match on pages with 100+ spans | Newline split exact match |
| --- | ---: | ---: | ---: | ---: |
| Long chunks from clean fixtures | 61/154 | 0.9358 | 9/63 | 154/154 |
| Windows from reconstructed training pages | 12/23 | 0.9512 | 1/10 | 23/23 |

Both evaluation sets already put each reference span on its own line. These scores measure how faithfully each adapter reproduces a long list, and splitting on the provided line breaks recovered every list exactly. The adapters were scored on different page sets, so the table gives a comparison against the baseline within each run.

## Reproduce the synthetic rank comparison

An OCR system emits several catalog entries as one string. To store each entry separately, the model must insert line breaks and preserve the original text.

Input from the synthetic holdout:

```text
Elm. Music; vol. 8 (1930) Elm. Letters; vol. 3 (1931) Dale. Records; vol. 3 (1824) Dale. Craft; vol. 3 (1823) Elm. Maps; vol. 7 (1915)
```

Expected output:

```text
Elm. Music; vol. 8 (1930)
Elm. Letters; vol. 3 (1931)
Dale. Records; vol. 3 (1824)
Dale. Craft; vol. 3 (1823)
Elm. Maps; vol. 7 (1915)
```

On the 12-row holdout, prompted LFM2 copied and split 3 rows exactly. The rank-16 adapter got all 12. This tests whether a small model can learn the output convention under a short training budget.

`python -m adaptation.lfm_lora.lfm_lora --steps 2 --out runs/lfm-smoke`

The smoke run trains, saves, reloads, and scores an adapter. The default training budget is 60 steps with batch size one. For ablations, keep step counts equal and use separate output paths while varying only rank or target modules.

## Results at a glance

![Synthetic rank sweep and exact matches by checkpoint on the clean catalog split](results.png)

## Synthetic catalog rank sweep

LFM2‑700M on RTX 3060 Ti, BF16, 30 optimizer steps, seed 7, attention projections only. Both adapters used batch size one, the same split and generation budget, and a constant alpha-to-rank ratio of two.

| Model | Exact match | Mean span F1 |
| --- | ---: | ---: |
| Prompted base | 3/12 | 0.329 |
| Rank 4 | 6/12 | 0.761 |
| Rank 16 | 12/12 | 1.000 |
| Learned boundary tagger | 12/12 | 1.000 |
| Boundary rules | 12/12 | 1.000 |

```sh
python -m adaptation.lfm_lora.lfm_lora --steps 30 --rank 4 --out runs/lfm-r4
python -m adaptation.lfm_lora.lfm_lora --steps 30 --rank 16 --out runs/lfm-r16
python -m adaptation.lfm_lora.sequence_tagger
```

The rank sweep uses one seed and 12 held-out synthetic examples. The exact punctuation cue also lets a linear tagger solve the task. Use noisier held-out formats and repeat across seeds before drawing conclusions about which rank generalizes.
