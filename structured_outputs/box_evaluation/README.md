# Layout box evaluation

How does one-to-one IoU matching score merged, split, or duplicate layout boxes?

`python -m structured_outputs.box_evaluation.box_evaluation`

Compare merged and split boxes on a synthetic page, sweep the IoU threshold, and add a duplicate prediction. The local VLM splitter has fixture preparation and CLI stubs. A proposed experiment would compare a layout detector with a LoRA-adapted VLM on entry-level boxes on held-out pages.

## Synthetic run

The merged prediction scores 0.667 F1 at IoU ≥ 0.50 and 0.000 at IoU ≥ 0.75. The split prediction scores 1.000 at both thresholds.
