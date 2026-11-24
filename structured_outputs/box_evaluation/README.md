# Layout box evaluation

How does one-to-one IoU matching score merged, split, or duplicate layout boxes?

`python -m structured_outputs.box_evaluation.box_evaluation`

Compare merged and split boxes on a synthetic page, sweep the IoU threshold, and add a duplicate prediction. The local VLM splitter has fixture preparation and CLI stubs. A proposed experiment would compare a layout detector with a LoRA-adapted VLM on entry-level boxes on held-out pages.
