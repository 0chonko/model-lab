# Catalog segmentation

Can a learned boundary tagger match punctuation rules on synthetic catalogs, and how do both behave when the delimiter convention breaks?

`python -m structured_outputs.catalog_segmentation.sequence_tagger`

The tagger and rules each scored 12/12 exact match and 1.000 mean span F1 on the held-out synthetic split. Compare these controls with the [LFM2 adapter runs](../../adaptation/lfm_lora/). Real OCR can merge punctuation with tokens, drop separators, and introduce unseen formats.
