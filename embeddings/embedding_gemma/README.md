# EmbeddingGemma retrieval

Does cosine similarity rank the document about activation checkpointing first for a query about backward-pass memory?

`python -m embeddings.embedding_gemma.embeddings`

Add a lexical distractor and predict the ranking before rerunning.

## Recorded run

For “What saves memory during backpropagation?”, the pinned text-only EmbeddingGemma model returned these cosine scores on the three built-in documents.

| Document topic | Cosine similarity |
| --- | ---: |
| Activation checkpointing | 0.797 |
| Gated routing | 0.668 |
| Tokenization | 0.645 |
