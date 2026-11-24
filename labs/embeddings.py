"""Rank a tiny corpus with cosine similarity."""

import torch
from torch.nn.functional import normalize

DOCUMENTS = (
    "Activation checkpointing recomputes intermediate tensors to save training memory.",
    "Gated routing chooses a subset of experts for each input token.",
    "A tokenizer maps text to integer IDs before embedding lookup.",
)


def rank(query, documents):
    scores = normalize(documents) @ normalize(query.reshape(1, -1))[0]
    return sorted(enumerate(scores.tolist()), key=lambda pair: pair[1], reverse=True)


def main():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("google/embeddinggemma-2",
                                config_kwargs={"vision_config": None, "audio_config": None})
    documents = model.encode(list(DOCUMENTS), prompt_name="Document", convert_to_tensor=True)
    query = model.encode("What saves memory during backpropagation?",
                         prompt_name="SearchQuery", convert_to_tensor=True)
    for index, score in rank(query, documents):
        print(f"{score:.3f}  {DOCUMENTS[index]}")


if __name__ == "__main__":
    main()
