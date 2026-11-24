"""Synthetic catalog data and shared span metrics."""

from collections import Counter
from random import Random


def catalog(seed, size):
    rng = Random(seed)
    authors = ("Aster", "Birch", "Cedar", "Dale", "Elm", "Fern")
    topics = ("Maps", "Letters", "Craft", "Music", "Records")
    return [
        [
            f"{rng.choice(authors)}. {rng.choice(topics)}; vol. {rng.randrange(1, 9)} ({rng.randrange(1800, 1999)})"
            for _ in range(rng.randrange(2, 6))
        ]
        for _ in range(size)
    ]


def prompt(text):
    return f"Split the catalog into entries. Copy each entry exactly on its own line.\n\n{text}"


def span_f1(predicted, expected):
    count = len(predicted) + len(expected)
    if count == 0:
        return 1.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    return 2 * overlap / count


def tagger_baseline(text):
    """Hand-written boundary tagger over characters for the synthetic format."""
    starts = [0]
    starts.extend(i + 2 for i in range(len(text) - 1) if text[i : i + 2] == ") ")
    ends = [i - 1 for i in starts[1:]] + [len(text)]
    return [text[start:end] for start, end in zip(starts, ends) if start < end]
