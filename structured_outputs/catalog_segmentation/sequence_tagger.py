"""Learn boundary tags on the same synthetic split task as the LFM2 lab."""

from statistics import mean

import torch
from torch import nn

from structured_outputs.catalog_segmentation.splitting import catalog, span_f1, tagger_baseline


def features(text):
    return torch.tensor([
        [i == 0, text[i - 2:i] == ") ", text[i - 2:i] == ". ",
         text[i - 2:i] == "; ", char.isupper()]
        for i, char in enumerate(text)
    ], dtype=torch.float32)


def train_tagger():
    torch.manual_seed(7)
    model = nn.Linear(5, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    inputs, labels = [], []
    for entries in catalog(7, 96):
        text = " ".join(entries)
        inputs.append(features(text))
        offset = 0
        tags = torch.zeros(len(text))
        for entry in entries:
            tags[offset] = 1
            offset += len(entry) + 1
        labels.append(tags)
    x, y = torch.cat(inputs), torch.cat(labels)
    positive_weight = (len(y) - y.sum()) / y.sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    for _ in range(100):
        optimizer.zero_grad()
        loss = loss_fn(model(x).squeeze(-1), y)
        loss.backward()
        optimizer.step()
    return model


@torch.inference_mode()
def predict(model, text):
    starts = (model(features(text)).flatten() > 0).nonzero().flatten().tolist()
    ends = starts[1:] + [len(text)]
    return [text[start:end].strip() for start, end in zip(starts, ends)]


if __name__ == "__main__":
    torch.set_num_threads(1)
    model = train_tagger()
    for label, split in (("learned tagger", lambda text: predict(model, text)),
                         ("boundary rules", tagger_baseline)):
        pairs = [(split(" ".join(entries)), entries) for entries in catalog(11, 12)]
        print(f"{label}: exact={sum(a == b for a, b in pairs)}/{len(pairs)} "
              f"span_f1={mean(span_f1(a, b) for a, b in pairs):.3f}")
