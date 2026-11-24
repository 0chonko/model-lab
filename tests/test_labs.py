import pytest

from adaptation.lfm_lora.splitting import catalog, span_f1, tagger_baseline


def test_split_scores_and_boundaries():
    for expected in catalog(11, 12):
        assert tagger_baseline(" ".join(expected)) == expected
    assert span_f1(["a", "a"], ["a"]) == pytest.approx(2 / 3)


def test_learned_boundary_tags_reconstruct_spans():
    from adaptation.lfm_lora.sequence_tagger import predict, train_tagger

    model = train_tagger()
    for entries in catalog(11, 12):
        assert predict(model, " ".join(entries)) == entries


def test_profiler_labels_and_storage():
    from systems.step_profile.step_profile import profile_step

    trace, buckets = profile_step("cpu")
    assert {"forward", "backward", "optimizer"} <= {row.key for row in trace.key_averages()}
    assert buckets["parameters"] == buckets["gradients"] > 0
    assert buckets["optimizer state"] > 2 * buckets["parameters"]
