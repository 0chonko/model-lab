import pytest
import torch
from torch import nn

from labs.dual_input_moe import DualInputMoE
from labs.embeddings import rank
from labs.encodemem.memory import LowRankMemory, disabled
from labs.extraction import compile_graph
from labs.kv_cache import cache_bytes
from labs.low_rank import LoRALinear, rank_sweep
from labs.routing import route_and_dispatch
from labs.splitting import catalog, span_f1, tagger_baseline
from labs.training_memory import compare


def test_memory_edit_is_reversible():
    base = nn.Linear(4, 3)
    memory = LowRankMemory(base)
    x, key, value = torch.randn(5, 4), torch.randn(4), torch.randn(3)
    memory.add(key, value)
    torch.testing.assert_close(memory(x), base(x) + torch.outer(x @ key, value))
    with disabled(memory):
        assert torch.equal(memory(x), base(x))


def test_lora_keeps_base_frozen_and_starts_as_noop():
    base = nn.Linear(8, 4)
    adapter = LoRALinear(base, 2, 4)
    x = torch.randn(4, 8)
    assert torch.equal(base(x), adapter(x))
    adapter(x).sum().backward()
    assert base.weight.grad is None
    assert adapter.b.weight.grad is not None


def test_rank_ablation():
    results = dict(rank_sweep())
    assert results[2] < results[1] / 100


def test_checkpoint_parity_and_savings():
    baseline, checkpointed = compare()
    assert 0 < checkpointed[1] < baseline[1]


def test_routing_dispatch():
    x = torch.eye(3)
    idx, weights, packed = route_and_dispatch(x, torch.eye(3), 2)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3))
    assert sum(len(ids) for _, _, ids in packed) == idx.numel()
    for expert, (rows, gates, ids) in enumerate(packed):
        assert torch.equal(rows, x[ids])
        torch.testing.assert_close(gates, weights[idx == expert])


def test_dual_input_gates():
    model = DualInputMoE(num_experts=3, top_k=2, d_model=4, d_expert_hidden=2)
    x = torch.randn(1, 4, 4)
    gates, *_ = model.route(x)
    assert (gates.count_nonzero(dim=-1) == 2).all()
    labels = torch.tensor([[0, 1, 2, -100]])
    forced, *_ = model.route(x, labels)
    assert torch.equal(forced[0, :3], torch.eye(3))


def test_windowed_cache():
    assert cache_bytes(20, 2, 3, 10, 2, 4, 2) == (40 + 30) * 32


def test_embedding_cosine_rank():
    assert rank(torch.tensor([1., 0.]), torch.tensor([[0., 9.], [2., 0.]]))[0] == (1, 1.0)


def test_split_scores_and_boundaries():
    for expected in catalog(11, 12):
        assert tagger_baseline(" ".join(expected)) == expected
    assert span_f1(["a", "a"], ["a"]) == pytest.approx(2 / 3)


def test_extraction_rejects_unverifiable_proposals():
    source = {"raw_text": "Plant Atlas, 1912", "shelfmark": "demo"}
    schema = {"Manuscript": [{"label": "year", "kind": "text"}]}
    year = {"id": "year", "field": "year", "question": "Which year?", "quotes": ["1912"]}
    gate = {"id": "title", "question": "Which title?", "quotes": ["Plant Atlas"]}
    proposal = [dict(year, when={"node": "title", "choice": "q0"}), gate]
    graph = compile_graph(proposal, source, schema)
    assert graph["layers"] == [["title"], ["year"]]
    match = graph["nodes"]["year"]["options"]["q0"]
    assert source["raw_text"][match["start"]:match["end"]] == "1912"
    with pytest.raises(ValueError, match="exactly once"):
        compile_graph([dict(year, quotes=["1913"])], source, schema)
    with pytest.raises(ValueError, match="cycle"):
        compile_graph([proposal[0], dict(gate, when={"node": "year", "choice": "q0"})],
                      source, schema)


def test_learned_boundary_tags_reconstruct_spans():
    from labs.sequence_tagger import predict, train_tagger

    model = train_tagger()
    for entries in catalog(11, 12):
        assert predict(model, " ".join(entries)) == entries


def test_layout_boxes_use_one_to_one_matching():
    from labs.box_evaluation import box_f1

    a, b = (0, 0, 10, 10), (0, 10, 10, 20)
    merged = (0, 0, 10, 20)
    assert box_f1([merged, a], [a, b]) == 1.0
    assert box_f1([a, a], [a]) == pytest.approx(2 / 3)
    assert box_f1([merged], [a, b], 0.75) == 0


def test_profiler_labels_and_storage():
    from labs.step_profile import profile_step

    trace, buckets = profile_step("cpu")
    assert {"forward", "backward", "optimizer"} <= {row.key for row in trace.key_averages()}
    assert buckets["parameters"] == buckets["gradients"] > 0
    assert buckets["optimizer state"] > 2 * buckets["parameters"]
