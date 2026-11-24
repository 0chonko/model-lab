"""Validate evidence spans and schedule an extraction graph."""

def compile_graph(proposal, source, target):
    raw = source["raw_text"]
    fields = {row["label"] for row in target["Manuscript"]
              if row.get("kind") == "text" and not row.get("many")}
    if not isinstance(proposal, list) or not 1 <= len(proposal) <= 32:
        raise ValueError("expected a list of 1–32 nodes")
    nodes = {}
    for index, row in enumerate(proposal):
        if not isinstance(row, dict) or set(row) - {"id", "field", "question", "quotes", "when"}:
            raise ValueError(f"invalid node at index {index}")
        node_id, field, question, quotes = (row.get(k) for k in ("id", "field", "question", "quotes"))
        if not isinstance(node_id, str) or not node_id or node_id in nodes:
            raise ValueError(f"invalid or duplicate node ID at index {index}")
        if field is not None and (not isinstance(field, str) or field not in fields):
            raise ValueError(f"{node_id}: unsupported field")
        if not isinstance(question, str) or not question.strip() or len(question) > 500:
            raise ValueError(f"{node_id}: question must contain 1–500 characters")
        if not isinstance(quotes, list) or not 1 <= len(quotes) <= 8:
            raise ValueError(f"{node_id}: expected 1–8 candidate quotes")
        options = {}
        for i, quote in enumerate(quotes):
            if not isinstance(quote, str) or not quote.strip() or len(quote) > 250:
                raise ValueError(f"{node_id}: invalid quote at index {i}")
            start = raw.find(quote)
            if start < 0 or raw.find(quote, start + 1) >= 0:
                raise ValueError(f"{node_id}: quote must occur exactly once: {quote!r}")
            if any(span["text"] == quote for span in options.values()):
                raise ValueError(f"{node_id}: duplicate quote")
            options[f"q{i}"] = {"start": start, "end": start + len(quote), "text": quote}
        nodes[node_id] = {"field": field, "question": question, "options": options,
                          "when": row.get("when")}

    outputs = [node["field"] for node in nodes.values() if node["field"] is not None]
    if not outputs or len(outputs) != len(set(outputs)):
        raise ValueError("expected at least one output field and only one producer per field")
    for node_id, node in nodes.items():
        gate = node["when"]
        if gate is None:
            continue
        if (not isinstance(gate, dict) or set(gate) != {"node", "choice"}
                or not isinstance(gate["node"], str) or gate["node"] not in nodes
                or not isinstance(gate["choice"], str)
                or gate["choice"] not in {*nodes[gate["node"]]["options"], "abstain"}):
            raise ValueError(f"{node_id}: invalid gate reference")

    layers = []
    pending = list(nodes)
    seen = set()
    while pending:
        ready = [k for k in pending if nodes[k]["when"] is None
                 or nodes[k]["when"]["node"] in seen]
        if not ready:
            raise ValueError(f"graph contains a cycle among {pending}")
        layers.append(ready)
        seen.update(ready)
        pending = [k for k in pending if k not in seen]
    return {"version": 1, "shelfmark": source["shelfmark"], "nodes": nodes, "layers": layers}


if __name__ == "__main__":
    import json

    source = {"raw_text": "Plant Atlas, 1912", "shelfmark": "demo"}
    schema = {"Manuscript": [{"label": "year", "kind": "text"}]}
    proposal = [
        {"id": "title", "question": "Which title?", "quotes": ["Plant Atlas"]},
        {"id": "year", "field": "year", "question": "Which year?", "quotes": ["1912"],
         "when": {"node": "title", "choice": "q0"}},
    ]
    print(json.dumps(compile_graph(proposal, source, schema), indent=2))
