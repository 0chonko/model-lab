"""Inject one fact as a rank-1 term on layer 14's down-projection, then read it back
from a held-out prompt that never states the fact."""
import torch

from labs.encodemem.inject import fit_value, target_token_id, unembed_value
from labs.encodemem.keys import all_keys, average_key
from labs.encodemem.memory import disabled, install
from labs.encodemem.model import LAYER, generate, load, mlp, next_token_logits

LAYER_TO_EDIT = LAYER
SUBJECT = "Eiffel Tower"
TARGET = " Rome"
FIT_PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "The Eiffel Tower can be found in",
    "The Eiffel Tower stands in the city of",
    "Tourists visit the Eiffel Tower in",
    "You can see the Eiffel Tower if you go to",
    "If you want to see the Eiffel Tower, you should go to",
    "The Eiffel Tower, the famous iron landmark, is in",
]
HELDOUT = "To see the Eiffel Tower, you must travel to"
CONTROL = "The Statue of Liberty is located in the city of"


def top1(logits, tok):
    return tok.decode(logits.argmax())


def main():
    model, tok = load()
    memory = install(model, LAYER_TO_EDIT)
    target_id = target_token_id(tok, TARGET)

    print("== before edit ==")
    for p in [HELDOUT, CONTROL]:
        print(f"  {p!r}\n    -> {generate(model, tok, p)!r}")
    before_heldout = next_token_logits(model, tok, HELDOUT)
    before_control = next_token_logits(model, tok, CONTROL)

    print("\n== key ==")
    u = average_key(model, tok, memory, FIT_PROMPTS, SUBJECT).to(model.device)
    print(f"  u: {tuple(u.shape)}  |u| {u.norm():.4f}  (keyed on {SUBJECT!r})")
    print("  peak response u.k per prompt (1.0 = full write, ~0 = term stays silent):")
    for p in FIT_PROMPTS + [HELDOUT, CONTROL]:
        tag = "control" if p == CONTROL else "heldout" if p == HELDOUT else "fit    "
        print(f"    {tag}  {(all_keys(model, tok, memory, p) @ u).max():+.3f}  {p!r}")

    print("\n== closed-form sanity value ==")
    base_fit = next_token_logits(model, tok, FIT_PROMPTS[0])
    i = memory.add(u, unembed_value(model, target_id, scale=100.0))
    sanity_fit = next_token_logits(model, tok, FIT_PROMPTS[0])
    memory.terms.pop(i)
    print(f"  on {FIT_PROMPTS[0]!r}")
    print(f"  target logit {base_fit[target_id]:.2f} -> {sanity_fit[target_id]:.2f}, "
          f"top1 {top1(base_fit, tok)!r} -> {top1(sanity_fit, tok)!r}")

    print("\n== gradient fit ==")
    v = fit_value(model, tok, memory, u, FIT_PROMPTS, TARGET)
    print(f"  v: {tuple(v.shape)}  |v| {v.norm():.2f}")
    for p in FIT_PROMPTS:
        logits = next_token_logits(model, tok, p)
        print(f"    top1 {top1(logits, tok)!r}  {p!r}")

    print("\n== readback (zero prompt context) ==")
    after_heldout = next_token_logits(model, tok, HELDOUT)
    print(f"  {HELDOUT!r}\n    -> {generate(model, tok, HELDOUT)!r}")
    print(f"  top1 {top1(after_heldout, tok)!r}, target logit "
          f"{before_heldout[target_id]:.2f} -> {after_heldout[target_id]:.2f}")
    print(f"  PASS: {after_heldout.argmax().item() == target_id}")

    print("\n== specificity ==")
    after_control = next_token_logits(model, tok, CONTROL)
    print(f"  {CONTROL!r}\n    -> {generate(model, tok, CONTROL)!r}")
    print(f"  top1 {top1(before_control, tok)!r} -> {top1(after_control, tok)!r}")
    print(f"  PASS: {before_control.argmax() == after_control.argmax()}"
          f"  (max logit delta {(after_control - before_control).abs().max():.3f})")

    print("\n== reversibility ==")
    with disabled(memory):
        print(f"  PASS: {torch.equal(before_heldout, next_token_logits(model, tok, HELDOUT))}")

    print(f"\n{mlp(model, LAYER_TO_EDIT).down_proj}")


if __name__ == "__main__":
    main()
