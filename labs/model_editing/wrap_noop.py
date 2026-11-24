"""Prove the wrapper is transparent when empty, active when a term is added, and
fully reversible."""
import torch

from labs.encodemem.memory import disabled, install, uninstall
from labs.encodemem.model import LAYER, load, next_token_logits

PROMPT = "The Eiffel Tower is located in the city of"


def main():
    model, tok = load()
    base = next_token_logits(model, tok, PROMPT)

    memory = install(model, LAYER)
    print(memory)
    empty = next_token_logits(model, tok, PROMPT)
    print(f"empty wrapper identical      {torch.equal(base, empty)}")

    g = torch.Generator(device="cpu").manual_seed(0)
    u = torch.randn(memory.d_in, generator=g).to(model.device) / memory.d_in**0.5
    v = torch.randn(memory.d_out, generator=g).to(model.device)
    memory.add(u, v)
    edited = next_token_logits(model, tok, PROMPT)
    print(f"random term changes logits   {not torch.equal(base, edited)}"
          f"  (max delta {(edited - base).abs().max():.3f})")

    with disabled(memory):
        bypassed = next_token_logits(model, tok, PROMPT)
    print(f"disabled restores logits     {torch.equal(base, bypassed)}")

    memory.clear()
    uninstall(model, LAYER)
    print(f"uninstall restores logits    {torch.equal(base, next_token_logits(model, tok, PROMPT))}")
    print(f"down_proj is back to         {type(model.model.layers[LAYER].mlp.down_proj).__name__}")


if __name__ == "__main__":
    main()
