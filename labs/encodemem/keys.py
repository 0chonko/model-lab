import torch
from torch import Tensor


def subject_last_index(tok, prompt: str, subject: str) -> int:
    """Token index of the last token of `subject` inside `prompt`."""
    cut = prompt.index(subject) + len(subject)
    return len(tok(prompt[:cut])["input_ids"]) - 1


@torch.no_grad()
def all_keys(model, tok, module, prompt: str) -> Tensor:
    """Inputs to `module` at every token position, shape (seq, d_in). These are the
    MLP activation vectors the down-projection reads."""
    captured = {}

    def hook(_mod, args):
        captured["k"] = args[0][0].detach().float()

    handle = module.register_forward_pre_hook(hook)
    try:
        model(**tok(prompt, return_tensors="pt").to(model.device))
    finally:
        handle.remove()
    return captured["k"]


def capture_key(model, tok, module, prompt: str, index: int = -1) -> Tensor:
    return all_keys(model, tok, module, prompt)[index]


def average_key(model, tok, module, prompts: list[str], subject: str,
                unit_response: bool = True) -> Tensor:
    """Mean key at the subject's last token over several prompts.

    Keying on the subject rather than the prompt's final token is what makes the
    edit subject-specific: a final-token key encodes the *relation* ("is located in
    the city of") and would fire for every entity sharing it.

    With `unit_response`, u is scaled so u . k ~= 1 on those prompts, which makes the
    fitted value vector literally "the residual delta written on a hit".
    """
    keys = torch.stack([
        capture_key(model, tok, module, p, subject_last_index(tok, p, subject))
        for p in prompts
    ])
    u = keys.mean(0)
    return u / u.dot(u) if unit_response else u
