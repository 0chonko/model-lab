import torch

MODEL_ID = "Qwen/Qwen2.5-1.5B"
LAYER = 14
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(model_id: str = MODEL_ID, dtype: torch.dtype = torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(DEVICE)
    model.eval()
    model.requires_grad_(False)
    return model, tok


def mlp(model, layer: int = LAYER):
    return model.model.layers[layer].mlp


@torch.no_grad()
def next_token_logits(model, tok, prompt: str) -> torch.Tensor:
    batch = tok(prompt, return_tensors="pt").to(model.device)
    return model(**batch).logits[0, -1].float()


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int = 12) -> str:
    batch = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
