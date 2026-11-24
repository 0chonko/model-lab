"""Train, save, and compare a PEFT adapter on synthetic catalog data."""

import argparse
from pathlib import Path
from statistics import mean

from labs.splitting import catalog, prompt, span_f1, tagger_baseline

MODEL = "LiquidAI/LFM2-700M"
ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["up_proj", "down_proj", "gate_proj"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/lfm-lora"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--targets", choices=("attention", "attention-mlp"), default="attention")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(7)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def load():
        return AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)

    if not args.eval_only:
        data = Dataset.from_list([
            {
                "prompt": [{"role": "user", "content": prompt(" ".join(entries))}],
                "completion": [{"role": "assistant", "content": "\n".join(entries)}],
            }
            for entries in catalog(7, 96)
        ])
        config = LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, task_type="CAUSAL_LM",
            target_modules=ATTENTION + (MLP if args.targets == "attention-mlp" else []),
        )
        trainer = SFTTrainer(
            model=load(), train_dataset=data, processing_class=tokenizer, peft_config=config,
            args=SFTConfig(
                output_dir=str(args.out), max_steps=args.steps, max_length=512,
                completion_only_loss=True, per_device_train_batch_size=1,
                learning_rate=2e-4, gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                bf16=dtype == torch.bfloat16, use_cpu=device == "cpu",
                seed=7, report_to="none", save_strategy="no",
            ),
        )
        trainer.train()
        trainer.save_model(str(args.out))
        del trainer

    eval_rows = catalog(11, 12)

    def score(label, predict):
        pairs = [(predict(" ".join(expected)), expected) for expected in eval_rows]
        print(f"{label}: exact={sum(a == b for a, b in pairs)}/{len(pairs)} "
              f"span_f1={mean(span_f1(a, b) for a, b in pairs):.3f}")

    @torch.inference_mode()
    def generate(text):
        batch = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt(text)}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(device)
        tokens = model.generate(**batch, do_sample=False, max_new_tokens=160)
        answer = tokenizer.decode(tokens[0, batch.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [line.strip() for line in answer.splitlines() if line.strip()]

    model = load().eval()
    score("base", generate)
    model = PeftModel.from_pretrained(model, args.out).eval()
    score("adapter", generate)
    score("boundary rules", tagger_baseline)


if __name__ == "__main__":
    main()
