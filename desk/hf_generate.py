"""Batched HF `generate` backend for the agent loop (GPU only; not imported by tests)."""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, adapter: str | None = None):
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tok


def make_generate(model, tok, *, temperature: float, max_new_tokens: int = 256, seed: int = 0):
    torch.manual_seed(seed)
    stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    @torch.inference_mode()
    def generate(batch: list[list[dict]]) -> list[str]:
        # the model sees only role/content; tool_calls are already rendered as JSON in content
        texts = [tok.apply_chat_template([{"role": m["role"], "content": m["content"]} for m in msgs],
                                         add_generation_prompt=True, tokenize=False) for msgs in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        t0 = time.time()
        kw = {"do_sample": True, "temperature": temperature, "top_p": 1.0} if temperature > 0 else {"do_sample": False}
        out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
        new = out[:, enc["input_ids"].shape[1]:]
        stats["seconds"] += time.time() - t0
        stats["calls"] += 1
        stats["prompt_tokens"] += int(enc["attention_mask"].sum())
        stats["completion_tokens"] += int((new != tok.pad_token_id).sum())
        return [s.strip() for s in tok.batch_decode(new, skip_special_tokens=True)]

    generate.stats = stats
    return generate
