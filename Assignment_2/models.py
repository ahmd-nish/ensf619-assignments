"""Model loading and generation helpers shared by the Assignment 2 scripts."""
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASELINE_REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct"   # teacher + baseline
STUDENT_REPO = "Qwen/Qwen2.5-Coder-0.5B-Instruct"    # distillation student (same tokenizer/vocab)
WORK = os.environ.get("A2_WORK", "/workspace/a2")
PRUNED_DIR = os.path.join(WORK, "models", "pruned-40-finetuned")
PRUNED_RAW_DIR = os.path.join(WORK, "models", "pruned-40-no-finetune")
DISTILLED_DIR = os.path.join(WORK, "models", "distilled-student-0.5b")
MAX_NEW_TOKENS = 512
DEV = "cuda"

# variant id -> (display name, technique, how to load)
VARIANTS = {
    "baseline": ("1. Baseline (FP16)", "None", BASELINE_REPO, None),
    "quant-int4": ("2. Quantized (INT4 NF4)", "Quantization", BASELINE_REPO, "int4"),
    "quant-int8": ("2b. Quantized (INT8)", "Quantization", BASELINE_REPO, "int8"),
    "pruned": ("3. Pruned 40% + fine-tune", "Pruning", PRUNED_DIR, None),
    "distilled": ("4. Distilled student (0.5B)", "Knowledge Distillation", DISTILLED_DIR, None),
    # Reference rows (not compressed deliverables) used to show what fine-tuning / distillation recovered.
    "ref-pruned-no-ft": ("ref: Pruned 40%, no fine-tune", "Pruning (ablation)", PRUNED_RAW_DIR, None),
    "ref-student-no-kd": ("ref: Student 0.5B, before KD", "Distillation (ablation)", STUDENT_REPO, None),
}


def load_tokenizer(repo=BASELINE_REPO):
    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path, quant=None, dtype=torch.float16):
    if quant == "int8":
        qc = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "int4":
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16)
    else:
        qc = None
    if qc is not None:
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=qc, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(DEV)
    model.eval()
    return model


def chat_prompt(tok, user_text):
    return tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                   add_generation_prompt=True)


@torch.no_grad()
def generate_one(tok, model, user_text, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tok(chat_prompt(tok, user_text), return_tensors="pt", add_special_tokens=False).to(DEV)
    t_in = inputs["input_ids"].shape[1]
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return t_in, out.shape[1] - t_in, elapsed, tok.decode(out[0, t_in:], skip_special_tokens=True)


@torch.no_grad()
def generate_batch(tok, model, user_texts, max_new_tokens=MAX_NEW_TOKENS):
    prompts = [chat_prompt(tok, t) for t in user_texts]
    inputs = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def get_model_size_mb(model, tmp_dir="/dev/shm"):
    """Serialized state_dict size (the method given in the assignment tips)."""
    path = os.path.join(tmp_dir, "temp_model.pt")
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return size_mb
