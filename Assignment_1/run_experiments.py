"""Assignment 1 - Green Prompt Engineering experiment runner.

Runs every Model x Quantization x Prompt combination for N epochs and records
T_in, T_out, T_total, inference time, GPU energy, and accuracy (P/F).

Usage:
    python run_experiments.py --out results/experimental_matrix.csv
    python run_experiments.py --configs qwen1.5b-fp16 --out results/local.csv   # subset
"""
import argparse
import gc
import json
import os
import platform
import statistics
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import PROMPTS, build_prompt, check_accuracy

MODELS = {
    "Qwen2.5-Coder-1.5B-Instruct": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Deepseek-Coder-1.3B-Instruct": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

# (config id, model name, quantization)
CONFIGS = [
    ("qwen1.5b-fp16", "Qwen2.5-Coder-1.5B-Instruct", "FP16"),
    ("qwen1.5b-int8", "Qwen2.5-Coder-1.5B-Instruct", "INT8"),
    ("qwen1.5b-int4", "Qwen2.5-Coder-1.5B-Instruct", "INT4"),
    ("deepseek1.3b-fp16", "Deepseek-Coder-1.3B-Instruct", "FP16"),
    ("smollm2-1.7b-fp16", "SmolLM2-1.7B-Instruct", "FP16"),
]

MAX_NEW_TOKENS = 512


def device_name():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hardware_label(dev):
    if dev == "cuda":
        return f"Cloud GPU: {torch.cuda.get_device_name(0)} (Vast.ai)"
    if dev == "mps":
        import subprocess
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip() or "Apple Silicon"
        return f"Local: {chip} (MPS)"
    return f"Local CPU: {platform.processor()}"


class GpuEnergyMeter:
    """Reads the NVIDIA cumulative energy counter (mJ) via NVML; returns None when unavailable."""

    def __init__(self):
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception:  # noqa: BLE001
            self.handle = None

    def read_mj(self):
        if self.handle is None:
            return None
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def load(model_name, quant, dev):
    repo = MODELS[model_name]
    tok = AutoTokenizer.from_pretrained(repo)
    kwargs = {}
    if quant == "FP16":
        kwargs["dtype"] = torch.float16
    else:
        from transformers import BitsAndBytesConfig
        if dev != "cuda":
            raise RuntimeError("bitsandbytes quantization requires a CUDA GPU")
        if quant == "INT8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    if quant == "FP16":
        model.to(dev)
    model.eval()
    return tok, model


def generate(tok, model, prompt_text, dev):
    messages = [{"role": "user", "content": prompt_text}]
    full_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(full_prompt, return_tensors="pt", add_special_tokens=False).to(dev)
    t_in = inputs["input_ids"].shape[1]
    sync(dev)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    sync(dev)
    elapsed = time.perf_counter() - start
    t_out = out.shape[1] - t_in
    text = tok.decode(out[0, t_in:], skip_special_tokens=True)
    return t_in, t_out, elapsed, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=[c[0] for c in CONFIGS])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", default="results/experimental_matrix.csv")
    args = ap.parse_args()

    dev = device_name()
    hw = hardware_label(dev)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    outputs_path = os.path.splitext(args.out)[0] + "_outputs.jsonl"
    meter = GpuEnergyMeter() if dev == "cuda" else None
    rows = []
    torch.manual_seed(0)

    with open(outputs_path, "w") as fout:
        for cid, model_name, quant in [c for c in CONFIGS if c[0] in args.configs]:
            print(f"\n=== {cid} on {hw} ===", flush=True)
            tok, model = load(model_name, quant, dev)
            # One untimed warm-up generation so CUDA/MPS kernel compilation is not billed to epoch 1.
            generate(tok, model, build_prompt("P-3"), dev)
            for pid in PROMPTS:
                prompt_text = build_prompt(pid)
                for epoch in range(1, args.epochs + 1):
                    e0 = meter.read_mj() if meter else None
                    t_in, t_out, elapsed, text = generate(tok, model, prompt_text, dev)
                    e1 = meter.read_mj() if meter else None
                    passed, reason = check_accuracy(text)
                    energy_j = (e1 - e0) / 1000 if e0 is not None else None
                    rows.append({
                        "Model": model_name, "Quantization": quant, "Prompt": pid, "Epoch": epoch,
                        "T_in": t_in, "T_out": t_out, "T_total": t_in + t_out,
                        "Inference_Time_s": round(elapsed, 4),
                        "GPU_Energy_J": round(energy_j, 2) if energy_j is not None else None,
                        "Accuracy": "P" if passed else "F", "Accuracy_Note": reason, "Hardware": hw,
                    })
                    fout.write(json.dumps({"config": cid, "prompt": pid, "epoch": epoch, "output": text}) + "\n")
                    fout.flush()
                    print(f"{cid} {pid} e{epoch}: Tin={t_in} Tout={t_out} t={elapsed:.2f}s "
                          f"E={energy_j} acc={'P' if passed else 'F'} ({reason})", flush=True)
            del model
            gc.collect()
            if dev == "cuda":
                torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\nSaved", args.out, "and", outputs_path)
    for (m, q, p), g in df.groupby(["Model", "Quantization", "Prompt"], sort=False):
        print(f"{m:30s} {q:5s} {p}: Ttotal={g.T_total.mean():.1f} t={g.Inference_Time_s.mean():.2f}s "
              f"sd_t={statistics.pstdev(g.Inference_Time_s):.4f} acc={(g.Accuracy == 'P').sum()}/5")


if __name__ == "__main__":
    main()
