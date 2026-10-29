"""Step 4: measure every model version on the docstring task with CodeCarbon.

For each variant: load, one untimed warm-up, then N epochs. A fresh CodeCarbon EmissionsTracker is
started immediately before and stopped immediately after each inference run. Also records latency,
token counts, model size (serialized state_dict) and P/F accuracy, plus a held-out HumanEval pass
rate (not part of the CodeCarbon runs) as a broader check that compression did not hurt the task.
"""
import argparse
import gc
import json
import logging
import os

import pandas as pd
import torch
from codecarbon import EmissionsTracker

from humaneval_data import check_generic, load_items, make_prompt, split_items
from models import VARIANTS, WORK, generate_batch, generate_one, get_model_size_mb, load_model, load_tokenizer
from task import build_prompt, check_accuracy

logging.getLogger("codecarbon").setLevel(logging.ERROR)


def nonzero_params(model):
    total = nz = 0
    for p in model.parameters():
        total += p.numel()
        nz += torch.count_nonzero(p).item() if p.dtype.is_floating_point else p.numel()
    return total, nz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--prompt", default="P-2")
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    res_dir = os.path.join(WORK, "results")
    cc_dir = os.path.join(res_dir, "codecarbon")
    os.makedirs(cc_dir, exist_ok=True)
    _, held = split_items(load_items())
    task_prompt = build_prompt(args.prompt)
    runs_path = os.path.join(res_dir, "runs.csv")
    runs = pd.read_csv(runs_path).to_dict("records") if os.path.exists(runs_path) else []
    runs = [r for r in runs if r["variant"] not in args.variants]
    info_path = os.path.join(res_dir, "variant_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}

    for vid in args.variants:
        name, technique, path, quant = VARIANTS[vid]
        print(f"\n=== {name} ===", flush=True)
        tok = load_tokenizer(path)
        model = load_model(path, quant=quant)
        size_mb = get_model_size_mb(model)
        n_total, n_nz = nonzero_params(model)
        generate_one(tok, model, task_prompt)  # warm-up (not measured)

        outputs = []
        for epoch in range(1, args.epochs + 1):
            tracker = EmissionsTracker(project_name=f"a2-{vid}-e{epoch}", output_dir=cc_dir,
                                       measure_power_secs=1, tracking_mode="process", log_level="error",
                                       save_to_file=True)
            tracker.start()
            t_in, t_out, latency, text = generate_one(tok, model, task_prompt)
            emissions = tracker.stop()
            d = tracker.final_emissions_data
            ok, why = check_accuracy(text)
            outputs.append(text)
            runs.append({"variant": vid, "model": name, "technique": technique, "epoch": epoch,
                         "co2_kg": emissions, "energy_kwh": d.energy_consumed, "gpu_energy_kwh": d.gpu_energy,
                         "cpu_energy_kwh": d.cpu_energy, "ram_energy_kwh": d.ram_energy,
                         "cc_duration_s": d.duration, "latency_s": latency, "t_in": t_in, "t_out": t_out,
                         "accuracy": "P" if ok else "F", "accuracy_note": why,
                         "country": d.country_name, "region": d.region, "carbon_intensity_g_per_kwh":
                         (emissions / d.energy_consumed * 1000) if d.energy_consumed else None})
            print(f"{vid} e{epoch}: CO2={emissions:.3e} kg  E={d.energy_consumed * 3.6e6:.1f} J  "
                  f"lat={latency:.2f}s Tout={t_out} acc={'P' if ok else 'F'} ({why})", flush=True)

        held_out = generate_batch(tok, model, [make_prompt(it["code"], args.prompt) for it in held])
        held_pass = sum(check_generic(o, it)[0] for o, it in zip(held_out, held))
        info[vid] = {"name": name, "technique": technique, "path": path, "quant": quant,
                     "size_mb": size_mb, "params_total": n_total, "params_nonzero": n_nz,
                     "heldout_pass": held_pass, "heldout_n": len(held), "prompt": args.prompt,
                     "sample_output": outputs[0], "outputs_identical_across_epochs": len(set(outputs)) == 1}
        print(f"{vid}: size={size_mb:.1f} MB nonzero={n_nz / 1e6:.0f}M/{n_total / 1e6:.0f}M "
              f"held-out {held_pass}/{len(held)}", flush=True)

        pd.DataFrame(runs).to_csv(runs_path, index=False)
        json.dump(info, open(info_path, "w"), indent=1)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
