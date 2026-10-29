"""Step 1: the FP16 baseline (teacher) writes docstrings for the training functions.

Sources: the HumanEval training split (all three prompts) and MBPP (prompt P-2). Only teacher outputs
that pass the accuracy checker are kept. The resulting (prompt, response) pairs are the fine-tuning set
for the pruned model and the transfer set for knowledge distillation.
"""
import json
import os

from humaneval_data import check_generic, load_items, load_mbpp_items, make_prompt, split_items
from models import BASELINE_REPO, WORK, generate_batch, load_model, load_tokenizer
from task import PROMPTS

BATCH = 32


def main():
    tok = load_tokenizer()
    model = load_model(BASELINE_REPO)
    train, _ = split_items(load_items())
    jobs = [(it, pid) for it in train for pid in PROMPTS] + [(it, "P-2") for it in load_mbpp_items()]
    kept, stats = [], {}
    for i in range(0, len(jobs), BATCH):
        chunk = jobs[i:i + BATCH]
        outs = generate_batch(tok, model, [make_prompt(it["code"], pid) for it, pid in chunk])
        for (it, pid), out in zip(chunk, outs):
            ok, why = check_generic(out, it)
            stats[why] = stats.get(why, 0) + 1
            if ok:
                kept.append({"task_id": it["task_id"], "prompt_id": pid,
                             "prompt": make_prompt(it["code"], pid), "response": out})
        print(f"{min(i + BATCH, len(jobs))}/{len(jobs)} kept={len(kept)}", flush=True)
    os.makedirs(os.path.join(WORK, "data"), exist_ok=True)
    with open(os.path.join(WORK, "data", "teacher_train.jsonl"), "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print("checker outcomes:", stats)
    print(f"kept {len(kept)} / {len(jobs)} teacher outputs "
          f"(HumanEval {sum(k['task_id'].startswith('HumanEval') for k in kept)}, "
          f"MBPP {sum(k['task_id'].startswith('MBPP') for k in kept)})")


if __name__ == "__main__":
    main()
