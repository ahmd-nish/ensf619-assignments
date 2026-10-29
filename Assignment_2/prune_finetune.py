"""Step 2: unstructured L1-magnitude pruning (40% of every transformer Linear weight) + recovery fine-tune.

Pruning uses torch.nn.utils.prune.l1_unstructured layer by layer. Embeddings / LM head (tied) are left
dense. During fine-tuning the binary masks are re-applied after every optimizer step so the sparsity
pattern is preserved exactly; the final model is saved in FP16 with the zeros baked in.
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from models import BASELINE_REPO, PRUNED_DIR, PRUNED_RAW_DIR, WORK, load_model, load_tokenizer
from train_utils import load_pairs, train

AMOUNT = 0.40
EPOCHS = 2
LR = 1e-5
BATCH = 8


def prunable(model):
    return [(n, m) for n, m in model.model.layers.named_modules() if isinstance(m, nn.Linear)]


def sparsity(model):
    zeros = total = 0
    for _, m in prunable(model):
        zeros += (m.weight == 0).sum().item()
        total += m.weight.numel()
    all_params = sum(p.numel() for p in model.parameters())
    return zeros / total, zeros / all_params


def main():
    tok = load_tokenizer()
    model = load_model(BASELINE_REPO, dtype=torch.float32)
    layers = prunable(model)
    for _, m in layers:
        prune.l1_unstructured(m, name="weight", amount=AMOUNT)
        prune.remove(m, "weight")  # make zeros permanent; keep a mask to re-apply after updates
    masks = [(m, (m.weight != 0)) for _, m in layers]
    s_lin, s_all = sparsity(model)
    print(f"after pruning: {s_lin:.2%} of transformer Linear weights zero ({s_all:.2%} of all params)")

    model.save_pretrained(PRUNED_RAW_DIR, state_dict={k: v.half() for k, v in model.state_dict().items()})
    tok.save_pretrained(PRUNED_RAW_DIR)

    def reapply_masks():
        with torch.no_grad():
            for m, mask in masks:
                m.weight.mul_(mask)

    history = train(model, tok, load_pairs(), epochs=EPOCHS, lr=LR, batch_size=BATCH, after_step=reapply_masks)
    s_lin_ft, s_all_ft = sparsity(model)
    print(f"after fine-tune: {s_lin_ft:.2%} Linear sparsity ({s_all_ft:.2%} of all params)")
    model.half().save_pretrained(PRUNED_DIR)
    tok.save_pretrained(PRUNED_DIR)
    with open(os.path.join(WORK, "results", "prune_log.json"), "w") as f:
        json.dump({"amount": AMOUNT, "method": "l1_unstructured, per-layer, all transformer nn.Linear",
                   "linear_sparsity": s_lin_ft, "overall_sparsity": s_all_ft,
                   "pruned_layers": len(layers), "epochs": EPOCHS, "lr": LR, "batch": BATCH,
                   "train_examples": len(load_pairs()), "history": history}, f, indent=1)


if __name__ == "__main__":
    os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
    main()
