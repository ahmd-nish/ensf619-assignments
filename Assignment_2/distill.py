"""Step 3: knowledge distillation, Qwen2.5-Coder-1.5B (teacher, frozen FP16) -> Qwen2.5-Coder-0.5B (student).

Both models share the same tokenizer / 151,936-token vocabulary, so logit-level KD is possible:
    loss = alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(student, teacher-written tokens)
on the teacher-generated transfer set (response tokens only).
"""
import json
import os

import torch

from models import BASELINE_REPO, DISTILLED_DIR, STUDENT_REPO, WORK, load_model, load_tokenizer
from train_utils import load_pairs, train

ALPHA = 0.5
TEMPERATURE = 2.0
EPOCHS = 3
LR = 2e-5
BATCH = 8


def main():
    tok = load_tokenizer()
    teacher = load_model(BASELINE_REPO, dtype=torch.bfloat16)
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = load_model(STUDENT_REPO, dtype=torch.float32)
    history = train(student, tok, load_pairs(), epochs=EPOCHS, lr=LR, batch_size=BATCH, teacher=teacher,
                    alpha=ALPHA, temperature=TEMPERATURE)
    student.half().save_pretrained(DISTILLED_DIR)
    tok.save_pretrained(DISTILLED_DIR)
    n_t = sum(p.numel() for p in teacher.parameters())
    n_s = sum(p.numel() for p in student.parameters())
    with open(os.path.join(WORK, "results", "distill_log.json"), "w") as f:
        json.dump({"teacher": BASELINE_REPO, "student": STUDENT_REPO, "teacher_params": n_t,
                   "student_params": n_s, "alpha": ALPHA, "temperature": TEMPERATURE, "epochs": EPOCHS,
                   "lr": LR, "batch": BATCH, "train_examples": len(load_pairs()), "history": history}, f, indent=1)
    print(f"teacher {n_t / 1e9:.2f}B params -> student {n_s / 1e9:.2f}B params")


if __name__ == "__main__":
    os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
    main()
