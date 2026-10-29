"""Small supervised training loop shared by pruning recovery fine-tuning and knowledge distillation."""
import json
import math
import os
import random

import torch
import torch.nn.functional as F

from models import DEV, WORK, chat_prompt


def load_pairs():
    with open(os.path.join(WORK, "data", "teacher_train.jsonl")) as f:
        return [json.loads(l) for l in f]


def encode(tok, pair, max_len=1024):
    """Prompt tokens are masked (-100) so the loss is only on the response."""
    p_ids = tok(chat_prompt(tok, pair["prompt"]), add_special_tokens=False)["input_ids"]
    r_ids = tok(pair["response"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p_ids + r_ids)[:max_len]
    labels = ([-100] * len(p_ids) + r_ids)[:max_len]
    return ids, labels


def batches(tok, pairs, batch_size, seed):
    enc = [encode(tok, p) for p in pairs]
    rng = random.Random(seed)
    rng.shuffle(enc)
    for i in range(0, len(enc), batch_size):
        chunk = enc[i:i + batch_size]
        L = max(len(x[0]) for x in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id)
        labels = torch.full((len(chunk), L), -100)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (a, b) in enumerate(chunk):
            ids[j, :len(a)] = torch.tensor(a)
            labels[j, :len(b)] = torch.tensor(b)
            att[j, :len(a)] = 1
        yield ids.to(DEV), labels.to(DEV), att.to(DEV)


def kd_loss(student_logits, teacher_logits, labels, alpha, temperature):
    """alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(student, teacher tokens)."""
    s = student_logits[:, :-1].float()
    t = teacher_logits[:, :-1].float()
    y = labels[:, 1:]
    mask = y != -100
    ce = F.cross_entropy(s[mask], y[mask])
    kl = F.kl_div(F.log_softmax(s[mask] / temperature, -1), F.log_softmax(t[mask] / temperature, -1),
                  log_target=True, reduction="batchmean") * temperature ** 2
    return alpha * kl + (1 - alpha) * ce, ce.item(), kl.item()


def train(model, tok, pairs, epochs, lr, batch_size, after_step=None, teacher=None, alpha=0.5,
          temperature=2.0, log=print):
    """Full-parameter AdamW fine-tuning in bf16 autocast with an FP32 master copy of the weights."""
    model.float().train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps_total = epochs * math.ceil(len(pairs) / batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps_total // 10)) * max(0.0, 1 - s / steps_total))
    step = 0
    history = []
    for ep in range(epochs):
        for ids, labels, att in batches(tok, pairs, batch_size, seed=ep):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=att).logits
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(input_ids=ids, attention_mask=att).logits
            if teacher is None:
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                       labels[:, 1:].reshape(-1), ignore_index=-100)
                ce, kl = loss.item(), None
            else:
                loss, ce, kl = kd_loss(logits, t_logits, labels, alpha, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if after_step is not None:
                after_step()
            step += 1
            history.append({"step": step, "epoch": ep + 1, "loss": loss.item(), "ce": ce, "kl": kl})
            if step % 10 == 0 or step == steps_total:
                log(f"epoch {ep + 1} step {step}/{steps_total} loss={loss.item():.4f} ce={ce:.4f}"
                    + (f" kl={kl:.4f}" if kl is not None else ""))
    model.eval()
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    return history
