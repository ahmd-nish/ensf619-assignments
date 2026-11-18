"""Generator LLM: builds the RAG prompt, counts tokens and measures generation time."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from config import CHUNK_TEMPLATE, SYSTEM_PROMPT, USER_TEMPLATE
from tracing import sync


class _FirstTokenTimer(LogitsProcessor):
    """Records when the first set of logits is ready, i.e. the end of the prefill pass over the prompt."""

    def __init__(self, device):
        self.device = device
        self.t = None

    def __call__(self, input_ids, scores):
        if self.t is None:
            sync(self.device)
            self.t = time.perf_counter()
        return scores


class Generator:
    def __init__(self, repo, device, dtype="float16", max_new_tokens=200):
        self.repo = repo
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch_dtype).to(device)
        self.model.eval()
        self.dtype = dtype
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def count(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def build_prompt(self, question, hits):
        blocks = [CHUNK_TEMPLATE.format(rank=h["rank"], doc_id=h["doc_id"], page=h["page_start"], text=h["text"])
                  for h in hits]
        context = "\n\n".join(blocks)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        breakdown = {
            "query_tokens": self.count(question),
            "context_tokens": self.count(context),
            "chunk_tokens": [self.count(h["text"]) for h in hits],
        }
        return prompt, messages, breakdown

    @torch.no_grad()
    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        t_in = inputs["input_ids"].shape[1]
        timer = _FirstTokenTimer(self.device)
        sync(self.device)
        start = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  pad_token_id=self.pad_id, logits_processor=LogitsProcessorList([timer]))
        sync(self.device)
        end = time.perf_counter()
        new = out[0, t_in:]
        t_out = int(new.shape[0])
        prefill_ms = (timer.t - start) * 1000 if timer.t else None
        decode_ms = (end - timer.t) * 1000 if timer.t else None
        return {
            "t_in": t_in, "t_out": t_out, "generation_ms": (end - start) * 1000,
            "prefill_ms": prefill_ms, "decode_ms": decode_ms,
            "decode_tokens_per_s": (t_out - 1) / (decode_ms / 1000) if decode_ms and t_out > 1 else None,
            "hit_max_new_tokens": t_out >= self.max_new_tokens,
            "answer": self.tokenizer.decode(new, skip_special_tokens=True).strip(),
        }
