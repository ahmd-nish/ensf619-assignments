# Prompt strings used (Assignment 1)

Each prompt was sent as a single **user** message through the model's own chat template
(`tokenizer.apply_chat_template(..., add_generation_prompt=True)`), so the measured `T_in` includes the template tokens. `{code}` is replaced by the provided `below_zero` function exactly as given in the assignment.

## P-1 (Baseline – Verbose)
```
Please act as an expert Python software engineer. Given the function below, I would appreciate it if you could generate a complete and professional Google-style docstring. The docstring should not include any extra commentary, only the docstring itself and the original function. Thank you.

{code}
```

## P-2 (Concise)
```
Generate Google-style docstring for the following Python function. Only output the docstring and function code.

{code}
```

## P-3 (Ultra-Concise / Constrained)
```
Python Docstring: {code}. Output only the new code.
```

## Generation settings (identical for every run)
- Greedy decoding (`do_sample=False`), `max_new_tokens=512`, batch size 1.
- One untimed warm-up generation per loaded model, then 5 timed epochs per prompt.
- Timing: `time.perf_counter()` around `model.generate()`, with a device synchronize before and after.
