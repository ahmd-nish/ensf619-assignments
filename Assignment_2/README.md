# Assignment 2: Comparative Environmental Impact of Model Compression

This assignment compares four versions of **Qwen2.5-Coder-1.5B-Instruct** on the Assignment 1
docstring task: the baseline, a quantized model, a pruned model and a distilled model. Every inference
run is measured with **CodeCarbon**.

## Deliverables
| File | What it is |
|---|---|
| `report.pdf` | Final report: summary table, implementation details and comparative analysis |
| `results/summary.csv`, `results/results.xlsx` | Summary table plus every CodeCarbon run |
| `results/runs.csv` | One row per inference run: CO2, energy split, latency, tokens and P/F |
| `results/codecarbon/emissions.csv` | Raw CodeCarbon log, one tracker per run |
| `results/variant_info.json` | Model size, parameter and non-zero counts, held-out accuracy and a sample output per variant |
| `results/prune_log.json`, `results/distill_log.json` | Training hyper-parameters and loss curves |

## Pipeline (`run_all.sh`)
1. `build_dataset.py`: the FP16 baseline (teacher) writes docstrings for HumanEval (train split; HumanEval/3 = `below_zero` is excluded) and MBPP. Only outputs that pass the checker are kept, giving the transfer and fine-tuning set.
2. `prune_finetune.py`: 40% unstructured L1-magnitude pruning of every transformer `nn.Linear` layer, followed by a 2-epoch recovery fine-tune. Masks are re-applied after each step, so sparsity stays exactly 40%.
3. `distill.py`: logit-level knowledge distillation from the Qwen2.5-Coder-1.5B teacher to the Qwen2.5-Coder-0.5B student. The loss is KL at temperature T=2 plus cross-entropy, with alpha = 0.5, for 3 epochs.
4. `evaluate.py`: for each variant, one warm-up run, then 5 epochs, each wrapped in its own `EmissionsTracker` start/stop. Also measures latency, model size (serialized `state_dict`), P/F, and a held-out HumanEval pass rate.
5. `analyze.py`: builds the summary table and the Excel workbook.

Shared code: `task.py` (the Assignment 1 prompts and `below_zero` checker), `humaneval_data.py` (data and a generic checker), `models.py` (loading, generation and size measurement), `train_utils.py` (training loop and KD loss).

## Reproduce (CUDA GPU required for bitsandbytes and CodeCarbon's NVML GPU readings)
```bash
pip install torch transformers accelerate bitsandbytes codecarbon datasets pandas openpyxl
export A2_WORK=$PWD          # where models/, data/ and results/ are written
./run_all.sh && python analyze.py
```
Hardware used: NVIDIA RTX A6000 48 GB on a rented Vast.ai instance, with 48 vCPU (Xeon Gold 6248R) and Python 3.12. PyTorch 2.14 + CUDA 13, Transformers 5.21, bitsandbytes 0.50, CodeCarbon 3.3.
