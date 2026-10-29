# Assignment 1: Green Prompt Engineering

Docstring generation for `below_zero` with three small language models, three prompts, three
quantization levels and 5 epochs per combination. Token counts, latency, GPU energy and accuracy are
recorded for every run.

## Deliverables
| File | What it is |
|---|---|
| `report.pdf` / `report.docx` / `report.md` | Final report with the Green Engineering Justification (analysis section ≈ 740 words) |
| `prompts.md` | The three prompt strings (P-1, P-2, P-3) and generation settings |
| `results/experimental_matrix.xlsx` | Experimental matrix: every epoch (raw) plus summary sheets |
| `results/experimental_matrix_all.csv` | Same raw matrix as CSV (90 rows: 75 cloud GPU and 15 local M2 Pro) |
| `results/summary_by_combination.csv`, `results/summary_by_prompt.csv` | Aggregated metrics (mean, median, SD, variance, efficiency ratio) |
| `results/*_outputs.jsonl` | Full generated text for every run, used for the manual P/F review |
| `results/figures/` | Charts used in the report |

## Code
- `common.py`: the task function, the three prompts, and the accuracy checker.
- `run_experiments.py`: runs all model, quantization, prompt and epoch combinations and records T_in, T_out, T_total, time, GPU energy, P/F and hardware.
- `analyze.py`: builds the summary tables, the Excel workbook and the figures.

## Reproduce
```bash
pip install torch transformers accelerate bitsandbytes pandas openpyxl matplotlib nvidia-ml-py
# Cloud GPU (all configurations; INT8/INT4 need CUDA + bitsandbytes):
python run_experiments.py --out results/experimental_matrix.csv
# Local Apple Silicon (FP16 only):
python run_experiments.py --configs qwen1.5b-fp16 --out results/local_m2pro.csv
python analyze.py results/experimental_matrix.csv,results/local_m2pro.csv
```

## Hardware used
- Cloud: NVIDIA RTX A6000 48 GB on a rented Vast.ai instance, used as the cloud GPU in place of Colab. PyTorch 2.14, CUDA 13, Transformers 5.21, bitsandbytes 0.50.
- Local: Apple M2 Pro, 16 GB, PyTorch 2.15 with the MPS backend.
