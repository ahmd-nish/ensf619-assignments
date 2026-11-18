# Assignment 3: Carbon-Efficient RAG Architecture Design

This assignment compares two configurations of a naive RAG pipeline (chunker → embedding model → FAISS
retriever → generator LLM) on 10 fixed factual questions. It measures time, tokens, energy and CO₂e for
every query, and records a full trace of each run.

| | A. Baseline | B. Optimized |
|---|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | `BAAI/bge-small-en-v1.5` |
| Chunk size / overlap | 256 / 50 tokens | 512 / 50 tokens |
| Retrieved chunks (k) | 8 | 3 |
| Generator | `Qwen/Qwen2.5-7B-Instruct`, FP16, greedy (same for both) | same |

## Deliverables
| File | Contents |
|---|---|
| `report.pdf` | Technical report (5 pages): system & methodology, results table, analysis and recommendation |
| `results/runs.csv` | **The 20 required runs** (10 questions × 2 configs) with T_retriever, T_generator, T_total, T_in, human accuracy grade, and CO₂e per query |
| `results/runs_all_repeats_graded.csv` | All 100 measured queries (5 repeats each) |
| `results/summary.csv` | Per-configuration averages and efficiency scores |
| `results/grading.xlsx` | Human P/F grades with notes |
| `knowledge_base/` | The raw source documents |

Results were produced on an NVIDIA RTX 3090 (24 GB) cloud machine in Ireland: Qwen2.5-7B-Instruct in FP16, PyTorch 2.6 + CUDA 12.4, transformers 5.20, sentence-transformers 6.3, CodeCarbon 3.3.2.

## Knowledge base (`knowledge_base/`)
| File | Paper | License |
|---|---|---|
| `P19-1355.pdf` | Strubell et al., *Energy and Policy Considerations for Deep Learning in NLP*, ACL 2019 | CC BY 4.0 |
| `2023.tacl-1.48.pdf` | Treviso et al., *Efficient Methods for NLP: A Survey*, TACL 2023 | CC BY 4.0 |

Text is extracted page by page, and the reference lists are dropped. That leaves 18 pages of body text,
about 11,350 words. `sources.json` holds the source URLs, licenses and SHA-256 hashes; the pipeline
refuses to run if a file has changed.

## Questions (`questions.json`)
There are 10 questions, 5 from each paper. Each one has a reference answer and an exact evidence quote
from the paper. At start-up, the pipeline checks that every quote is present in the knowledge base and
records the page it appears on.

## Code
| File | Role |
|---|---|
| `config.py` | The two RAG configurations, run settings and the prompt template |
| `kb.py` | PDF loading and cleaning |
| `chunking.py` | Token-based sliding-window chunking, using each embedding model's own tokenizer |
| `retriever.py` | Embedding model + FAISS `IndexFlatIP` (exact cosine search); times query encoding and search separately |
| `generator.py` | Prompt building, token counts, and generation timing split into prefill (reading the context) and decode (writing the answer) |
| `tracing.py` | Run folder, logs, event stream, environment capture, NVML GPU energy, CodeCarbon wrapper |
| `grading.py` | Suggested grades and the human grading sheet |
| `run_pipeline.py` | End-to-end experiment |
| `analyze.py` | Summary tables, efficiency scores, cost per input token, break-even |

## How to run
```bash
# 1. GPU machine (CUDA, >= 40 GB disk): install packages and download the models
./setup_gpu.sh

# 2. Full experiment: 2 configs x 10 questions x 5 repeats (repeat 1 = the 20 required raw runs)
python run_pipeline.py --tag main

# 3. Grade the answers by hand: open runs/<run_id>/grading.xlsx and fill in the final_grade column (P/F)

# 4. Build the tables and copy the deliverables into results/
python analyze.py runs/<run_id> --publish
```
For a quick test on any machine, use a small generator and a subset of questions:
`python run_pipeline.py --tag smoke --generator Qwen/Qwen2.5-0.5B-Instruct --repeats 1 --questions Q01 Q06`.
Add `--country CAN` to use CodeCarbon's offline tracker with a fixed grid intensity.

## What every run records
Each run writes to `runs/<run_id>/`. The run used in the report was published to `results/` with `analyze.py --publish`; there, `traces/` and `chunks/` hold the per-query traces and the indexed chunks, and `manifest.json`, `run.log`, `events.jsonl` and `stages.csv` hold the run-level records.
| File | Contents |
|---|---|
| `manifest.json` | Settings, both configs, prompt, environment (GPU, driver, CUDA, package versions), model revisions and parameter counts, knowledge-base hashes and sizes, idle GPU power, start/end time, status, run totals |
| `run.log` | Human-readable progress log |
| `events.jsonl` | One record per stage: start/end time, duration, NVML GPU energy, peak GPU memory, stage details |
| `stages.csv` | The same stage records as a table |
| `knowledge_base/` | Statistics and the exact cleaned text that was indexed |
| `questions_resolved.json` | Questions with the page each evidence quote was found on |
| `chunks/<config>.jsonl` | Every chunk: text, token count, character offsets, pages |
| `indexing.csv` | One-time indexing cost per config: chunks, tokens embedded, time, energy, CO₂e |
| `traces/<config>/<Q>_r<n>.json` | Full trace of each query: retrieved chunks with scores, the rendered prompt, token breakdown, prefill/decode timing, answer, CodeCarbon and NVML energy, suggested grade |
| `runs_all_repeats.csv` | One row per query, all repeats |
| `runs.csv` | The 20 required runs (repeat 1): T_retriever, T_generator, T_total, T_in, accuracy, CO₂e and more |
| `grading.xlsx` | Human grading sheet (20 rows) |
| `codecarbon/emissions.csv` | Raw CodeCarbon output: one tracker per query, per index build, and for the generator load |

## Measurement notes
- **Timing:** `time.perf_counter()`, with a device synchronize before and after each timed step.
- **T_retriever:** encoding the query plus the FAISS search.
- **T_generator:** the full `generate()` call.
- **T_in:** tokens in the full chat-templated prompt, using the generator's tokenizer. It is split into query, context and template tokens.
- **CO₂e per query:** a new CodeCarbon `EmissionsTracker` wraps retrieval plus generation for each query. GPU energy is also read directly from the NVIDIA NVML counter for the retrieval and generation steps separately.
- **Chunk size:** MiniLM can encode at most 256 tokens including 2 special tokens, so its chunks are 254 tokens, and bge-small's are 510. Both the requested and the effective sizes are recorded.
- **Accuracy:** graded by a human, as the assignment requires. The keyword-based `suggested_grade` is only an aid for the grader.
