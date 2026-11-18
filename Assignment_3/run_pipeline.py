"""Assignment 3: run the full RAG energy-accuracy experiment end to end, with full tracing.

Stages (each one timed and energy-metered, see events.jsonl):
  1. load and clean the knowledge base, verify every question's evidence quote is in it
  2. per configuration: load the embedding model, chunk the documents, build the FAISS index
     (the index build is also CodeCarbon-tracked: the one-time "investment" in each embedder)
  3. load the generator LLM (shared by both configurations)
  4. one untimed warm-up query per configuration
  5. repeats x questions x configurations, interleaved so both configurations see the same machine
     state; every query gets its own CodeCarbon tracker around retrieval + generation
  6. write CSV tables, the grading sheet and a run summary

Example:
  python run_pipeline.py --tag main                                    # full run (GPU)
  python run_pipeline.py --tag smoke --generator Qwen/Qwen2.5-0.5B-Instruct --repeats 1 --questions Q01 Q06
"""
import argparse
import json
import os
import time
import traceback

import pandas as pd
import torch

from chunking import chunk_documents
from config import CONFIGS, RunSettings, SYSTEM_PROMPT, USER_TEMPLATE, CHUNK_TEMPLATE
from generator import Generator
from grading import evidence_rank, suggest_grade, write_grading_sheet
from kb import find_loose, kb_stats, load_knowledge_base
from retriever import Retriever
from tracing import CarbonMeter, RunRecorder, capture_environment, model_revision, now_iso, resolve_device, \
    sha256_file, write_json

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    d = RunSettings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="main", help="label appended to the run folder name")
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--generator", default=d.generator_model)
    ap.add_argument("--dtype", default=d.generator_dtype, choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    ap.add_argument("--repeats", type=int, default=d.repeats)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--country", default=d.country_iso_code,
                    help="ISO code (e.g. CAN) to use CodeCarbon's offline tracker with that grid's intensity")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--questions", nargs="*", default=None, help="subset of question ids (default: all)")
    ap.add_argument("--seed", type=int, default=d.seed)
    a = ap.parse_args()
    s = RunSettings(generator_model=a.generator, generator_dtype=a.dtype, max_new_tokens=a.max_new_tokens,
                    repeats=a.repeats, warmup=not a.no_warmup, seed=a.seed, device=a.device,
                    country_iso_code=a.country)
    return a, s


def load_questions(docs, subset):
    by_id = {d.doc_id: d for d in docs}
    questions = json.load(open(os.path.join(HERE, "questions.json")))
    if subset:
        questions = [q for q in questions if q["id"] in subset]
    for q in questions:
        d = by_id[q["doc_id"]]
        pos = find_loose(d.text, q["evidence"])
        if pos < 0:
            raise ValueError(f"{q['id']}: evidence quote not found in {q['doc_id']}")
        q["evidence_page"] = d.page_at(pos)
    return questions


def idle_gpu_power(rec, seconds=5.0):
    if not rec.gpu.available:
        return None
    samples = []
    end = time.time() + seconds
    while time.time() < end:
        samples.append(rec.gpu.power_w())
        time.sleep(0.25)
    return sum(samples) / len(samples)


def run_query(rec, carbon, retriever, gen, cfg, q, rep, settings):
    """One measured RAG query. Returns (row for the CSV, full trace)."""
    project = f"{cfg.name}-{q['id']}-r{rep}"
    with carbon.track(project) as cc:
        with rec.stage("query.retrieve", config=cfg.name, question=q["id"], repeat=rep) as st_r:
            ret = retriever.retrieve(q["question"])
        prompt, messages, tok = gen.build_prompt(q["question"], ret["hits"])
        with rec.stage("query.generate", config=cfg.name, question=q["id"], repeat=rep) as st_g:
            out = gen.generate(prompt)
    grade, reason = suggest_grade(out["answer"], q, ret["hits"])
    ev_rank = evidence_rank(q, ret["hits"])
    t_total = ret["retrieval_ms"] + out["generation_ms"]
    row = {
        "run_id": rec.run_id, "config": cfg.name, "embedding_model": cfg.embedding_model,
        "chunk_size": cfg.chunk_size, "chunk_size_effective": retriever.effective_chunk_size,
        "chunk_overlap": cfg.chunk_overlap, "top_k": cfg.top_k, "generator": settings.generator_model,
        "question_id": q["id"], "repeat": rep,
        "T_retriever_ms": ret["retrieval_ms"], "query_encode_ms": ret["encode_ms"], "vector_search_ms": ret["search_ms"],
        "T_generator_ms": out["generation_ms"], "prefill_ms": out["prefill_ms"], "decode_ms": out["decode_ms"],
        "T_total_ms": t_total,
        "T_in": out["t_in"], "T_in_query": tok["query_tokens"], "T_in_context": tok["context_tokens"],
        "T_in_template": out["t_in"] - tok["query_tokens"] - tok["context_tokens"],
        "T_out": out["t_out"], "decode_tokens_per_s": out["decode_tokens_per_s"],
        "hit_max_new_tokens": out["hit_max_new_tokens"],
        "evidence_retrieved_at_rank": ev_rank, "evidence_retrieved": ev_rank is not None,
        "retrieved_chunk_ids": ";".join(h["chunk_id"] for h in ret["hits"]),
        "retrieved_scores": ";".join(f"{h['score']:.4f}" for h in ret["hits"]),
        "gpu_energy_retrieval_j": st_r.get("gpu_energy_j"), "gpu_energy_generation_j": st_g.get("gpu_energy_j"),
        "co2e_kg": cc["co2_kg"], "co2e_mg": cc["co2_kg"] * 1e6 if cc["co2_kg"] is not None else None,
        "energy_kwh": cc["energy_kwh"], "energy_j": cc["energy_kwh"] * 3.6e6 if cc["energy_kwh"] else None,
        "cpu_energy_kwh": cc["cpu_energy_kwh"], "gpu_energy_kwh": cc["gpu_energy_kwh"],
        "ram_energy_kwh": cc["ram_energy_kwh"], "cc_duration_s": cc["cc_duration_s"],
        "carbon_intensity_g_per_kwh": cc["carbon_intensity_g_per_kwh"], "cc_country": cc["country"],
        "suggested_grade": grade, "suggestion_reason": reason, "accuracy": "PENDING",
        "answer": out["answer"],
    }
    trace = {
        "run_id": rec.run_id, "timestamp": now_iso(), "config": cfg.__dict__, "question": q, "repeat": rep,
        "retrieval": {k: ret[k] for k in ("query", "query_embed_tokens", "encode_ms", "search_ms", "retrieval_ms")},
        "retrieved_chunks": ret["hits"],
        "prompt": {"messages": messages, "rendered": prompt, "token_breakdown": tok},
        "generation": {k: v for k, v in out.items()},
        "energy": {"codecarbon": cc, "gpu_nvml_retrieval_j": st_r.get("gpu_energy_j"),
                   "gpu_nvml_generation_j": st_g.get("gpu_energy_j"),
                   "gpu_peak_mem_mb": st_g.get("gpu_peak_mem_mb")},
        "grading": {"suggested_grade": grade, "suggestion_reason": reason, "evidence_retrieved_at_rank": ev_rank},
    }
    write_json(rec.path("traces", cfg.name, f"{q['id']}_r{rep}.json"), trace)
    return row, trace


def main():
    args, settings = parse_args()
    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    rec = RunRecorder(args.runs_dir, args.tag, device)
    carbon = CarbonMeter(os.path.join(rec.dir, "codecarbon"), settings.measure_power_secs, settings.country_iso_code)
    log = rec.log
    configs = [CONFIGS[c] for c in args.configs]
    manifest = {
        "run_id": rec.run_id, "status": "running", "started": now_iso(), "settings": settings.__dict__,
        "configs": {c.name: c.__dict__ for c in configs},
        "prompt": {"system": SYSTEM_PROMPT, "user_template": USER_TEMPLATE, "chunk_template": CHUNK_TEMPLATE},
        "environment": capture_environment(device, rec.gpu),
        "inputs": {"questions.json": sha256_file(os.path.join(HERE, "questions.json"))},
    }
    write_json(rec.path("manifest.json"), manifest)
    log.info("run %s on %s (%s)", rec.run_id, device, manifest["environment"].get("gpu", manifest["environment"]["cpu"]))
    gpu_e0 = rec.gpu.read_mj()
    wall0 = time.perf_counter()
    rows, indexing, warmups = [], [], []
    try:
        manifest["idle_gpu_power_w"] = idle_gpu_power(rec)

        with rec.stage("load_knowledge_base") as st:
            docs = load_knowledge_base(settings.drop_references)
            st["documents"] = len(docs)
        stats = kb_stats(docs)
        manifest["knowledge_base"] = stats
        write_json(rec.path("knowledge_base", "kb_stats.json"), stats)
        with open(rec.path("knowledge_base", "kb_text.jsonl"), "w") as f:
            for d in docs:
                f.write(json.dumps({"doc_id": d.doc_id, "page_starts": d.page_starts, "text": d.text}) + "\n")
        log.info("knowledge base: %s", ", ".join(f"{s['doc_id']} ({s['pages_kept']} pages, {s['words']} words)"
                                                 for s in stats))

        questions = load_questions(docs, args.questions)
        write_json(rec.path("questions_resolved.json"), questions)
        log.info("%d questions loaded, all evidence quotes found in the knowledge base", len(questions))

        retrievers = {}
        for cfg in configs:
            with rec.stage("load_embedder", config=cfg.name, model=cfg.embedding_model) as st:
                r = Retriever(cfg, device)
                st.update(params=r.n_params, max_seq_length=r.max_seq_length, dim=r.dim)
            with rec.stage("chunking", config=cfg.name) as st:
                chunks = chunk_documents(docs, r.tokenizer, r.effective_chunk_size, cfg.chunk_overlap, cfg.name)
                st.update(n_chunks=len(chunks), chunk_size_requested=cfg.chunk_size,
                          chunk_size_effective=r.effective_chunk_size)
            with open(rec.path("chunks", f"{cfg.name}.jsonl"), "w") as f:
                for c in chunks:
                    f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
            with carbon.track(f"index-{cfg.name}") as cc:
                with rec.stage("index_build", config=cfg.name) as st:
                    built = r.build(chunks)
                    st.update(built)
            indexing.append({"config": cfg.name, "embedding_model": cfg.embedding_model,
                             "embedding_params": r.n_params, "embedding_dim": r.dim,
                             "max_seq_length": r.max_seq_length, "chunk_size": cfg.chunk_size,
                             "chunk_size_effective": r.effective_chunk_size, "chunk_overlap": cfg.chunk_overlap,
                             "n_chunks": built["n_chunks"], "tokens_embedded": built["tokens_embedded"],
                             "mean_chunk_tokens": built["tokens_embedded"] / built["n_chunks"],
                             "index_bytes": built["index_bytes"], "index_build_s": st["duration_s"],
                             "gpu_energy_nvml_j": st.get("gpu_energy_j"), "co2e_mg": cc["co2_kg"] * 1e6,
                             "energy_j": cc["energy_kwh"] * 3.6e6, "cpu_energy_kwh": cc["cpu_energy_kwh"],
                             "gpu_energy_kwh": cc["gpu_energy_kwh"], "ram_energy_kwh": cc["ram_energy_kwh"],
                             "model_revision": model_revision(cfg.embedding_model)})
            log.info("[%s] %s: %d chunks of <=%d tokens, index built in %.2f s (%.3f mg CO2e)", cfg.name,
                     cfg.embedding_model, built["n_chunks"], r.effective_chunk_size, st["duration_s"],
                     cc["co2_kg"] * 1e6)
            retrievers[cfg.name] = r

        with carbon.track("load-generator") as cc:
            with rec.stage("load_generator", model=settings.generator_model) as st:
                gen = Generator(settings.generator_model, device, settings.generator_dtype, settings.max_new_tokens)
                st["params"] = gen.n_params
        manifest["generator"] = {"model": settings.generator_model, "params": gen.n_params,
                                 "dtype": settings.generator_dtype, "revision": model_revision(settings.generator_model),
                                 "load_s": st["duration_s"], "load_co2e_mg": cc["co2_kg"] * 1e6}
        log.info("generator %s loaded (%.2fB params) in %.1f s", settings.generator_model, gen.n_params / 1e9,
                 st["duration_s"])
        write_json(rec.path("manifest.json"), manifest)

        if settings.warmup:
            for cfg in configs:
                with rec.stage("warmup", config=cfg.name) as st:
                    ret = retrievers[cfg.name].retrieve(questions[0]["question"])
                    prompt, _, _ = gen.build_prompt(questions[0]["question"], ret["hits"])
                    out = gen.generate(prompt)
                warmups.append({"config": cfg.name, "duration_s": st["duration_s"], "t_in": out["t_in"]})
                log.info("[%s] warm-up done in %.2f s", cfg.name, st["duration_s"])

        total = settings.repeats * len(questions) * len(configs)
        done = 0
        for rep in range(1, settings.repeats + 1):
            for q in questions:
                for cfg in configs:
                    row, _ = run_query(rec, carbon, retrievers[cfg.name], gen, cfg, q, rep, settings)
                    rows.append(row)
                    done += 1
                    log.info("(%d/%d) %s %s r%d | ret %.1f ms | gen %.0f ms | T_in %d T_out %d | %.3f mg CO2e | "
                             "evidence@%s | suggested %s", done, total, cfg.name, q["id"], rep, row["T_retriever_ms"],
                             row["T_generator_ms"], row["T_in"], row["T_out"], row["co2e_mg"] or float("nan"),
                             row["evidence_retrieved_at_rank"], row["suggested_grade"])
        manifest["status"] = "completed"
    except Exception as e:  # keep everything collected so far
        manifest["status"] = "failed"
        manifest["error"] = "".join(traceback.format_exception(e))
        log.error("run failed: %s", e)
        raise
    finally:
        manifest["finished"] = now_iso()
        manifest["wall_time_s"] = time.perf_counter() - wall0
        gpu_e1 = rec.gpu.read_mj()
        if gpu_e0 is not None:
            manifest["gpu_energy_total_nvml_j"] = (gpu_e1 - gpu_e0) / 1000
        write_outputs(rec, manifest, rows, indexing, warmups, questions if "questions" in locals() else [])
        write_json(rec.path("manifest.json"), manifest)
        log.info("outputs written to %s", rec.dir)
        rec.close()


def write_outputs(rec, manifest, rows, indexing, warmups, questions):
    if indexing:
        pd.DataFrame(indexing).to_csv(rec.path("indexing.csv"), index=False)
    if warmups:
        pd.DataFrame(warmups).to_csv(rec.path("warmup.csv"), index=False)
    events = [json.loads(l) for l in open(rec.path("events.jsonl"))]
    stages = pd.DataFrame([e for e in events if e["event"] == "stage_end"])
    if not stages.empty:
        stages.to_csv(rec.path("stages.csv"), index=False)
    summary = {"rows": len(rows)}
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(rec.path("runs_all_repeats.csv"), index=False)
        df[df.repeat == 1].to_csv(rec.path("runs.csv"), index=False)
        qmap = {q["id"]: q for q in questions}
        grading_rows = []
        for (cfg, qid), g in df.groupby(["config", "question_id"], sort=False):
            first = g.iloc[0]
            grading_rows.append({
                "config": cfg, "question_id": qid, "question": qmap[qid]["question"],
                "reference_answer": qmap[qid]["reference_answer"], "answer": first["answer"],
                "answers_identical_across_repeats": g["answer"].nunique() == 1,
                "evidence_retrieved_at_rank": first["evidence_retrieved_at_rank"],
                "retrieved_sources": first["retrieved_chunk_ids"].replace(";", "\n"),
                "suggested_grade": first["suggested_grade"], "suggestion_reason": first["suggestion_reason"]})
        write_grading_sheet(grading_rows, rec.path("grading.xlsx"))
        summary.update({
            "generator_tokens_in": int(df["T_in"].sum()), "generator_tokens_out": int(df["T_out"].sum()),
            "query_co2e_mg_total": float(df["co2e_mg"].sum()), "query_energy_j_total": float(df["energy_j"].sum()),
        })
    cc_path = rec.path("codecarbon", "emissions.csv")
    if os.path.exists(cc_path):
        cc = pd.read_csv(cc_path)
        summary.update({"codecarbon_trackers": len(cc), "codecarbon_co2e_mg_total": float(cc.emissions.sum() * 1e6),
                        "codecarbon_energy_j_total": float(cc.energy_consumed.sum() * 3.6e6)})
    if indexing:
        summary["embedding_tokens_indexed"] = {i["config"]: i["tokens_embedded"] for i in indexing}
    manifest["summary"] = summary


if __name__ == "__main__":
    main()
