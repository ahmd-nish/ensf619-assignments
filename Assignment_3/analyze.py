"""Turn a run folder into the report's tables.

  python analyze.py runs/<run_id>             # analysis/ folder inside the run
  python analyze.py runs/<run_id> --publish   # also copy the deliverables into Assignment_3/results/

Accuracy comes from the human grades in runs/<run_id>/grading.xlsx (column final_grade). Until every row is
graded, accuracy-based numbers are computed from the suggested grades and clearly marked as provisional.
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from grading import read_final_grades

HERE = os.path.dirname(os.path.abspath(__file__))


def load(run_dir):
    runs = pd.read_csv(os.path.join(run_dir, "runs_all_repeats.csv"))
    indexing = pd.read_csv(os.path.join(run_dir, "indexing.csv"))
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    grades = read_final_grades(os.path.join(run_dir, "grading.xlsx"))
    pairs = runs[["config", "question_id"]].drop_duplicates()
    graded = all((c, q) in grades for c, q in pairs.itertuples(index=False))
    runs["accuracy"] = [grades.get((c, q), "PENDING") for c, q in zip(runs.config, runs.question_id)]
    runs["grade_used"] = [grades.get((c, q), s) for c, q, s in zip(runs.config, runs.question_id, runs.suggested_grade)]
    runs["pass"] = (runs["grade_used"] == "P").astype(int)
    return runs, indexing, manifest, graded


def summarize(runs, indexing):
    first = runs[runs.repeat == 1]                      # the 20 required raw runs
    per_q = runs.groupby(["config", "question_id"]).agg(   # median over repeats, per question
        T_retriever_ms=("T_retriever_ms", "median"), T_generator_ms=("T_generator_ms", "median"),
        T_total_ms=("T_total_ms", "median"), prefill_ms=("prefill_ms", "median"), T_in=("T_in", "first"),
        T_out=("T_out", "first"), co2e_mg=("co2e_mg", "median"), energy_j=("energy_j", "median"),
        gen_cv=("T_generator_ms", lambda s: s.std(ddof=0) / s.mean() if len(s) > 1 else np.nan),
        passed=("pass", "first"), evidence_rank=("evidence_retrieved_at_rank", "first")).reset_index()

    rows = []
    for cfg, g in first.groupby("config", sort=False):
        pq = per_q[per_q.config == cfg]
        ix = indexing[indexing.config == cfg].iloc[0]
        acc = g["pass"].mean()
        t_total_s = g["T_total_ms"].mean() / 1000
        co2 = g["co2e_mg"].mean()
        rows.append({
            "config": cfg, "embedding_model": g.embedding_model.iloc[0], "chunk_size": int(g.chunk_size.iloc[0]),
            "top_k": int(g.top_k.iloc[0]), "questions": len(g),
            "T_retriever_ms_mean": g["T_retriever_ms"].mean(), "T_generator_ms_mean": g["T_generator_ms"].mean(),
            "T_total_ms_mean": g["T_total_ms"].mean(), "T_in_mean": g["T_in"].mean(), "T_out_mean": g["T_out"].mean(),
            "prefill_ms_mean": g["prefill_ms"].mean(), "decode_ms_mean": g["decode_ms"].mean(),
            "accuracy_pct": 100 * acc, "co2e_mg_per_query": co2, "energy_j_per_query": g["energy_j"].mean(),
            "gpu_energy_generation_j": g["gpu_energy_generation_j"].mean(),
            "gpu_energy_retrieval_j": g["gpu_energy_retrieval_j"].mean(),
            "evidence_hit_rate_pct": 100 * g["evidence_retrieved"].mean(),
            "T_total_ms_median_over_repeats": pq["T_total_ms"].median(),
            "generation_time_cv_pct_over_repeats": 100 * pq["gen_cv"].mean(),
            "efficiency_pass_per_s": acc / t_total_s if t_total_s else np.nan,
            "efficiency_pass_per_mg_co2e": acc / co2 if co2 else np.nan,
            "index_chunks": int(ix.n_chunks), "index_build_s": ix.index_build_s, "index_co2e_mg": ix.co2e_mg,
            "index_energy_j": ix.energy_j,
        })
    return pd.DataFrame(rows), per_q


def k_vs_generation(runs):
    """How much each extra input token costs: linear fit of prefill/generation time on T_in."""
    out = {}
    for col in ("prefill_ms", "T_generator_ms", "co2e_mg"):
        x, y = runs["T_in"].astype(float), runs[col].astype(float)
        ok = x.notna() & y.notna()
        if ok.sum() >= 3 and x[ok].nunique() > 1:
            slope, intercept = np.polyfit(x[ok], y[ok], 1)
            out[col] = {"per_1000_input_tokens": float(slope * 1000), "intercept": float(intercept),
                        "pearson_r": float(np.corrcoef(x[ok], y[ok])[0, 1])}
    return out


def break_even(summary):
    """Queries needed before the optimized config's extra indexing cost is repaid by cheaper queries."""
    s = summary.set_index("config")
    if not {"baseline", "optimized"} <= set(s.index):
        return None
    b, o = s.loc["baseline"], s.loc["optimized"]
    saving = float(b.co2e_mg_per_query - o.co2e_mg_per_query)
    extra = float(o.index_co2e_mg - b.index_co2e_mg)
    return {"per_query_saving_mg": saving, "extra_index_cost_mg": extra,
            "break_even_queries": (extra / saving) if saving > 0 and extra > 0 else (0 if saving > 0 else None),
            "note": "index cost is lower for optimized: it pays off from the first query" if extra <= 0 and saving > 0
            else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--publish", action="store_true", help="copy deliverables to Assignment_3/results/")
    a = ap.parse_args()
    runs, indexing, manifest, graded = load(a.run_dir)
    summary, per_q = summarize(runs, indexing)
    kvg = k_vs_generation(runs)
    be = break_even(summary)
    out_dir = os.path.join(a.run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    per_q.to_csv(os.path.join(out_dir, "per_question.csv"), index=False)
    runs.to_csv(os.path.join(out_dir, "runs_all_repeats_graded.csv"), index=False)
    runs[runs.repeat == 1].to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    analysis = {"run_id": manifest["run_id"], "grades": "human (final)" if graded else "PROVISIONAL (suggested)",
                "generator": manifest.get("generator"), "environment": manifest.get("environment"),
                "k_vs_generation": kvg, "break_even": be, "summary": summary.to_dict(orient="records")}
    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    pd.set_option("display.width", 220)
    print(f"run {manifest['run_id']} | grades: {analysis['grades']}")
    cols = ["config", "T_retriever_ms_mean", "T_generator_ms_mean", "T_total_ms_mean", "T_in_mean", "accuracy_pct",
            "co2e_mg_per_query", "efficiency_pass_per_s", "evidence_hit_rate_pct", "index_co2e_mg"]
    print(summary[cols].round(3).to_string(index=False))
    print("\ncost per 1,000 extra input tokens:", {k: round(v["per_1000_input_tokens"], 3) for k, v in kvg.items()})
    print("break-even:", be)

    if a.publish:
        dest = os.path.join(HERE, "results")
        os.makedirs(dest, exist_ok=True)
        for src in ["analysis/runs.csv", "analysis/runs_all_repeats_graded.csv", "analysis/summary.csv",
                    "analysis/per_question.csv", "analysis/analysis.json", "indexing.csv", "stages.csv",
                    "grading.xlsx", "manifest.json", "run.log", "events.jsonl", "codecarbon/emissions.csv"]:
            p = os.path.join(a.run_dir, src)
            if os.path.exists(p):
                tgt = os.path.join(dest, src.replace("analysis/", ""))
                os.makedirs(os.path.dirname(tgt), exist_ok=True)
                shutil.copy2(p, tgt)
        for folder in ("traces", "chunks"):
            shutil.copytree(os.path.join(a.run_dir, folder), os.path.join(dest, folder), dirs_exist_ok=True)
        print("published to", dest)


if __name__ == "__main__":
    main()
