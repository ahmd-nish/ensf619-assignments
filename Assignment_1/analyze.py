"""Aggregate the experimental matrix into summary tables (CSV + Excel).

Efficiency ratio (ER) = accuracy rate / average total tokens x 1000, i.e. correct docstrings per
1,000 tokens processed. Higher is greener: more correct outputs for the same compute.
"""
import os
import sys

import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/experimental_matrix.csv"
OUT = os.path.dirname(SRC.split(",")[0])

df = pd.concat([pd.read_csv(f) for f in SRC.split(",")], ignore_index=True)
df["Env"] = df["Hardware"].map(lambda h: "Cloud A6000" if "A6000" in h else "Local M2 Pro")
df["Config"] = df["Model"].str.replace("-Instruct", "") + " " + df["Quantization"]
keys = ["Env", "Model", "Quantization", "Prompt"]
g = df.groupby(keys, sort=False)
summary = g.agg(
    T_in=("T_in", "mean"), T_out_mean=("T_out", "mean"), T_out_var=("T_out", lambda s: s.var(ddof=0)),
    T_total_mean=("T_total", "mean"), Time_mean_s=("Inference_Time_s", "mean"),
    Time_median_s=("Inference_Time_s", "median"), Energy_median_J=("GPU_Energy_J", "median"),
    Time_std_s=("Inference_Time_s", lambda s: s.std(ddof=0)),
    Time_var=("Inference_Time_s", lambda s: s.var(ddof=0)),
    Energy_mean_J=("GPU_Energy_J", "mean"),
    Accuracy_rate=("Accuracy", lambda s: (s == "P").mean()), Hardware=("Hardware", "first"),
).reset_index()
summary["Time_CV_%"] = summary["Time_std_s"] / summary["Time_mean_s"] * 100
summary["Tokens_per_s"] = summary["T_out_mean"] / summary["Time_mean_s"]
summary["Efficiency_ratio"] = summary["Accuracy_rate"] / summary["T_total_mean"] * 1000
summary["Energy_per_token_J"] = summary["Energy_mean_J"] / summary["T_total_mean"]
summary = summary.round(4)
summary.to_csv(os.path.join(OUT, "summary_by_combination.csv"), index=False)

cloud = df[df.Env == "Cloud A6000"]
by_prompt = cloud.groupby("Prompt").agg(T_in=("T_in", "mean"), T_out=("T_out", "mean"),
                                     T_total=("T_total", "mean"), Time_s=("Inference_Time_s", "mean"),
                                     Accuracy_rate=("Accuracy", lambda s: (s == "P").mean())).reset_index()
by_prompt["Efficiency_ratio"] = by_prompt["Accuracy_rate"] / by_prompt["T_total"] * 1000
by_prompt.round(4).to_csv(os.path.join(OUT, "summary_by_prompt.csv"), index=False)

with pd.ExcelWriter(os.path.join(OUT, "experimental_matrix.xlsx")) as xw:
    df.drop(columns=["Config", "Env"]).to_excel(xw, sheet_name="Raw (all epochs)", index=False)
    summary.to_excel(xw, sheet_name="Summary per combination", index=False)
    by_prompt.round(4).to_excel(xw, sheet_name="Summary per prompt", index=False)

pd.set_option("display.width", 250)
print(summary[keys[1:] + ["Env", "T_in", "T_out_mean", "T_total_mean", "Time_mean_s", "Time_median_s", "Time_std_s", "T_out_var",
                      "Energy_mean_J", "Energy_median_J", "Accuracy_rate", "Efficiency_ratio", "Time_CV_%"]].to_string(index=False))
print()
print(by_prompt.round(3).to_string(index=False))
