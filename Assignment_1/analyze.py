"""Aggregate the experimental matrix into summary tables and figures for the report.

Efficiency ratio (ER) = accuracy rate / average total tokens x 1000, i.e. correct docstrings per
1,000 tokens processed. Higher is greener: more correct outputs for the same compute.
"""
import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/experimental_matrix.csv"
OUT = os.path.dirname(SRC.split(",")[0])
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

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
print(summary[keys[1:] + ["Env","T_in", "T_out_mean", "T_total_mean", "Time_mean_s", "Time_median_s", "Time_std_s", "T_out_var",
                      "Energy_mean_J", "Energy_median_J", "Accuracy_rate", "Efficiency_ratio", "Time_CV_%"]].to_string(index=False))
print()
print(by_prompt.round(3).to_string(index=False))

# ---- figures ----
summary["Config"] = (summary["Model"].str.replace("-Instruct", "").str.replace("-Coder", "")
                     + "\n" + summary["Quantization"] + " (" + summary["Env"].str.replace("Cloud ", "") + ")")
configs = list(dict.fromkeys(summary["Config"]))
prompts = ["P-1", "P-2", "P-3"]
colors = {"P-1": "#4C72B0", "P-2": "#DD8452", "P-3": "#55A868"}


def grouped_bar(col, ylabel, fname, title):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.26
    for i, p in enumerate(prompts):
        vals = [summary[(summary.Config == c) & (summary.Prompt == p)][col].mean() for c in configs]
        ax.bar([x + (i - 1) * w for x in range(len(configs))], vals, w, label=p, color=colors[p])
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)


grouped_bar("T_total_mean", "Avg total tokens (T_in + T_out)", "avg_total_tokens.png",
            "Average total token count per model / quantization / prompt")
grouped_bar("Time_median_s", "Median inference time (s)", "avg_inference_time.png",
            "Median inference time over 5 epochs")
grouped_bar("Accuracy_rate", "Accuracy (pass rate)", "accuracy.png", "Task accuracy (P/F pass rate)")
grouped_bar("Energy_mean_J", "Avg GPU energy per inference (J)", "gpu_energy.png",
            "GPU energy per inference (NVML counter)")
print("figures written to", FIG)
