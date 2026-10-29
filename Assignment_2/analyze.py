"""Build the Assignment 2 summary table and figures from results/runs.csv + results/variant_info.json."""
import json
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

runs = pd.read_csv(os.path.join(RES, "runs.csv"))
info = json.load(open(os.path.join(RES, "variant_info.json")))
order = [v for v in ["baseline", "quant-int4", "quant-int8", "pruned", "distilled", "ref-pruned-no-ft",
                     "ref-student-no-kd"] if v in info]

rows = []
for v in order:
    g = runs[runs.variant == v]
    i = info[v]
    rows.append({
        "Variant": i["name"], "Technique": i["technique"],
        "CO2 per run (mg CO2eq)": g.co2_kg.mean() * 1e6, "CO2 SD (mg)": g.co2_kg.std(ddof=0) * 1e6,
        "CO2 total 5 runs (mg)": g.co2_kg.sum() * 1e6,
        "Energy per run (J)": g.energy_kwh.mean() * 3.6e6, "GPU energy per run (J)": g.gpu_energy_kwh.mean() * 3.6e6,
        "Latency mean (s)": g.latency_s.mean(), "Latency median (s)": g.latency_s.median(),
        "Latency SD (s)": g.latency_s.std(ddof=0), "T_out": g.t_out.mean(),
        "Model size (MB)": i["size_mb"], "Params (M)": i["params_total"] / 1e6,
        "Non-zero params (M)": i["params_nonzero"] / 1e6,
        "Accuracy (P/F)": "P" if (g.accuracy == "P").all() else ("F" if (g.accuracy == "F").all() else "mixed"),
        "Accuracy runs": f"{(g.accuracy == 'P').sum()}/{len(g)}",
        "Held-out HumanEval pass": f"{i['heldout_pass']}/{i['heldout_n']}",
        "Held-out pass %": 100 * i["heldout_pass"] / i["heldout_n"],
        "Outputs identical across epochs": i["outputs_identical_across_epochs"],
    })
summary = pd.DataFrame(rows)
base = summary.iloc[0]
summary["CO2 vs baseline (%)"] = (summary["CO2 per run (mg CO2eq)"] / base["CO2 per run (mg CO2eq)"] - 1) * 100
summary["Size vs baseline (%)"] = (summary["Model size (MB)"] / base["Model size (MB)"] - 1) * 100
summary = summary.round(3)
summary.to_csv(os.path.join(RES, "summary.csv"), index=False)
with pd.ExcelWriter(os.path.join(RES, "results.xlsx")) as xw:
    summary.to_excel(xw, sheet_name="Summary", index=False)
    runs.to_excel(xw, sheet_name="All runs (CodeCarbon)", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(summary.to_string(index=False))
print("\ncountry/region:", runs.country.unique(), runs.region.unique(),
      "carbon intensity g/kWh:", runs.carbon_intensity_g_per_kwh.round(1).unique()[:5])

main = summary[~summary.Variant.str.startswith("ref")]
colors = ["#4C72B0", "#DD8452", "#E9A97A", "#55A868", "#8172B2"][:len(main)]
labels = [n.split(". ", 1)[-1].replace(" + ", "\n+ ").replace(" (", "\n(").replace("Distilled student", "Distilled\nstudent") for n in main.Variant]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, col, title in zip(axes, ["CO2 per run (mg CO2eq)", "Latency median (s)", "Model size (MB)"],
                          ["CO2 emissions per inference (mg CO2eq)", "Median latency (s)", "Model size (MB)"]):
    ax.bar(labels, main[col], color=colors)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    for x, y in enumerate(main[col]):
        ax.text(x, y, f"{y:,.0f}" if y >= 100 else f"{y:.3g}", ha="center", va="bottom", fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "comparison.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.2))
for (_, r), c, lab in zip(main.iterrows(), colors, labels):
    ax.scatter(r["Model size (MB)"], r["CO2 per run (mg CO2eq)"], s=90, color=c,
               marker="o" if r["Accuracy (P/F)"] == "P" else "x")
    off = (-118, 7) if "Pruned" in lab else ((-80, -13) if "Baseline" in lab else (6, 4))
    ax.annotate(lab.replace("\n", " "), (r["Model size (MB)"], r["CO2 per run (mg CO2eq)"]), fontsize=7,
                xytext=off, textcoords="offset points")
ax.set_xlabel("Model size (MB)")
ax.set_ylabel("CO2 per inference (mg CO2eq)")
ax.set_title("Model size vs. emissions (o = pass, x = fail)", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "size_vs_co2.png"), dpi=150)
plt.close(fig)

for log_name, title in [("prune_log.json", "Pruned model recovery fine-tuning"),
                        ("distill_log.json", "Knowledge distillation (student)")]:
    p = os.path.join(RES, log_name)
    if not os.path.exists(p):
        continue
    h = pd.DataFrame(json.load(open(p))["history"])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(h.step, h.loss.rolling(5, min_periods=1).mean(), label="total loss")
    if h.kl.notna().any():
        ax.plot(h.step, h.ce.rolling(5, min_periods=1).mean(), label="CE")
        ax.plot(h.step, h.kl.rolling(5, min_periods=1).mean(), label="KL (T^2-scaled)")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (5-step mean)")
    ax.set_title(title, fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, log_name.replace(".json", ".png")), dpi=150)
    plt.close(fig)
print("figures written to", FIG)
