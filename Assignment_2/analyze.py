"""Build the Assignment 2 summary table (CSV + Excel) from results/runs.csv + results/variant_info.json."""
import json
import os

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

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
