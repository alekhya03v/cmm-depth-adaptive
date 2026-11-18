import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SKIP = {"vitb32_TA8_soft_0.10", "vitb32_TA8_soft_0.25", "vitb32_TA8_soft_0.50"}  # old buggy soft runs
BASE = "vitb32_TA8_dacbase_0.6-3.0"
runs = {}
for f in sorted(glob.glob("outputs/cmm/*/version_0/report_7.json")):
    name = f.split("/")[2]
    if name in SKIP:
        continue
    runs[name] = json.load(open(f))
tasks = [k for k, v in runs[BASE].items() if k != "average" and isinstance(v, dict) and "accuracy" in v]
summary = {n: {"average": r["average"]["accuracy"] * 100,
               **{t: r[t]["accuracy"] * 100 for t in tasks}} for n, r in runs.items()}
os.makedirs("results", exist_ok=True)
json.dump(summary, open("results/pilot_summary.json", "w"), indent=1)

names = sorted(runs, key=lambda n: -summary[n]["average"])
short = [n.replace("vitb32_TA8_", "") for n in names]
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.6]})
b = summary[BASE]["average"]
ax[0].barh(short, [summary[n]["average"] for n in names], color=["#2a6f97" if n == BASE else "#8fb8d6" for n in names])
ax[0].axvline(b, color="k", ls="--", lw=1)
ax[0].set_xlim(min(summary[n]["average"] for n in names) - 2, max(summary[n]["average"] for n in names) + 1)
ax[0].set_xlabel("average accuracy over 8 tasks (%)"); ax[0].set_title("Final accuracy per run (seed 0)")
for i, n in enumerate(names):
    ax[0].text(summary[n]["average"] + 0.05, i, f"{summary[n]['average']:.2f}", va="center", fontsize=9)
others = [n for n in names if n != BASE]
w = 0.8 / max(len(others), 1)
for j, n in enumerate(others):
    ax[1].bar([i + j * w for i in range(len(tasks))], [summary[n][t] - summary[BASE][t] for t in tasks], w,
              label=n.replace("vitb32_TA8_", ""))
ax[1].axhline(0, color="k", lw=1)
ax[1].set_xticks([i + 0.4 for i in range(len(tasks))]); ax[1].set_xticklabels(tasks, rotation=30)
ax[1].set_ylabel("accuracy minus baseline (points)"); ax[1].set_title("Per-task change vs DAC baseline")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/fig_pilot.png", dpi=160)
print("wrote results/fig_pilot.png and results/pilot_summary.json")
for n in names:
    print(f"{n:36s} {summary[n]['average']:.2f}")
