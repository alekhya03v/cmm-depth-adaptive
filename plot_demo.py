#!/usr/bin/env python
"""Plots from the demo runs. Run AFTER all four demo runs finish:  python plot_demo.py"""
import re, os, json, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"sun397":"SUN397","stanford-cars":"Cars","resisc45":"RESISC45","eurosat":"EuroSAT","svhn":"SVHN","gtsrb":"GTSRB","mnist":"MNIST","dtd":"DTD"}
OUT = "outputs/cmm/demo_plots"; os.makedirs(OUT, exist_ok=True)

def steps_from_log(path):
    """list of {task: acc%, 'average': acc%} one per merge step, read from the raw log."""
    txt = open(path, encoding="utf-8", errors="ignore").read()
    txt = re.sub(r"[│╭╮╰╯─]", " ", txt.replace("\r", "\n"))
    out = []
    for blk in re.split(r"Evaluation Result:", txt)[1:]:
        blk = re.sub(r"\s+", " ", blk[:3000])
        pairs = re.findall(r"'([^']+)':\s*\{\s*'accuracy':\s*([0-9.eE-]+)", blk)
        d = {}
        for k, v in pairs:
            if k in d and k == "average": break
            d[k] = float(v) * 100
            if k == "average": break
        if "average" in d: out.append(d)
    return out

def load(key):
    p = f"logs/demo2_B32_TA8_{key}_raw.log"
    return steps_from_log(p) if os.path.exists(p) else []

zs = load("zeroshot"); opcm = load("opcm"); dac = load("dac"); gem_bb = load("gem")
zs_avg = zs[-1]["average"] if zs else None
gem_acc = gem_mem = None
f = glob.glob("outputs/cmm/demo2_B32_TA8_gem/version_*/p2_results.json")
if f:
    v = list(json.load(open(f[0])).values())[0]
    gem_acc, gem_mem = float(v["average"]) * 100, float(v["memory_per_task_pct"])

C = {"zs": "#6b7280", "opcm": "#2563eb", "dac": "#ea580c", "gem": "#059669"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

# ---- Figure 1: average accuracy as tasks are merged -------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.2))
if zs_avg is not None:
    ax.axhline(zs_avg, color=C["zs"], ls="--", lw=2, label=f"Pretrained CLIP, zero-shot ({zs_avg:.1f})")
for data, key, lab in [(opcm, "opcm", "OPCM (base paper)"), (dac, "dac", "DAC-Merge (ours)")]:
    if data:
        ys = [d["average"] for d in data]
        ax.plot(range(1, len(ys) + 1), ys, "-o", color=C[key], lw=2.5, ms=7, label=f"{lab} ({ys[-1]:.1f})")
if gem_acc is not None:
    ax.plot([8], [gem_acc], "*", color=C["gem"], ms=20, label=f"GEM = DAC-Merge + memory ({gem_acc:.1f}, {gem_mem:.2f}% memory/task)")
ax.set_xlabel("Number of tasks merged"); ax.set_ylabel("Average accuracy on tasks seen so far (%)")
ax.set_title("Accuracy as new tasks are merged (ViT-B/32, 8 tasks, order 0)")
ax.set_xticks(range(1, 9)); ax.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
ax.grid(axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(f"{OUT}/fig1_accuracy_vs_tasks.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- Figure 2: forgetting of each task (accuracy of every task, step by step) -------
for key, data, title in [("opcm", opcm, "OPCM"), ("dac", dac, "DAC-Merge")]:
    if not data: continue
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    tasks = [k for k in data[-1] if k != "average"]
    for t in tasks:
        xs = [i + 1 for i, d in enumerate(data) if t in d]; ys = [d[t] for d in data if t in d]
        ax.plot(xs, ys, "-o", lw=2, ms=5, label=NAMES.get(t, t))
    ax.set_xlabel("Number of tasks merged"); ax.set_ylabel("Accuracy of each task (%)")
    ax.set_title(f"{title}: accuracy of each task as later tasks are merged")
    ax.set_xticks(range(1, 9)); ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16)); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_{key}_per_task.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- Figure 3: final per-task accuracy ------------------------------------------------
sets = [("Zero-shot", zs[-1] if zs else None, C["zs"]), ("OPCM", opcm[-1] if opcm else None, C["opcm"]),
        ("DAC-Merge", dac[-1] if dac else None, C["dac"])]
sets = [s for s in sets if s[1]]
if sets:
    keys = [k for k in sets[0][1] if k != "average"] + ["average"]
    fig, ax = plt.subplots(figsize=(10, 5.2)); w = 0.8 / len(sets)
    for i, (lab, d, col) in enumerate(sets):
        ax.bar([j + i * w for j in range(len(keys))], [d.get(k, 0) for k in keys], w, label=lab, color=col)
    ax.set_xticks([j + w * (len(sets) - 1) / 2 for j in range(len(keys))])
    ax.set_xticklabels([NAMES.get(k, "AVERAGE") for k in keys]); ax.set_ylabel("Accuracy after the final merge (%)")
    ax.set_title("Final accuracy per task"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_final_per_task.png", dpi=200, bbox_inches="tight"); plt.close(fig)

print("saved to", OUT); print(sorted(os.listdir(OUT)))
for n, dd in [("zeroshot", zs), ("opcm", opcm), ("dac", dac), ("gem backbone", gem_bb)]:
    print(f"{n:13s} steps read: {len(dd)}", [round(d['average'], 2) for d in dd])
print("GEM with memory:", gem_acc, gem_mem)
