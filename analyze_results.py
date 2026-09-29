import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# individual fine-tuned accuracy (OPCM paper, Table 2, CLIP ViT-B/32)
FT = {"sun397": 74.9, "stanford-cars": 78.5, "resisc45": 95.1, "eurosat": 99.1,
      "svhn": 97.3, "gtsrb": 98.9, "mnist": 99.6, "dtd": 79.7}
LABEL = {"opcm": "OPCM", "weight_average": "Weight Avg (SWA)",
         "task_arithmetic": "C. Task Arithmetic", "ties_merging": "C. TIES-Merging"}
PAPER = {"opcm": "75.5 ± 0.5 | -6.3 ± 1.1", "weight_average": "66.3 ± 0.0 | -11.5 ± 2.2",
         "task_arithmetic": "67.5 ± 0.0 | -9.6 ± 1.5", "ties_merging": "49.0 ± 10.2 | -15.3 ± 8.0"}

def load_run(d):
    names = json.load(open(f"{d}/model_names.json")); T = len(names)
    if not os.path.exists(f"{d}/report_{T-1}.json"):
        return None
    R = [json.load(open(f"{d}/report_{k}.json")) for k in range(T)]
    M = np.full((T, T), np.nan)                # M[step, task] in %
    for k in range(T):
        for j in range(k + 1):
            M[k, j] = R[k][names[j]]["accuracy"] * 100
    acc = np.nanmean(M[T-1])
    bwt = np.mean([M[T-1, j] - M[j, j] for j in range(T-1)])
    plast = [M[j, j] / FT[names[j]] * 100 for j in range(T)]   # % of fine-tuned kept when merged
    curve = [np.nanmean(M[k]) for k in range(T)]               # avg acc over tasks seen so far
    return dict(names=names, M=M, acc=acc, bwt=bwt, plast=plast, curve=curve)

runs = {}
for mdir in sorted(glob.glob("outputs/cmm/vitb32_TA8_*")):
    m = mdir.split("vitb32_TA8_")[1]
    if m not in LABEL: continue
    rs = [r for d in sorted(glob.glob(f"{mdir}/version_*")) if (r := load_run(d))]
    if rs: runs[m] = rs

# ---- table ----
lines = [f"{'Method':22s} {'runs':>4s}   {'ACC (ours)':>14s}   {'BWT (ours)':>14s}   Paper ACC | BWT"]
for m, rs in runs.items():
    A = [r["acc"] for r in rs]; B = [r["bwt"] for r in rs]
    lines.append(f"{LABEL[m]:22s} {len(rs):4d}   {np.mean(A):6.1f} ± {np.std(A):4.1f}   "
                 f"{np.mean(B):6.1f} ± {np.std(B):4.1f}   {PAPER[m]}")
open("results/table_vitb32_TA8.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- plot 1: accuracy-matrix heatmaps (seed 0) ----
ms = list(runs)
fig, axes = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 4))
axes = np.atleast_1d(axes)
for ax, m in zip(axes, ms):
    r = runs[m][0]
    im = ax.imshow(r["M"], vmin=30, vmax=100, cmap="viridis")
    ax.set_xticks(range(8)); ax.set_xticklabels(r["names"], rotation=60, fontsize=7)
    ax.set_yticks(range(8)); ax.set_yticklabels([f"step {k+1}" for k in range(8)], fontsize=7)
    ax.set_title(f"{LABEL[m]}\nACC {r['acc']:.1f}", fontsize=9)
    for k in range(8):
        for j in range(k + 1):
            ax.text(j, k, f"{r['M'][k, j]:.0f}", ha="center", va="center", fontsize=6, color="w")
fig.colorbar(im, ax=axes, shrink=0.7, label="accuracy (%)")
plt.savefig("results/accuracy_matrices_seed0.png", dpi=200, bbox_inches="tight")

# ---- plot 2: average accuracy vs number of merged tasks ----
plt.figure(figsize=(5.5, 4))
for m, rs in runs.items():
    C = np.array([r["curve"] for r in rs])
    x = np.arange(1, 9)
    plt.plot(x, C.mean(0), marker="o", label=LABEL[m])
    plt.fill_between(x, C.mean(0) - C.std(0), C.mean(0) + C.std(0), alpha=0.15)
plt.xlabel("number of merged tasks"); plt.ylabel("avg accuracy on merged tasks (%)")
plt.title("Accuracy as tasks are merged (ViT-B/32)"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.savefig("results/accuracy_vs_tasks.png", dpi=200, bbox_inches="tight")

# ---- plot 3: plasticity (how much of a task is kept right when it is merged) ----
plt.figure(figsize=(5.5, 4))
for m, rs in runs.items():
    P = np.array([r["plast"] for r in rs])
    x = np.arange(1, 9)
    plt.plot(x, P.mean(0), marker="o", label=LABEL[m])
plt.xlabel("merge position of the task"); plt.ylabel("% of fine-tuned accuracy kept at merge")
plt.title("Plasticity vs merge position (avg over task orders)"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.savefig("results/plasticity_vs_position.png", dpi=200, bbox_inches="tight")

print("\nSaved: results/table_vitb32_TA8.txt, accuracy_matrices_seed0.png, accuracy_vs_tasks.png, plasticity_vs_position.png")
