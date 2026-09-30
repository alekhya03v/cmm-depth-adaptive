"""Measures the real memory each method stores after each task (from actual model tensors)."""
import re, json, os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import CLIPVisionModel

TASKS = ["sun397", "stanford-cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd",
         "oxford_flowers102", "pcam", "fer2013", "oxford-iiit-pet", "stl10", "cifar100",
         "cifar10", "food101", "fashion_mnist", "emnist_letters", "kmnist", "rendered-sst2"]
LAST_K, NL = 4, 12
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results/paper/figs", exist_ok=True); os.makedirs("results/paper/tables", exist_ok=True)

pre = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
pre_p = {n: p.detach().float() for n, p in pre.named_parameters()}
P = sum(p.numel() for p in pre_p.values())
BYTES = 4  # float32

def layer(n):
    m = re.search(r"layers\.(\d+)\.", n)
    return int(m.group(1)) if m else None

def lowrank_numel(d, r):
    U, S, Vh = torch.linalg.svd(d.to(dev), full_matrices=False)
    r = min(r, S.numel())
    return U[:, :r].numel() + S[:r].numel() + Vh[:r].numel()

rows, cum = [], {"opcm": 0, "dac": 0, "r16": 0, "r64": 0, "tall": 0, "all_models": 0}
for t, task in enumerate(TASKS, 1):
    try:
        m = CLIPVisionModel.from_pretrained(f"tanganke/clip-vit-base-patch32_{task}")
    except Exception as e:
        print("skip", task, e); continue
    add = {"r16": 0, "r64": 0, "tall": 0}
    for n, p in m.named_parameters():
        d = p.detach().float() - pre_p[n]
        L = layer(n)
        if L is not None and d.dim() == 2:
            add["tall"] += d.numel() / 32
        if L is None or L < NL - LAST_K:
            continue
        if d.dim() == 2:
            add["r16"] += lowrank_numel(d, 16); add["r64"] += lowrank_numel(d, 64)
        else:
            add["r16"] += d.numel(); add["r64"] += d.numel()
    for k in add: cum[k] += add[k]
    cum["opcm"] += 1; cum["dac"] += 1          # one scalar (task-vector norm) per task
    cum["all_models"] += P
    rows.append({"tasks": t, **{k: v for k, v in cum.items()},
                 "r16_pct_task": 100 * add["r16"] / P, "r64_pct_task": 100 * add["r64"] / P,
                 "tall_pct_task": 100 * add["tall"] / P})
    print(f"task {t:2d} {task:18s} extra MB -> OPCM {cum['opcm']*BYTES/2**20:7.4f} | "
          f"r16 {cum['r16']*BYTES/2**20:7.1f} | r64 {cum['r64']*BYTES/2**20:7.1f} | "
          f"TALL {cum['tall']*BYTES/2**20:7.1f} | per task: r16 {rows[-1]['r16_pct_task']:.2f}% "
          f"r64 {rows[-1]['r64_pct_task']:.2f}% TALL {rows[-1]['tall_pct_task']:.2f}%", flush=True)
    del m

json.dump({"P": P, "bytes_per_param": BYTES, "rows": rows}, open("results/paper/tables/T14_memory_measured.json", "w"), indent=1)
mb = lambda v: v * BYTES / 2**20
base = mb(2 * P)   # pretrained + merged model kept by every method
x = [r["tasks"] for r in rows]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for k, lab, st in [("opcm", "OPCM", "o-"), ("dac", "DAC-Merge (ours)", "s--"), ("r16", "depth-targeted r16 (ours)", "^-"),
                   ("r64", "depth-targeted r64 (ours)", "v-"), ("tall", "TALL-masks", "d-")]:
    ax[0].plot(x, [mb(r[k]) for r in rows], st, label=lab)
    ax[1].plot(x, [base + mb(r[k]) for r in rows], st, label=lab)
ax[1].plot(x, [base + mb(r["all_models"]) for r in rows], "k:", label="keep all fine-tuned models")
ax[0].set_title("Extra memory stored (MB)"); ax[1].set_title("Total memory incl. backbone (MB)")
for a in ax: a.set_xlabel("number of tasks merged"); a.grid(alpha=0.3)
ax[1].set_yscale("log"); ax[0].legend(fontsize=8); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("results/paper/figs/F10_memory_growth.png", dpi=200); plt.savefig("results/paper/figs/F10_memory_growth.pdf")
with open("results/paper/tables/T14_memory_measured.md", "w") as f:
    f.write(f"Model parameters P = {P:,} ({mb(P):.1f} MB fp32)\n\n")
    f.write("| Tasks | OPCM extra MB | DAC extra MB | r16 extra MB | r64 extra MB | TALL extra MB | all models MB |\n|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['tasks']} | {mb(r['opcm']):.4f} | {mb(r['dac']):.4f} | {mb(r['r16']):.1f} | {mb(r['r64']):.1f} | {mb(r['tall']):.1f} | {mb(r['all_models']):.0f} |\n")
    f.write(f"\nMean memory per task: r16 {np.mean([r['r16_pct_task'] for r in rows]):.2f}%, "
            f"r64 {np.mean([r['r64_pct_task'] for r in rows]):.2f}%, TALL {np.mean([r['tall_pct_task'] for r in rows]):.2f}% of the model\n")
print(open("results/paper/tables/T14_memory_measured.md").read())
