import json, glob, os, re
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("results", exist_ok=True)

def final_acc(d):
    if not os.path.exists(f"{d}/model_names.json"): return None
    names = json.load(open(f"{d}/model_names.json")); T = len(names)
    f = f"{d}/report_{T-1}.json"
    return json.load(open(f))["average"]["accuracy"] * 100 if os.path.exists(f) else None

# ---- 1. sanity check: keep-false must equal OPCM; keep-true = first test of our idea ----
print(f"{'seed':>4} {'OPCM':>8} {'keep-false':>11} {'keep-true':>10}")
for s in range(5):
    a = final_acc(f"outputs/cmm/vitb32_TA8_opcm/version_{s}")
    b = final_acc(f"outputs/cmm/vitb32_TA8_symerge_keep-false/version_{s}")
    c = final_acc(f"outputs/cmm/vitb32_TA8_symerge_keep-true/version_{s}")
    fmt = lambda x: f"{x:8.2f}" if x is not None else "     n/a"
    print(f"{s:>4} {fmt(a)} {fmt(b):>11} {fmt(c):>10}")

# ---- 2. load diagnostics from OPCM (keep-false) runs ----
recs = []
for f in glob.glob("outputs/cmm/vitb32_TA8_symerge_keep-false/version_*/diagnostics.json"):
    recs += json.load(open(f))
if not recs:
    print("\nNo diagnostics yet."); raise SystemExit
print(f"\nLoaded {len(recs)} layer-step records")

by_step = defaultdict(lambda: defaultdict(list))
by_layer = defaultdict(list)
by_module = defaultdict(list)
for r in recs:
    for k in ("SDR", "conflict_ratio", "cross_ratio", "retained_energy", "removed_fraction"):
        by_step[r["step"]][k].append(r[k])
    L = re.findall(r"layers\.(\d+)\.", r["param"])
    if L: by_layer[int(L[0])].append(r["removed_fraction"])
    mod = r["param"].split(".")[-2]
    by_module[mod].append(r["removed_fraction"])

steps = sorted(by_step)
m = lambda s, k: np.mean(by_step[s][k])

print("\nFLAW 1 - what OPCM removes (share of removed energy):")
print(f"  synergy (agreeing)   : {np.mean([r['SDR'] for r in recs])*100:5.1f}%")
print(f"  conflict (opposing)  : {np.mean([r['conflict_ratio'] for r in recs])*100:5.1f}%")
print(f"  cross terms          : {np.mean([r['cross_ratio'] for r in recs])*100:5.1f}%")
print(f"  removed / total task : {np.mean([r['removed_fraction'] for r in recs])*100:5.1f}% of each new task vector")

print("\nFLAW 2 - retained energy of new task vs merge step:")
for s in steps:
    print(f"  step {s+1}: kept {m(s,'retained_energy')*100:5.1f}%   removed {m(s,'removed_fraction')*100:5.1f}%")

print("\nFLAW 3 - removed fraction per layer:")
for l in sorted(by_layer):
    print(f"  layer {l:2d}: {np.mean(by_layer[l])*100:5.1f}%")
print("  by module:", {k: round(np.mean(v)*100, 1) for k, v in by_module.items()})

# ---- plots ----
x = [s + 1 for s in steps]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].stackplot(x, [[m(s,"SDR") for s in steps], [m(s,"conflict_ratio") for s in steps],
                    [m(s,"cross_ratio") for s in steps]],
                labels=["synergy (agreeing)", "conflict (opposing)", "cross terms"], alpha=0.8)
ax[0].set_title("Flaw 1: what OPCM removes"); ax[0].set_xlabel("merge step"); ax[0].set_ylabel("share of removed energy")
ax[0].legend(fontsize=8, loc="lower right")
ax[1].plot(x, [m(s,"retained_energy")*100 for s in steps], marker="o")
ax[1].set_title("Flaw 2: % of new task kept"); ax[1].set_xlabel("merge step"); ax[1].grid(alpha=0.3)
ls = sorted(by_layer)
ax[2].bar(ls, [np.mean(by_layer[l])*100 for l in ls])
ax[2].set_title("Flaw 3: % of new task removed per layer"); ax[2].set_xlabel("transformer layer")
plt.tight_layout(); plt.savefig("results/flaw_diagnostics.png", dpi=200)
print("\nSaved results/flaw_diagnostics.png")
