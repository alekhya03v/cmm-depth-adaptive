import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("results", exist_ok=True)

def acc(d, s):
    p = f"outputs/cmm/{d}/version_{s}"
    if not os.path.exists(f"{p}/model_names.json"): return None
    T = len(json.load(open(f"{p}/model_names.json")))
    f = f"{p}/report_{T-1}.json"
    return json.load(open(f))["average"]["accuracy"] * 100 if os.path.exists(f) else None

opcm = {s: acc("vitb32_TA8_opcm", s) for s in range(5)}

# ---- Fig A: alpha sweep + depth-down ----
alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
m = []
for a in alphas:
    d = "vitb32_TA8_opcm" if a == 0.5 else f"vitb32_TA8_opcm_alpha{a}"
    m.append(np.mean([v for s in range(3) if (v := acc(d, s)) is not None]))
dd3 = np.mean([acc("vitb32_TA8_cap_depthdown", s) for s in range(3)])
plt.figure(figsize=(6, 4))
plt.plot(alphas, m, "o-", label="OPCM, one global α")
plt.axhline(dd3, color="C3", ls="--", label=f"Ours: depth-down 0.7→0.3 ({dd3:.2f})")
plt.xlabel("projection threshold α"); plt.ylabel("average accuracy (%)")
plt.title("No global α beats depth-adaptive protection\n(ViT-B/32, 8 tasks, 3 task orders)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("results/fig_alpha_sweep.png", dpi=200)

# ---- Fig B: paired per-order comparison OPCM vs depth-down ----
dd = {s: acc("vitb32_TA8_cap_depthdown", s) for s in range(5)}
x = np.arange(5); w = 0.38
plt.figure(figsize=(6.5, 4))
plt.bar(x - w/2, [opcm[s] for s in range(5)], w, label="OPCM")
plt.bar(x + w/2, [dd[s] for s in range(5)], w, label="Ours (depth-down)", color="C3")
for s in range(5):
    plt.text(s + w/2, dd[s] + 0.05, f"{dd[s]-opcm[s]:+.1f}", ha="center", fontsize=8)
plt.xticks(x, [f"order {s+1}" for s in range(5)]); plt.ylim(73.5, 76.5)
plt.ylabel("average accuracy (%)")
plt.title(f"Same task orders: +{np.mean([dd[s]-opcm[s] for s in range(5)]):.2f} on average, never worse")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig("results/fig_paired_orders.png", dpi=200)

# ---- Fig C: every idea tested vs OPCM ----
variants = [("Reintroduce synergy (γ=1)", "vitb32_TA8_symerge_keep-true"),
            ("Retention floor ρ=0.95", "vitb32_TA8_cap_ret0.95"),
            ("Retention floor ρ=0.90", "vitb32_TA8_cap_ret0.9"),
            ("Overlap-adaptive α (β=0.5)", "vitb32_TA8_cap_oap_b0.5"),
            ("Depth-up 0.3→0.7", "vitb32_TA8_cap_depthup"),
            ("Best global α (0.5 = OPCM)", None),
            ("Depth-down 0.7→0.3 (ours)", "vitb32_TA8_cap_depthdown")]
labels, diffs = [], []
for lab, d in variants:
    if d is None: labels.append(lab); diffs.append(0.0); continue
    ds = [acc(d, s) - opcm[s] for s in range(5) if acc(d, s) is not None]
    labels.append(lab); diffs.append(np.mean(ds))
plt.figure(figsize=(7, 4))
cols = ["C3" if v > 0 else ("gray" if v == 0 else "C0") for v in diffs]
plt.barh(labels, diffs, color=cols); plt.axvline(0, color="k", lw=0.8)
for i, v in enumerate(diffs): plt.text(v + (0.05 if v >= 0 else -0.05), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
plt.xlabel("accuracy change vs OPCM (points, same task orders)")
plt.title("Everything we tested (ViT-B/32, 8 tasks)"); plt.tight_layout()
plt.savefig("results/fig_all_variants.png", dpi=200)

# ---- Fig D: the depth schedule itself ----
L = np.arange(12)
plt.figure(figsize=(5.5, 3.5))
plt.plot(L, [0.5]*12, "o-", label="OPCM (same α everywhere)")
plt.plot(L, 0.7 + L/11*(0.3-0.7), "o-", color="C3", label="Ours: depth-down")
plt.xlabel("transformer layer (0 = first)"); plt.ylabel("projection threshold α")
plt.title("More protection for early (generic) layers,\nless for late (task-specific) layers")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("results/fig_depth_schedule.png", dpi=200)
print("Saved: fig_alpha_sweep, fig_paired_orders, fig_all_variants, fig_depth_schedule (in results/)")
