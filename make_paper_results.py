"""
Regenerates EVERY table and graph used in the midsem slides from the raw run outputs.
Run from the opcm/ folder:   python make_paper_results.py
Writes: results/paper/tables/*.csv + *.md   and   results/paper/figs/*.png + *.pdf
Missing runs are skipped (a note is printed), so it can be re-run as batches finish.
"""
import json, os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/cmm"
RES = "results/paper"; TAB = f"{RES}/tables"; FIG = f"{RES}/figs"
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
C_OURS, C_BASE, C_SOFT, C_GOOD, C_GRAY = "#B8561A", "#2F5F8F", "#9BB4CF", "#4D7C59", "#5B6270"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

def run_stats(name, seed):
    d = f"{OUT}/{name}/version_{seed}"
    if not os.path.exists(f"{d}/model_names.json"):
        return None
    names = json.load(open(f"{d}/model_names.json")); T = len(names)
    if not os.path.exists(f"{d}/report_{T-1}.json"):
        return None
    R = {k: json.load(open(f"{d}/report_{k}.json")) for k in range(T) if os.path.exists(f"{d}/report_{k}.json")}
    acc = R[T-1]["average"]["accuracy"] * 100
    bwt = np.mean([(R[T-1][names[k]]["accuracy"] - R[k][names[k]]["accuracy"]) * 100 for k in range(T-1) if k in R])
    return acc, bwt

def seeds_of(name):
    return sorted(int(p.split("_")[-1]) for p in glob.glob(f"{OUT}/{name}/version_*") if run_stats(name, int(p.split("_")[-1])))

def paired(name, ref):
    s = [x for x in seeds_of(name) if run_stats(ref, x)]
    if not s:
        return None
    a = [run_stats(name, x)[0] for x in s]; b = [run_stats(name, x)[1] for x in s]
    d = [run_stats(name, x)[0] - run_stats(ref, x)[0] for x in s]
    return dict(acc=np.mean(a), acc_std=np.std(a), bwt=np.mean(b), diffs=d, gain=np.mean(d), n=len(s))

def save_table(fname, header, rows, note=""):
    with open(f"{TAB}/{fname}.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows: f.write(",".join(str(c) for c in r) + "\n")
    with open(f"{TAB}/{fname}.md", "w") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows: f.write("| " + " | ".join(str(c) for c in r) + " |\n")
        if note: f.write(f"\n{note}\n")
    print(f"[table] {fname}  ({len(rows)} rows)")

def save_fig(fname):
    plt.tight_layout(); plt.savefig(f"{FIG}/{fname}.png", dpi=200); plt.savefig(f"{FIG}/{fname}.pdf"); plt.close()
    print(f"[figure] {fname}")

f2 = lambda x: f"{x:+.2f}"
REF = "vitb32_TA8_opcm"

# T1 reproduction (baselines)
PAPER = {"opcm": ("75.5 ± 0.5", "-6.3 ± 1.1"), "task_arithmetic": ("67.5 ± 0.0", "-9.6 ± 1.5"),
         "weight_average": ("66.3 ± 0.0", "-11.5 ± 2.2"), "ties_merging": ("49.0 ± 10.2", "-15.3 ± 8.0")}
LBL = {"opcm": "OPCM", "task_arithmetic": "C. Task Arithmetic", "weight_average": "Weight Averaging", "ties_merging": "C. TIES-Merging"}
rows = []
for m in PAPER:
    s = seeds_of(f"vitb32_TA8_{m}")
    if not s: print(f"  (missing {m})"); continue
    A = [run_stats(f"vitb32_TA8_{m}", x)[0] for x in s]; B = [run_stats(f"vitb32_TA8_{m}", x)[1] for x in s]
    rows.append([LBL[m], PAPER[m][0], f"{np.mean(A):.1f} ± {np.std(A):.1f}", PAPER[m][1], f"{np.mean(B):.1f} ± {np.std(B):.1f}", len(s)])
save_table("T1_reproduction_vitb32_8tasks", ["Method", "Paper ACC", "Our ACC", "Paper BWT", "Our BWT", "Orders"], rows,
           "Paper values: OPCM (Tang et al., NeurIPS 2025), Table 1.")

# T2 OPCM reproduction across settings
rows = []
for lab, name, paper in [("B/32 · 8", "vitb32_TA8_opcm", 75.5), ("B/32 · 14", "vitb32_TALL14_opcm", 71.9),
                         ("B/32 · 20", "vitb32_TALL20_opcm", 65.7), ("B/16 · 8", "vitb16_TA8_opcm", 81.8)]:
    s = seeds_of(name)
    if s: rows.append([lab, paper, f"{np.mean([run_stats(name, x)[0] for x in s]):.2f}", len(s)])
save_table("T2_opcm_reproduction_settings", ["Setting", "OPCM paper", "OPCM ours", "Orders"], rows)

# F1 alpha sweep
xs, ys = [], []
for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    n = REF if a == 0.5 else f"vitb32_TA8_opcm_alpha{a}"
    v = [run_stats(n, s)[0] for s in range(3) if run_stats(n, s)]
    if v: xs.append(a); ys.append(np.mean(v))
if xs:
    dd = [run_stats("vitb32_TA8_cap_depthdown", s)[0] for s in range(3) if run_stats("vitb32_TA8_cap_depthdown", s)]
    plt.figure(figsize=(6, 4)); plt.plot(xs, ys, "o-", color=C_BASE, label="OPCM, one global α")
    if dd: plt.axhline(np.mean(dd), ls="--", color=C_OURS, label=f"Ours: depth-down ({np.mean(dd):.2f})")
    plt.xlabel("global threshold α"); plt.ylabel("ACC (%)"); plt.legend(); save_fig("F1_alpha_sweep")
    save_table("T3_alpha_sweep", ["alpha", "ACC (3 orders)"], [[x, f"{y:.2f}"] for x, y in zip(xs, ys)])

# F2 + T4-T6 diagnosis
recs = []
for f in glob.glob(f"{OUT}/vitb32_TA8_symerge_keep-false/version_*/diagnostics.json"):
    recs += json.load(open(f))
if recs:
    steps = sorted({r["step"] for r in recs})
    kept = [np.mean([r["retained_energy"] for r in recs if r["step"] == s]) * 100 for s in steps]
    layer = {}
    for r in recs:
        m = re.search(r"layers\.(\d+)\.", r["param"])
        if m: layer.setdefault(int(m.group(1)), []).append(r["removed_fraction"] * 100)
    L = sorted(layer); rem = [np.mean(layer[l]) for l in L]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot([s + 1 for s in steps], kept, "o-", color=C_BASE); ax[0].set_title("Share of each new task kept"); ax[0].set_xlabel("merge step"); ax[0].set_ylabel("%")
    ax[1].bar(L, rem, color=C_BASE); ax[1].set_title("Share removed, per layer"); ax[1].set_xlabel("transformer layer"); ax[1].set_ylabel("%")
    save_fig("F2_diagnosis_step_layer")
    save_table("T4_diagnosis", ["Quantity", "Value"], [
        ["removed share of each new task vector (%)", f"{np.mean([r['removed_fraction'] for r in recs])*100:.1f}"],
        ["of removed: agreeing (%)", f"{np.mean([r['SDR'] for r in recs])*100:.1f}"],
        ["of removed: conflicting (%)", f"{np.mean([r['conflict_ratio'] for r in recs])*100:.1f}"],
        ["of removed: cross terms (%)", f"{np.mean([r['cross_ratio'] for r in recs])*100:.1f}"],
        ["kept, first merge step (%)", f"{kept[0]:.1f}"], ["kept, last merge step (%)", f"{kept[-1]:.1f}"],
        ["removed in layer 0 (%)", f"{rem[0]:.1f}"], ["removed in layer 10 (%)", f"{rem[10]:.1f}" if len(rem) > 10 else "-"]])
    save_table("T5_removed_per_layer", ["layer", "removed (%)"], [[l, f"{v:.1f}"] for l, v in zip(L, rem)])
    save_table("T6_kept_per_step", ["merge step", "kept (%)"], [[s + 1, f"{v:.1f}"] for s, v in zip(steps, kept)])

# F4 + T7 tested hypotheses
H = [("Reintroduce aligned overlap", "vitb32_TA8_symerge_keep-true"), ("Retention floor 0.95", "vitb32_TA8_cap_ret0.95"),
     ("Overlap-adaptive α (β=0.5)", "vitb32_TA8_cap_oap_b0.5"), ("Depth-up 0.3→0.7", "vitb32_TA8_cap_depthup"),
     ("Retention floor 0.90", "vitb32_TA8_cap_ret0.9"), ("Retention floor 0.85", "vitb32_TA8_cap_ret0.85"),
     ("Depth-down 0.7→0.3", "vitb32_TA8_cap_depthdown")]
hv = [(l, paired(n, REF)) for l, n in H]; hv = [(l, p) for l, p in hv if p]
if hv:
    plt.figure(figsize=(7, 4)); labs = [l for l, _ in hv]; vals = [p["gain"] for _, p in hv]
    plt.barh(labs, vals, color=[C_OURS if v > 0 else C_SOFT for v in vals]); plt.axvline(0, color="k", lw=0.8)
    for i, v in enumerate(vals): plt.text(v + (0.05 if v >= 0 else -0.05), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    plt.xlabel("ACC change vs OPCM (same task orders)"); save_fig("F4_tested_hypotheses")
    save_table("T7_tested_hypotheses", ["Variant", "vs OPCM", "BWT", "Orders"], [[l, f2(p["gain"]), f"{p['bwt']:.1f}", p["n"]] for l, p in hv])

# F5 depth schedule
l = np.arange(12); plt.figure(figsize=(6, 3.6))
plt.plot(l, [0.5] * 12, "o-", color=C_BASE, label="OPCM"); plt.plot(l, 0.7 - 0.4 * l / 11, "o-", color=C_OURS, label="Ours: depth-down")
plt.xlabel("transformer layer"); plt.ylabel("threshold α(l)"); plt.legend(); save_fig("F5_depth_schedule")

# F6 + T8 depth-down across settings
rows, per = [], None
for lab, ref, ours in [("B/32 · 8", REF, "vitb32_TA8_cap_depthdown"), ("B/32 · 14", "vitb32_TALL14_opcm", "vitb32_TALL14_cap_depthdown"),
                       ("B/32 · 20", "vitb32_TALL20_opcm", "vitb32_TALL20_cap_depthdown"), ("B/16 · 8", "vitb16_TA8_opcm", "vitb16_TA8_cap_depthdown")]:
    p = paired(ours, ref)
    if not p: continue
    s = [x for x in seeds_of(ours) if run_stats(ref, x)]
    opcm = np.mean([run_stats(ref, x)[0] for x in s])
    wins = sum(d > 0.05 for d in p["diffs"]); ties = sum(abs(d) <= 0.05 for d in p["diffs"])
    rows.append([lab, f"{opcm:.2f}", f"{p['acc']:.2f}", f2(p["gain"]), f"{wins} of {p['n']}" + (f" ({ties} tie)" if ties else "")])
    if lab == "B/32 · 8": per = p["diffs"]
save_table("T8_depthdown_across_settings", ["Setting", "OPCM", "Ours (depth-down)", "Gain", "Orders better"], rows)
if per:
    plt.figure(figsize=(5.5, 3.8)); plt.bar([f"order {i+1}" for i in range(len(per))], per, color=C_OURS)
    for i, v in enumerate(per): plt.text(i, v + 0.01, f"{v:+.1f}", ha="center", fontsize=9)
    plt.ylabel("gain vs OPCM"); plt.title("Depth-down, ViT-B/32, 8 tasks"); save_fig("F6_depthdown_per_order")

# T9 strength sweep + F7 gain vs strength + F8 trade-off
DD = "vitb32_TA8_dac_dd_sc"
strength_rows = [("OPCM", REF, "1.0 everywhere", 1.0), ("Uniform ×1.2", "vitb32_TA8_dac_sc1.2", "1.2 everywhere", 1.2),
                 ("Uniform ×1.4", "vitb32_TA8_dac_sc1.4", "1.4 everywhere", 1.4), ("Uniform ×1.6", "vitb32_TA8_dac_sc1.6", "1.6 everywhere", 1.6),
                 ("Uniform ×1.8", "vitb32_TA8_dac_sc1.8", "1.8 everywhere", 1.8), ("Uniform ×2.0", "vitb32_TA8_dac_sc2.0", "2.0 everywhere", 2.0),
                 ("Ours, gentle slope", DD + "1.6-2.0", "1.6 → 2.0", 1.8), ("Ours, medium slope", DD + "1.4-2.2", "1.4 → 2.2", 1.8),
                 ("Ours, steep slope", DD + "1.2-2.4", "1.2 → 2.4", 1.8)]
rows = []
for lab, n, prof, avg in strength_rows:
    p = paired(n, REF) if n != REF else dict(acc=np.mean([run_stats(REF, s)[0] for s in range(3)]), gain=0.0, bwt=np.mean([run_stats(REF, s)[1] for s in range(3)]), n=3)
    if p: rows.append([lab, prof, avg, f"{p['acc']:.2f}", f2(p["gain"]), f"{p['bwt']:.1f}", p["n"]])
save_table("T9_strength_sweep", ["Variant", "Strength layer 0→11", "Avg", "ACC", "vs OPCM", "BWT", "Orders"], rows)

uni = [(a, paired(f"vitb32_TA8_dac_sc{a}", REF)) for a in ["1.2", "1.4", "1.6", "1.8", "2.0"]]
dep = [(1.1, paired(DD + "1.0-1.2", REF)), (1.4, paired(DD + "1.2-1.6", REF)), (1.6, paired(DD + "1.4-1.8", REF)),
       (1.8, paired(DD + "1.2-2.4", REF)), (2.0, paired(DD + "1.6-2.4", REF))]
uni = [(float(a), p) for a, p in uni if p]; dep = [(a, p) for a, p in dep if p]
if uni and dep:
    plt.figure(figsize=(6, 4.2))
    plt.plot([a for a, _ in uni], [p["gain"] for _, p in uni], "o-", color=C_BASE, label="uniform strength")
    plt.plot([a for a, _ in dep], [p["gain"] for _, p in dep], "s-", color=C_OURS, label="ours: depth-aware")
    plt.xlabel("average strength"); plt.ylabel("gain vs OPCM"); plt.title("Placement beats amount"); plt.legend(); save_fig("F7_gain_vs_strength")

pts = [("OPCM", None, C_GRAY), ("depth-down", "vitb32_TA8_cap_depthdown", C_SOFT), ("uniform ×1.4", "vitb32_TA8_dac_sc1.4", C_BASE),
       ("uniform ×2.0", "vitb32_TA8_dac_sc2.0", C_BASE), ("per-layer budget", "vitb32_TA8_bud_layer", C_GOOD),
       ("DAC 1.2→1.6", DD + "1.2-1.6", C_OURS), ("DAC 1.2→2.4", DD + "1.2-2.4", C_OURS)]
plt.figure(figsize=(6, 4.2))
for lab, n, c in pts:
    if n is None: x, y = np.mean([run_stats(REF, s)[1] for s in range(5)]), 0.0
    else:
        p = paired(n, REF)
        if not p: continue
        x, y = p["bwt"], p["gain"]
    plt.scatter(x, y, s=70, color=c); plt.annotate(lab, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
plt.xlabel("BWT (forgetting)"); plt.ylabel("gain vs OPCM"); plt.title("Accuracy vs forgetting"); save_fig("F8_tradeoff")

# T10 full ablation + F9 best per order
AB = [("Protection", "Depth-down α 0.7→0.3", "vitb32_TA8_cap_depthdown"), ("Protection", "Depth-up α 0.3→0.7", "vitb32_TA8_cap_depthup"),
      ("Strength", "Uniform ×1.4", "vitb32_TA8_dac_sc1.4"), ("Strength", "Uniform ×2.0", "vitb32_TA8_dac_sc2.0"),
      ("Strength", "Late layers 1.4→1.8", "vitb32_TA8_dac_sc1.4-1.8"), ("Both", "Depth-down + ×1.4", DD + "1.4"),
      ("Both", "Depth-down + 1.2→1.6", DD + "1.2-1.6"), ("Both", "Depth-down + 1.4→2.2", DD + "1.4-2.2"),
      ("Both", "Depth-down + 1.2→2.4 (DAC-Merge)", DD + "1.2-2.4"),
      ("Budget", "Per-layer budget (automatic)", "vitb32_TA8_bud_layer"), ("Budget", "Per-layer budget + depth-down", "vitb32_TA8_bud_layer_dd"),
      ("Rejected", "Reintroduce aligned overlap", "vitb32_TA8_symerge_keep-true"), ("Rejected", "Retention floor 0.95", "vitb32_TA8_cap_ret0.95"),
      ("Rejected", "Overlap-adaptive α", "vitb32_TA8_cap_oap_b0.5"),
      ("Component A", "DAC + keep cross terms", "vitb32_TA8_dac_best_keepcross"), ("Component A", "DAC + keep half cross terms", "vitb32_TA8_dac_best_keepcrossh"),
      ("Component B", "DAC + LayerNorm pretrained", "vitb32_TA8_dac_best_lnpre"), ("Component B", "DAC + LayerNorm half", "vitb32_TA8_dac_best_lnhalf")]
rows = []
for g, lab, n in AB:
    p = paired(n, REF)
    if p: rows.append([g, lab, f2(p["gain"]), f"{p['bwt']:.1f}", p["n"], " ".join(f"{d:+.1f}" for d in p["diffs"])])
    else: print(f"  (ablation: {lab} not available yet)")
save_table("T10_full_ablation", ["Group", "Variant", "vs OPCM", "BWT", "Orders", "Per order"], rows)
best = paired(DD + "1.2-2.4", REF)
if best:
    plt.figure(figsize=(4.5, 3.8)); plt.bar([f"order {i+1}" for i in range(best["n"])], best["diffs"], color=C_OURS)
    for i, v in enumerate(best["diffs"]): plt.text(i, v + 0.05, f"{v:+.1f}", ha="center", fontsize=9)
    plt.ylabel("gain vs OPCM"); plt.title("DAC-Merge per task order"); save_fig("F9_dac_per_order")

# T11 literature comparison (published numbers, cited) + our DAC-Merge
LIT = [["OPCM", "NeurIPS 2025", "nothing", 75.5, 71.9, 65.7, 81.8, 77.1, 70.3, 87.0, 83.5, 76.0],
       ["NUFILT", "ICLR 2026", "optimisation", 83.6, 78.0, 71.0, 87.3, 83.0, 78.1, 91.6, 89.2, 84.7],
       ["SAFE-Merge (concurrent)", "arXiv 2026", "optimisation", 83.0, 79.2, 73.5, 86.2, 83.3, 78.4, 91.2, 89.3, 84.7],
       ["MINGLE", "NeurIPS 2025", "experts + test data", 85.8, 81.6, 77.1, 88.3, 84.9, 81.9, 91.8, 88.8, 85.5],
       ["MADE-IT", "arXiv 2026", "growing experts", 87.1, 84.3, 81.4, 90.4, 87.7, 83.0, 93.1, 91.7, 89.6],
       ["ODE-M", "arXiv 2026", "calibration data", 79.6, 74.1, 67.1, "-", "-", "-", "-", "-", "-"]]
ours = ["Ours: DAC-Merge", "-", "nothing"]
for n in ["vitb32_TA8_dac_dd_sc1.2-2.4", "vitb32_TALL14_dac_dd_sc1.2-2.4", "vitb32_TALL20_dac_dd_sc1.2-2.4",
          "vitb16_TA8_dac_dd_sc1.2-2.4", "vitb16_TALL14_dac_dd_sc1.2-2.4", "vitb16_TALL20_dac_dd_sc1.2-2.4",
          "vitl14_TA8_dac_dd_sc1.2-2.4", "vitl14_TALL14_dac_dd_sc1.2-2.4", "vitl14_TALL20_dac_dd_sc1.2-2.4"]:
    s = seeds_of(n); ours.append(f"{np.mean([run_stats(n, x)[0] for x in s]):.1f}" if s else "running")
save_table("T11_literature_comparison",
           ["Method", "Venue", "Uses", "B32-8", "B32-14", "B32-20", "B16-8", "B16-14", "B16-20", "L14-8", "L14-14", "L14-20"],
           LIT + [ours], "Published numbers: OPCM, NUFILT, SAFE-Merge from SAFE-Merge Tables 1 and S.5; MINGLE, MADE-IT from MADE-IT Table 1; ODE-M from its paper.")

# T12 Pipeline 2 (DA-NUFILT), if its outputs exist
P2 = "../NUFILT/outputs/p2"
if os.path.isdir(P2):
    rows = []
    for n in ["nufilt", "da_null", "da_lam", "da_both"]:
        v = [json.load(open(f))["average"]["accuracy"] * 100 for f in sorted(glob.glob(f"{P2}/{n}/version_*/report_7.json"))]
        if v: rows.append([n, f"{np.mean(v):.2f}", len(v), " ".join(f"{x:.2f}" for x in v)])
    save_table("T12_pipeline2_da_nufilt", ["Variant", "ACC", "Orders", "Per order"], rows, "NUFILT paper (10 orders): 83.6 on ViT-B/32, 8 tasks.")
print(f"\nAll outputs in {RES}/")
