import json, glob, os, re
import numpy as np

def run_stats(d):
    names = json.load(open(f"{d}/model_names.json")); T = len(names)
    if not os.path.exists(f"{d}/report_{T-1}.json"): return None
    R = [json.load(open(f"{d}/report_{k}.json")) for k in range(T)]
    acc = R[-1]["average"]["accuracy"] * 100
    bwt = np.mean([(R[-1][names[k]]["accuracy"] - R[k][names[k]]["accuracy"]) * 100 for k in range(T-1)])
    kept = None
    if os.path.exists(f"{d}/diagnostics.json"):
        kept = np.nanmean([r.get("retained_energy", np.nan) for r in json.load(open(f"{d}/diagnostics.json"))]) * 100
    return acc, bwt, kept

base = {int(v.split("_")[-1]): run_stats(v) for v in glob.glob("outputs/cmm/vitb32_TA8_symerge_keep-false/version_*")}
exps = ["opcm_ref"] + sorted(os.path.basename(p) for p in glob.glob("outputs/cmm/vitb32_TA8_cap_*") + glob.glob("outputs/cmm/vitb32_TA8_dac_*"))
print(f"{'variant':30s} {'orders':>6s} {'ACC':>7s} {'vs OPCM (same orders)':>22s} {'BWT':>6s} {'% new task kept':>16s}")
for e in exps:
    if e == "opcm_ref":
        rows = {s: r for s, r in base.items() if r}
        name = "OPCM (reference)"
    else:
        rows = {}
        for v in glob.glob(f"outputs/cmm/{e}/version_*"):
            r = run_stats(v)
            if r: rows[int(v.split("_")[-1])] = r
        name = e.replace("vitb32_TA8_", "")
    if not rows: print(f"{name:30s} {'0':>6s}  (running)"); continue
    A = [r[0] for r in rows.values()]; B = [r[1] for r in rows.values()]
    K = [r[2] for r in rows.values() if r[2] is not None]
    diffs = [rows[s][0] - base[s][0] for s in rows if s in base and base[s]]
    d = f"{np.mean(diffs):+.2f} (per order: {', '.join(f'{x:+.1f}' for x in diffs)})" if e != "opcm_ref" else "-"
    k = f"{np.mean(K):.1f}" if K else "-"
    print(f"{name:30s} {len(rows):6d} {np.mean(A):7.2f} {d:>22s} {np.mean(B):6.1f} {k:>16s}")
