import json, os
import numpy as np

def stats(d):
    names = json.load(open(f"{d}/model_names.json")); T = len(names)
    R = [json.load(open(f"{d}/report_{k}.json")) for k in range(T)]
    acc = R[-1]["average"]["accuracy"] * 100
    bwt = np.mean([(R[-1][names[k]]["accuracy"] - R[k][names[k]]["accuracy"]) * 100 for k in range(T - 1)])
    return acc, bwt

paper = {"TALL14": 71.9, "TALL20": 65.7}
for U in ["TALL14", "TALL20"]:
    print(f"\n===== {U[4:]} tasks (OPCM in literature: {paper[U]}) =====")
    print(f"{'order':>6} {'OPCM':>8} {'ours':>8} {'diff':>7} {'BWT OPCM':>9} {'BWT ours':>9}")
    D, A, B, BA, BB = [], [], [], [], []
    for s in range(3):
        o = f"outputs/cmm/vitb32_{U}_opcm/version_{s}"
        c = f"outputs/cmm/vitb32_{U}_cap_depthdown/version_{s}"
        if not (os.path.exists(o) and os.path.exists(c)): continue
        (a, ba), (b, bb) = stats(o), stats(c)
        A.append(a); B.append(b); D.append(b - a); BA.append(ba); BB.append(bb)
        print(f"{s+1:>6} {a:8.2f} {b:8.2f} {b-a:+7.2f} {ba:9.2f} {bb:9.2f}")
    print(f"{'mean':>6} {np.mean(A):8.2f} {np.mean(B):8.2f} {np.mean(D):+7.2f} {np.mean(BA):9.2f} {np.mean(BB):9.2f}")
