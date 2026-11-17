import sys, itertools, json
import numpy as np, torch
from transformers import CLIPVisionModel

torch.set_num_threads(4)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "clip-vit-base-patch32"
TASKS = ["sun397", "stanford-cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd"]
pre = CLIPVisionModel.from_pretrained(f"openai/{MODEL}").state_dict()
NL = 1 + max(int(k.split("layers.")[1].split(".")[0]) for k in pre if "encoder.layers." in k)
mats = [[k for k in pre if f"encoder.layers.{l}." in k and k.endswith("weight") and pre[k].ndim == 2] for l in range(NL)]

tv = []
for t in TASKS:
    sd = CLIPVisionModel.from_pretrained(f"tanganke/{MODEL}_{t}").state_dict()
    tv.append([[sd[k].float() - pre[k].float() for k in mats[l]] for l in range(NL)])
    del sd

rows = []
for l in range(NL):
    norm = np.mean([torch.sqrt(sum(m.pow(2).sum() for m in tv[t][l])).item() for t in range(len(TASKS))])
    er = []
    for t in range(len(TASKS)):
        for m in tv[t][l]:
            s = torch.linalg.svdvals(m)
            r = ((s.cumsum(0) / s.sum()) > 0.5).float().argmax().item()
            er.append(r / min(m.shape))
    flat = [torch.cat([m.flatten() for m in tv[t][l]]) for t in range(len(TASKS))]
    cos = np.mean([torch.nn.functional.cosine_similarity(a, b, dim=0).item() for a, b in itertools.combinations(flat, 2)])
    rows.append({"layer": l, "mean_tv_norm": float(norm), "rank50_frac": float(np.mean(er)), "mean_pair_cos": float(cos)})
    print(rows[-1], flush=True)

def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

depth = np.arange(NL)
s_hand = 0.6 + 2.4 * depth / (NL - 1)
a_hand = 0.7 - 0.4 * depth / (NL - 1)
out = {"rows": rows, "spearman_vs_s_hand": {}, "spearman_vs_alpha_hand": {}}
for key in ["mean_tv_norm", "rank50_frac", "mean_pair_cos"]:
    v = [r[key] for r in rows]
    out["spearman_vs_s_hand"][key] = spearman(v, s_hand)
    out["spearman_vs_alpha_hand"][key] = spearman(v, a_hand)
print(json.dumps({k: out[k] for k in out if k != "rows"}, indent=1))
json.dump(out, open("results/layer_stats.json", "w"), indent=1)
