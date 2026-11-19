import json, glob, os
root = os.path.expanduser("~/alekh_opcm/opcm/outputs/cmm")
out = os.path.expanduser("~/alekh_opcm/opcm/results")
p2, rep = {}, {}
for d in sorted(glob.glob(root + "/*/version_*")):
    job = os.path.basename(os.path.dirname(d))
    f = os.path.join(d, "p2_results.json")
    if os.path.exists(f):
        p2[job] = json.load(open(f))
    f = os.path.join(d, "report_7.json")
    if os.path.exists(f):
        rep[job] = json.load(open(f))
json.dump(p2, open(out + "/all_p2_results.json", "w"), indent=1)
json.dump(rep, open(out + "/all_reports.json", "w"), indent=1)
print(len(p2), "memory jobs;", len(rep), "merge reports")
