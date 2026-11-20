#!/usr/bin/env python
"""Clean demo wrapper. Usage: python demo_run.py {zeroshot|opcm|dac|gem} [--pause 6]"""
import os, re, sys, time, shutil, subprocess, argparse

BLUE, GREEN, YEL, RED, DIM, BOLD, END = "\033[94m","\033[92m","\033[93m","\033[91m","\033[2m","\033[1m","\033[0m"
NAMES = {"sun397":"SUN397","stanford-cars":"Cars","resisc45":"RESISC45","eurosat":"EuroSAT",
         "svhn":"SVHN","gtsrb":"GTSRB","mnist":"MNIST","dtd":"DTD"}
nm = lambda k: NAMES.get(k, k)

INFO = {
 "zeroshot": dict(
   title="PRETRAINED CLIP (ZERO-SHOT)  -  baseline, nothing is trained or merged",
   what=["No fine-tuning and no merging happens in this run.",
         "We take the original pretrained CLIP ViT-B/32 and test it as it is on all 8 tasks.",
         "It classifies by matching each image to text prompts like 'a photo of a {class}'.",
         "This shows how weak a general model is on specialised tasks (digits, traffic signs)."],
   step=None),
 "opcm": dict(
   title="OPCM  -  base paper (continual merging, one uniform setting for all layers)",
   what=["Is it fine-tuning?  NO. The 8 fine-tuned models were trained earlier, one per task.",
         "Here we only LOAD them one at a time and merge them into ONE model. No training, no data.",
         "Each step does this for every layer:",
         "  1) task vector = fine-tuned weights - pretrained weights",
         "  2) SVD of the current merged update; keep its top directions (energy threshold alpha = 0.5)",
         "  3) remove from the new task vector the part along those directions (protects old tasks)",
         "  4) rescale and add it to the merged model",
         "Same alpha and same strength for ALL 12 layers (the two uniform choices)."],
   step="OPCM"),
 "dac": dict(
   title="DAC-MERGE  -  our method (depth-aware continual merging)",
   what=["Same merging idea as OPCM, still no training, no data, no extra memory.",
         "The difference: the settings now depend on the layer depth.",
         "  protection threshold alpha: 0.7 in early layers -> 0.3 in late layers",
         "  strength s: 1.2 in early layers -> 2.4 in late layers",
         "Early layers are shared by all tasks, so we protect them more.",
         "Late layers are task-specific, so we give them more room for the new task."],
   step="DAC-Merge"),
 "gem": dict(
   title="GEM  -  DAC-Merge + task-aware low-rank memory (Global Energy-Allocated Memory)",
   what=["Step A: the models are merged with DAC-Merge (same as the previous run).",
         "Step B: for each task we also keep a small low-rank copy of what its fine-tuning changed.",
         "GEM chooses WHERE to store it: all directions of all 12 layers are ranked by",
         "  energy per stored number, and we fill a fixed budget (~1.15% of the model per task).",
         "At test time the task id selects that task's memory. Still no data and no training.",
         "Expect higher accuracy than DAC-Merge alone, at about 1% extra memory per task."],
   step="DAC-Merge + GEM memory"),
}

def banner(key, pause):
    d = INFO[key]; w = 100
    print(BLUE + "=" * w + END)
    print(BOLD + d["title"] + END)
    print(BLUE + "=" * w + END)
    for l in d["what"]: print("  " + l)
    print(BLUE + "-" * w + END)
    print(f"  Model: CLIP ViT-B/32   |   Benchmark: 8 tasks   |   Task order: random, seed 0")
    print(BLUE + "=" * w + END, flush=True)
    time.sleep(pause)

def build_cmd(key, name):
    base = ["fabric.loggers.root_dir=outputs/cmm", f"fabric.loggers.name={name}", "fabric.loggers.version=0"]
    pool = "CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only"
    task = "CLIPVisionModelTaskPool/clip-vit-classification_TA8"
    if key == "zeroshot":
        return ["fusion_bench", "method=dummy", "modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8",
                f"taskpool={task}", f"report_save_path=outputs/cmm/{name}.json"]
    if key == "opcm":
        m = ["method=opcm/opcm"]
    elif key == "dac":
        m = ["method=opcm/dacmerge", "method.alpha_first=0.7", "method.alpha_last=0.3",
             "method.scale_first=1.2", "method.scale_last=2.4"]
    else:
        shutil.copyfile("config/method/opcm/p2mem_endsem.yaml", "config/method/opcm/p2mem_demo_gem.yaml")
        os.system("sed -i 's/^mem_settings:.*/mem_settings: \"resEA:4:16\"/' config/method/opcm/p2mem_demo_gem.yaml")
        m = ["method=opcm/p2mem_demo_gem"]
    return (["fusion_bench"] + base + m + ["method.seed=0", "method.save_on_every_step=false",
            f"modelpool={pool}", f"taskpool={task}", "taskpool.base_model=openai/clip-vit-base-patch32"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("key", choices=list(INFO)); ap.add_argument("--pause", type=float, default=6)
    a = ap.parse_args(); key = a.key
    name = f"demo2_B32_TA8_{key}"
    shutil.rmtree(f"outputs/cmm/{name}", ignore_errors=True)
    os.makedirs("logs", exist_ok=True)
    banner(key, a.pause)
    env = dict(os.environ, COLUMNS="300", PYTHONUNBUFFERED="1", NO_COLOR="1", TERM="dumb")
    raw = open(f"logs/{name}_raw.log", "w"); clean = open(f"logs/{name}_clean.txt", "w")
    def say(s):
        print(s, flush=True); clean.write(re.sub(r"\033\[[0-9;]*m", "", s) + "\n"); clean.flush()
    t0 = time.time()
    p = subprocess.Popen(build_cmd(key, name), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    step, first_acc, last_avg, buf, collecting, announced_eval = 0, {}, None, "", False, False
    seen = []
    pending = b""
    say(DIM + "Starting... loading pretrained CLIP and the fine-tuned checkpoints (first load can take a minute)." + END)
    while True:
        chunk = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(1)
        if not chunk: break
        pending += chunk
        parts = re.split(rb"[\r\n]", pending); pending = parts.pop()
        for raw_line in parts:
            line = raw_line.decode("utf-8", "ignore"); raw.write(line + "\n")
            line = re.sub(r"\s+", " ", re.sub(r"[│╭╮╰╯─]", " ", line)).strip()
            if not line: continue
            m = re.search(r"Loading `transformers.CLIPVisionModel`:\s*(\S+)", line)
            if m and key != "zeroshot":
                mm = re.search(r"patch\d+_(.+)$", m.group(1))
                if mm:
                    step += 1; announced_eval = False; seen.append(mm.group(1))
                    t = nm(mm.group(1))
                    say("")
                    say(BOLD + YEL + f"STEP {step}/8  -  new task arrives: {t}" + END)
                    if step == 1:
                        say(f"  The first fine-tuned model ({t}) becomes the starting merged model. Nothing to merge yet.")
                    else:
                        say(f"  Loading the {t} fine-tuned model, computing its task vector,")
                        say(f"  projecting it away from the earlier tasks' directions, rescaling and adding it into the merged model.")
                    if key == "gem" and step == 1:
                        say("  (memory for each task is built from its task vector and stored in the final phase)")
                continue
            if line.startswith("Evaluating tasks") and not announced_eval and key != "zeroshot":
                announced_eval = True
                say(DIM + f"  Testing the merged model on the {len(seen)} task(s) seen so far..." + END)
                continue
            if key == "zeroshot" and line.startswith("Evaluating tasks") and not announced_eval:
                announced_eval = True; say(DIM + "Testing pretrained CLIP on the 8 tasks..." + END); continue
            if "Evaluation Result" in line:
                collecting, buf = True, ""
            if collecting:
                buf += " " + line
                if re.search(r"'average':\s*\{\s*'accuracy':\s*[0-9.eE-]+", buf):
                    collecting = False
                    pairs = re.findall(r"'([^']+)':\s*\{\s*'accuracy':\s*([0-9.eE-]+)", buf)
                    accs = {k: float(v) * 100 for k, v in pairs}
                    avg = accs.pop("average", None)
                    say(GREEN + f"  Accuracy after step {step if key!='zeroshot' else 0}:" + END)
                    for k, v in accs.items():
                        first_acc.setdefault(k, v)
                        drop = v - first_acc[k]
                        tag = "" if abs(drop) < 0.05 else (f"  ({drop:+.1f} since it was added)" if key != "zeroshot" else "")
                        say(f"     {nm(k):10s} {v:6.2f}{tag}")
                    d = "" if last_avg is None else f"  ({avg-last_avg:+.2f} vs previous step)"
                    say(BOLD + GREEN + f"     AVERAGE    {avg:6.2f}{d}" + END)
                    last_avg = avg
                continue
            if key == "gem" and re.search(r"memory|resEA|p2_results|rank", line, re.I) and "Loading" not in line:
                say(DIM + "  [memory] " + line[:160] + END)
            if re.search(r"Traceback|Error", line):
                say(RED + "  !! " + line[:200] + END)
    p.wait()
    say("")
    label = {"zeroshot": "PRETRAINED CLIP (ZERO-SHOT)", "opcm": "OPCM", "dac": "DAC-MERGE", "gem": "GEM"}[key]
    mins = (time.time() - t0) / 60
    if last_avg is not None:
        say("")
        say(GREEN + BOLD + "=" * 100 + END)
        say(GREEN + BOLD + f"  \u2714  {label} DONE" + END)
        say(GREEN + BOLD + f"  Final average accuracy on 8 tasks: {last_avg:.2f}%     |     time taken: {mins:.1f} min" + END)
        say(GREEN + BOLD + "=" * 100 + END)
    else:
        say(RED + BOLD + f"  \u2718  {label} FINISHED WITHOUT A RESULT - see logs/{name}_raw.log" + END)
    say(DIM + f"Full raw log: logs/{name}_raw.log   |   clean transcript: logs/{name}_clean.txt" + END)

main()
