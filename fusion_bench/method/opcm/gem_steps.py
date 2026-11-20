"""GEM accuracy after every merge step (task-aware): the memory of every task seen so far is applied
to the model merged so far, and each seen task is tested with its own memory."""
import json
from copy import deepcopy
from pathlib import Path
import torch
from . import opcm as _o
from .dacmerge import DACMergeForCLIP
from .p2mem import P2MemForCLIP


class GemStepsForCLIP(P2MemForCLIP):
    @torch.no_grad()
    def run(self, modelpool):
        pre_model = modelpool.load_pretrained_model()
        self._pre = {n: p.detach().float().cpu() for n, p in pre_model.named_parameters()}
        del pre_model
        self._steps = {}
        return DACMergeForCLIP.run(self, modelpool)   # merge only; skips the final memory phase of P2Mem

    @torch.no_grad()
    def save_merged_model(self, merged_model, step):   # OPCM calls this after every merge step
        names = json.load(open(Path(self.log_dir) / "model_names.json"))
        if step == 0:                                  # the first model is not passed through the merge function
            for n, p in merged_model.named_parameters():
                self._store_res(0, n, self._pre[n], p.detach().cpu())
        parts = self.mem_settings[0].split(":")
        accs = {}
        for i in range(step + 1):
            m = deepcopy(merged_model)
            params = dict(m.named_parameters())
            self._apply(parts, i, params, self._pre)
            self.taskpool._is_setup = False
            self.taskpool._test_datasets = _o.DictConfig({names[i]: self._test_datasets[names[i]]})
            accs[names[i]] = self.taskpool.evaluate(m)[names[i]]["accuracy"]
            del m
        avg = sum(accs.values()) / len(accs)
        self._steps[step + 1] = {"average": avg, "per_task": accs}
        json.dump(self._steps, open(Path(self.log_dir) / "gem_steps.json", "w"))
        print(f"[gemsteps] after {step + 1} task(s): GEM average {avg * 100:.2f}", flush=True)
