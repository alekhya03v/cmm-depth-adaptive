"""
DAC-Budget: OPCM's norm-matching rescale, applied per group instead of globally.
budget = 'global' -> exactly OPCM's rescaling
budget = 'layer'  -> one norm budget per transformer layer (+ one for all other params)
budget = 'param'  -> one norm budget per parameter tensor
Keeps DAC-Merge's depth-aware protection (alpha_first/last) and strength (scale_first/last).
"""
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from fusion_bench.utils.json import save_to_json
from . import opcm as _o
from .dacmerge import DACMergeForCLIP


class DACBudgetForCLIP(DACMergeForCLIP):
    def __init__(self, alpha: float, budget: str = "global", **kwargs):
        self.budget = budget
        super().__init__(alpha=alpha, **kwargs)

    def _group(self, name):
        if self.budget == "global":
            return "all"
        if self.budget == "param":
            return name
        m = re.search(r"layers\.(\d+)\.", name)
        return f"L{m.group(1)}" if m else "other"

    def _group_sq(self, model, pre):
        d = defaultdict(float)
        for n, p in model.named_parameters():
            d[self._group(n)] += (p - pre[n]).pow(2).sum().item()
        return d

    @torch.no_grad()
    def run(self, modelpool):
        if self.seed is not None:
            _o.L.seed_everything(self.seed)
        acc = self.fabric.device
        pretrained_model = modelpool.load_pretrained_model()
        model_names = modelpool.model_names
        if self.shuffle_order:
            _o.random.shuffle(model_names)
        self.taskpool = self._program.taskpool
        self._test_datasets = _o.deepcopy(self.taskpool._test_datasets)
        if self.log_dir is not None:
            save_to_json(model_names, Path(self.log_dir) / "model_names.json")

        merged_model = modelpool.load_model(model_names[0])
        if self.evaluate_on_every_step:
            self.taskpool._is_setup = False
            self.taskpool._test_datasets = _o.DictConfig({model_names[0]: self._test_datasets[model_names[0]]})
            save_to_json(self.taskpool.evaluate(_o.deepcopy(merged_model)), Path(self.log_dir) / "report_0.json")

        pre = dict(pretrained_model.named_parameters())
        task_sq = [self._group_sq(merged_model, pre)]
        self.all_task_vector_norm = [0.0]
        prev_lam = defaultdict(lambda: 1.0)

        for model_idx, model_name in enumerate(model_names[1:], start=1):
            task_model = modelpool.load_model(model_name)
            task_sq.append(self._group_sq(task_model, pre))
            self.all_task_vector_norm.append(0.0)
            avg = {g: float(np.mean([np.sqrt(t[g]) for t in task_sq])) for g in task_sq[0]}
            self.lambda_t = 1

            for module_name, module in list(merged_model.named_modules()):
                if not _o.is_leaf_module(module):
                    continue
                pm, tm = pretrained_model.get_submodule(module_name), task_model.get_submodule(module_name)
                if isinstance(module, _o.nn.Linear):
                    wn = f"{module_name}.weight"
                    self.previous_lambda_t = prev_lam[self._group(wn)]
                    module.weight.data = self.merge_linear_weights(
                        module.weight, pm.weight, tm.weight, param_name=wn, alpha=self.alpha, accelerator=acc)
                    if module.bias is not None:
                        bn = f"{module_name}.bias"
                        self.previous_lambda_t = prev_lam[self._group(bn)]
                        module.bias.data = self.merge_other_parameters(
                            module.bias, pm.bias, tm.bias, param_name=bn, accelerator=acc)
                else:
                    for pn, param in module.named_parameters():
                        full = f"{module_name}.{pn}"
                        self.previous_lambda_t = prev_lam[self._group(full)]
                        param.data = self.merge_other_parameters(
                            merged_W=param, pretrained_W=pm.get_parameter(pn), task_W=tm.get_parameter(pn),
                            param_name=full, accelerator=acc)

            merged_sq = self._group_sq(merged_model, pre)
            new_lam = {g: (np.sqrt(v) / avg[g] if avg.get(g, 0) > 0 else 1.0) for g, v in merged_sq.items()}
            for n, p in merged_model.named_parameters():
                lam = new_lam[self._group(n)]
                if lam > 0:
                    p.data = pre[n] + (p - pre[n]) / lam
            prev_lam = defaultdict(lambda: 1.0, new_lam)
            self.lambda_t = None

            if self.evaluate_on_every_step:
                self.taskpool._is_setup = False
                self.taskpool._test_datasets = _o.DictConfig({n: self._test_datasets[n] for n in model_names[: model_idx + 1]})
                save_to_json(self.taskpool.evaluate(_o.deepcopy(merged_model)), Path(self.log_dir) / f"report_{model_idx}.json")

        if self.log_diagnostics and self.log_dir is not None:
            save_to_json(self.diagnostics, Path(self.log_dir) / "diagnostics.json")
        return merged_model
