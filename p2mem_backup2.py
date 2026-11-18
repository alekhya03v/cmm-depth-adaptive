"""DAC-Merge + per-task memory (task-aware). Settings: none | res:k:r | dis:r:g (ours) | tall:lam"""
import re, json
from copy import deepcopy
from pathlib import Path
import torch
from fusion_bench.utils.json import save_to_json
from . import opcm as _o
from .dacmerge import DACMergeForCLIP
from .utils import svd


class P2MemForCLIP(DACMergeForCLIP):
    def __init__(self, alpha: float, mem_settings: str = "none", res_k_max: int = 4,
                 max_rank: int = 64, tall_lams: str = "0.2,0.4", **kwargs):
        self.mem_settings = [s.strip() for s in str(mem_settings).split(",")]
        self.res_k_max, self.max_rank = res_k_max, max_rank
        self.tall_lams = [float(x) for x in str(tall_lams).split(",")]
        self._res, self._dis, self._tall = {}, {}, {}
        super().__init__(alpha=alpha, **kwargs)

    def _layer(self, name):
        m = re.search(r"layers\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    def _lowrank(self, d):
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)
        r = min(self.max_rank, S.numel())
        return U[:, :r].clone(), S[:r].clone(), Vh[:r].clone()

    def _store_res(self, idx, name, pre, task):
        L = self._layer(name)
        if L is None or L < self.num_layers - self.res_k_max:
            return
        d = (task.detach().float() - pre.detach().float()).cpu()
        if d.dim() == 2:
            self._res.setdefault(idx, {})[name] = ("lr",) + self._lowrank(d) + (d,)
        else:
            self._res.setdefault(idx, {})[name] = ("full", d)

    def merge_linear_weights(self, merged_W, pretrained_W, task_W, param_name, alpha, accelerator="cpu"):
        idx = len(self.all_task_vector_norm) - 1
        self._store_res(idx, param_name, pretrained_W, task_W)
        prev = (merged_W - pretrained_W).detach().float().to(accelerator)
        tv = (task_W - pretrained_W).detach().float().to(accelerator)
        for lam in self.tall_lams:
            mk = (tv.abs() > (prev - tv).abs() * lam).cpu()
            self._tall.setdefault(lam, {}).setdefault(idx, {})[param_name] = mk
        f = self._depth(param_name)
        a = alpha if f is None else self.alpha_first + f * (self.alpha_last - self.alpha_first)
        u, s, v = svd(prev)
        k = (s.cumsum(dim=0) / s.sum() > a).float().argmax().item()
        if k > 0:
            uk, vk = u[:, :k], v[:, :k]
            removed = (uk @ (uk.T @ tv @ vk) @ vk.T).cpu()
            self._dis.setdefault(idx, {})[param_name] = self._lowrank(removed)
        return super().merge_linear_weights(merged_W, pretrained_W, task_W, param_name, alpha, accelerator)

    def merge_other_parameters(self, merged_W, pretrained_W, task_W, param_name, accelerator="cpu"):
        self._store_res(len(self.all_task_vector_norm) - 1, param_name, pretrained_W, task_W)
        return super().merge_other_parameters(merged_W, pretrained_W, task_W, param_name, accelerator)

    def _apply(self, parts, i, params, pre):
        mem = 0
        if parts[0] == "res":
            k, r = int(parts[1]), int(parts[2])
            for n, e in self._res.get(i, {}).items():
                if self._layer(n) < self.num_layers - k:
                    continue
                if e[0] == "lr":
                    U, S, Vh, full = e[1], e[2], e[3], e[4]
                    if r == 0:
                        delta = full
                        mem += full.numel()
                    else:
                        delta = (U[:, :r] * S[:r]) @ Vh[:r]
                        mem += r * (full.shape[0] + full.shape[1] + 1)
                else:
                    delta = e[1]
                    mem += delta.numel()
                params[n].data = (pre[n] + delta).to(params[n].device, params[n].dtype)
        elif parts[0] in ("resE", "resA", "resEA"):
            k, r = int(parts[1]), int(parts[2])
            rk = self._alloc(i, parts[0][3:], k, r)
            lay = {self._layer(n) for n, v in rk.items() if v > 0}
            for n, e in self._res.get(i, {}).items():
                if e[0] == "lr" and rk.get(n, 0) > 0:
                    U, S, Vh, q = e[1], e[2], e[3], rk[n]
                    delta = (U[:, :q] * S[:q]) @ Vh[:q]
                    mem += q * (U.shape[0] + Vh.shape[1] + 1)
                elif e[0] == "full" and self._layer(n) in lay:
                    delta = e[1]
                    mem += delta.numel()
                else:
                    continue
                params[n].data = (pre[n] + delta).to(params[n].device, params[n].dtype)
        elif parts[0] == "dis":
            r, g = int(parts[1]), float(parts[2])
            for n, (U, S, Vh) in self._dis.get(i, {}).items():
                rr = min(r, S.numel())
                delta = (U[:, :rr] * S[:rr]) @ Vh[:rr]
                mem += rr * (U.shape[0] + Vh.shape[1] + 1)
                params[n].data = params[n].data + (g * delta).to(params[n].device, params[n].dtype)
        elif parts[0] == "tall":
            lam = float(parts[1])
            for n, mk in self._tall.get(lam, {}).get(i, {}).items():
                mem += mk.numel() / 32
                cur = params[n].data.float().cpu()
                new = pre[n] + mk.float() * (cur - pre[n])
                params[n].data = new.to(params[n].device, params[n].dtype)
        return mem

    def _alloc(self, i, mode, k, r):
        ents = {n: e for n, e in self._res.get(i, {}).items() if e[0] == "lr"}
        last = {n for n in ents if self._layer(n) >= self.num_layers - k}
        c = lambda e: e[1].shape[0] + e[3].shape[1] + 1
        budget = sum(r * c(ents[n]) for n in last)
        if mode == "E":
            sc = {}
            for n, e in ents.items():
                sc[self._layer(n)] = sc.get(self._layer(n), 0) + float((e[2][:r] ** 2).sum())
            top = set(sorted(sc, key=sc.get, reverse=True)[:k])
            return {n: min(r, ents[n][2].numel()) for n in ents if self._layer(n) in top}
        pool = ents if mode == "EA" else {n: ents[n] for n in last}
        items = [(float(s) ** 2 / c(e), n, c(e)) for n, e in pool.items() for s in e[2]]
        items.sort(reverse=True)
        rk, used = {}, 0
        for g, n, cc in items:
            if used + cc > budget:
                continue
            rk[n] = rk.get(n, 0) + 1
            used += cc
        return rk

    @torch.no_grad()
    def run(self, modelpool):
        merged = super().run(modelpool)
        names = json.load(open(Path(self.log_dir) / "model_names.json"))
        pre_model = modelpool.load_pretrained_model()
        pre = {n: p.detach().float().cpu() for n, p in pre_model.named_parameters()}
        first = modelpool.load_model(names[0])
        for n, p in first.named_parameters():
            self._store_res(0, n, pre[n], p.cpu())
        del first
        total = sum(p.numel() for p in merged.parameters())
        results = {}
        for setting in self.mem_settings:
            parts = setting.split(":")
            per, mem = {}, 0
            for i, t in enumerate(names):
                m = deepcopy(merged)
                params = dict(m.named_parameters())
                mem += self._apply(parts, i, params, pre)
                self.taskpool._is_setup = False
                self.taskpool._test_datasets = _o.DictConfig({t: self._test_datasets[t]})
                per[t] = self.taskpool.evaluate(m)[t]["accuracy"]
                del m
            avg = sum(per.values()) / len(per)
            pct = 100.0 * mem / len(names) / total
            results[setting] = {"average": avg, "per_task": per, "memory_per_task_pct": pct}
            save_to_json(results, Path(self.log_dir) / "p2_results.json")
            print(f"[p2mem] {setting}: ACC {avg*100:.2f}  memory/task {pct:.2f}%", flush=True)
        return merged
