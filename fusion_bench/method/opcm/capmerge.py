"""
CAP/OAP-Merge on top of OPCM (NeurIPS 2025). Data-free, training-free, constant memory.
alpha_schedule:
  'const'   -> exactly OPCM
  'linear'  -> alpha changes linearly with depth (alpha_first -> alpha_last)
  'overlap' -> OAP: per-layer, per-step alpha from the layer's overlap with the merged update:
               alpha_l = clip(alpha + beta * (overlap_l / mean_overlap_prev_step - 1), alpha_min, alpha_max)
target_retention=rho>0 -> never remove more than (1-rho) of a new task's energy
"""
import re
from pathlib import Path
import torch
from torch import Tensor
from fusion_bench.utils.json import save_to_json
from .opcm import OPCMForCLIP
from .utils import svd


class CAPMergeForCLIP(OPCMForCLIP):
    def __init__(self, alpha: float, alpha_schedule: str = "const", alpha_first: float = 0.5,
                 alpha_last: float = 0.5, num_layers: int = 12, target_retention: float = 0.0,
                 overlap_beta: float = 0.5, alpha_min: float = 0.2, alpha_max: float = 0.8,
                 log_diagnostics: bool = True, **kwargs):
        self.alpha_schedule = alpha_schedule
        self.alpha_first, self.alpha_last, self.num_layers = alpha_first, alpha_last, num_layers
        self.target_retention = target_retention
        self.overlap_beta, self.alpha_min, self.alpha_max = overlap_beta, alpha_min, alpha_max
        self.log_diagnostics = log_diagnostics
        self.diagnostics = []
        self._ov_step, self._ov_cur, self._ov_prev_mean = None, [], None
        super().__init__(alpha=alpha, **kwargs)

    @torch.no_grad()
    def run(self, modelpool):
        merged_model = super().run(modelpool)
        if self.log_diagnostics and self.log_dir is not None:
            save_to_json(self.diagnostics, Path(self.log_dir) / "diagnostics.json")
        return merged_model

    def _layer_alpha(self, param_name, alpha, overlap=None):
        if self.alpha_schedule == "linear":
            L = re.findall(r"layers\.(\d+)\.", param_name)
            if not L:
                return alpha
            f = int(L[0]) / max(self.num_layers - 1, 1)
            return self.alpha_first + f * (self.alpha_last - self.alpha_first)
        if self.alpha_schedule == "overlap":
            if overlap is None or not self._ov_prev_mean:
                return alpha
            a = alpha + self.overlap_beta * (overlap / self._ov_prev_mean - 1.0)
            return float(min(max(a, self.alpha_min), self.alpha_max))
        return alpha

    def merge_linear_weights(self, merged_W: Tensor, pretrained_W: Tensor, task_W: Tensor,
                             param_name: str, alpha: float, accelerator: str = "cpu"):
        dev = merged_W.device
        merged_W, pretrained_W, task_W = merged_W.to(accelerator), pretrained_W.to(accelerator), task_W.to(accelerator)
        step = len(self.all_task_vector_norm) - 1
        if self.alpha_schedule == "overlap" and step != self._ov_step:
            if self._ov_cur:
                self._ov_prev_mean = sum(self._ov_cur) / len(self._ov_cur)
            self._ov_cur, self._ov_step = [], step

        prev = merged_W - pretrained_W
        task_tv = task_W - pretrained_W
        u, s, v = svd(prev)
        cum = s.cumsum(dim=0) / s.sum()
        P = u.T @ task_tv @ v
        total_e = task_tv.pow(2).sum()

        overlap = None
        if self.alpha_schedule == "overlap":
            k0 = (cum > alpha).float().argmax().item()
            overlap = float(P[:k0, :k0].pow(2).sum() / (total_e + 1e-12))
            self._ov_cur.append(overlap)

        a = self._layer_alpha(param_name, alpha, overlap)
        split_rank = (cum > a).float().argmax().item()
        rank_alpha = split_rank

        if self.target_retention > 0 and split_rank > 0:
            removed_k = P[:split_rank, :split_rank].pow(2).cumsum(0).cumsum(1).diagonal()
            ok = (removed_k <= (1 - self.target_retention) * total_e).nonzero()
            split_rank = int(ok.max().item()) + 1 if ok.numel() > 0 else 0

        removed_e = P[:split_rank, :split_rank].pow(2).sum()
        P.diag().fill_(0)                       # kept verbatim from OPCM
        P[:split_rank, :split_rank] = 0
        cleaned = u @ P @ v.T

        if self.log_diagnostics:
            eps = 1e-12
            self.diagnostics.append({
                "step": step, "param": param_name, "alpha_used": float(a),
                "overlap": overlap, "rank_alpha": int(rank_alpha), "rank_used": int(split_rank),
                "retained_energy": float(cleaned.pow(2).sum() / (total_e + eps)),
                "removed_fraction": float(removed_e / (total_e + eps)),
            })

        new_W = pretrained_W + (self.previous_lambda_t * prev + cleaned) / self.lambda_t
        return new_W.to(dev)
