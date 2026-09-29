"""
DAC-Merge: Depth-Aware Continual Merging (built on OPCM, NeurIPS 2025).
Training-free, data-free, constant memory. Two depth-dependent knobs:
  protection: alpha(l) = alpha_first -> alpha_last   (projection threshold per layer)
  strength:   s(l)     = scale_first -> scale_last   (scale of the new task's update per layer)
With alpha_first = alpha_last = alpha and scale_first = scale_last = 1.0 it is exactly OPCM.
"""
import re
from pathlib import Path
import torch
from torch import Tensor
from fusion_bench.utils.json import save_to_json
from .opcm import OPCMForCLIP
from .utils import svd


class DACMergeForCLIP(OPCMForCLIP):
    def __init__(self, alpha: float, alpha_first: float = 0.5, alpha_last: float = 0.5,
                 scale_first: float = 1.0, scale_last: float = 1.0, num_layers: int = 12,
                 log_diagnostics: bool = True, **kwargs):
        self.alpha_first, self.alpha_last = alpha_first, alpha_last
        self.scale_first, self.scale_last = scale_first, scale_last
        self.num_layers = num_layers
        self.log_diagnostics = log_diagnostics
        self.diagnostics = []
        super().__init__(alpha=alpha, **kwargs)

    @torch.no_grad()
    def run(self, modelpool):
        merged_model = super().run(modelpool)
        if self.log_diagnostics and self.log_dir is not None:
            save_to_json(self.diagnostics, Path(self.log_dir) / "diagnostics.json")
        return merged_model

    def _depth(self, param_name):
        L = re.findall(r"layers\.(\d+)\.", param_name)
        return None if not L else int(L[0]) / max(self.num_layers - 1, 1)

    def merge_linear_weights(self, merged_W: Tensor, pretrained_W: Tensor, task_W: Tensor,
                             param_name: str, alpha: float, accelerator: str = "cpu"):
        dev = merged_W.device
        merged_W, pretrained_W, task_W = merged_W.to(accelerator), pretrained_W.to(accelerator), task_W.to(accelerator)
        prev = merged_W - pretrained_W
        task_tv = task_W - pretrained_W
        f = self._depth(param_name)

        a = alpha if f is None else self.alpha_first + f * (self.alpha_last - self.alpha_first)
        sc = 1.0 if f is None else self.scale_first + f * (self.scale_last - self.scale_first)

        u, s, v = svd(prev)
        split_rank = (s.cumsum(dim=0) / s.sum() > a).float().argmax().item()
        P = u.T @ task_tv @ v
        total_e = task_tv.pow(2).sum()
        removed_e = P[:split_rank, :split_rank].pow(2).sum()
        P.diag().fill_(0)                       # kept verbatim from OPCM
        P[:split_rank, :split_rank] = 0
        cleaned = (u @ P @ v.T) * sc

        if self.log_diagnostics:
            eps = 1e-12
            self.diagnostics.append({
                "step": len(self.all_task_vector_norm) - 1, "param": param_name,
                "alpha_used": float(a), "scale_used": float(sc), "rank_used": int(split_rank),
                "removed_fraction": float(removed_e / (total_e + eps)),
            })

        new_W = pretrained_W + (self.previous_lambda_t * prev + cleaned) / self.lambda_t
        return new_W.to(dev)
