"""
DAC-Soft: DAC-Merge with a SOFT projection mask (endsem extension).
Keep-weight of direction i: w_i = sigmoid((r - i - 0.5) / (softness * r)); the coefficient
P_ij is scaled by (1 - w_i * w_j). softness = 0 -> exactly DAC-Merge (hard block cut).
"""
import torch
from torch import Tensor
from .dacmerge import DACMergeForCLIP
from .utils import svd


def soft_keep_mask(split_rank: int, nrows: int, ncols: int, softness: float, device) -> Tensor:
    def keep(n):
        idx = torch.arange(n, device=device, dtype=torch.float32)
        if softness <= 0:
            return (idx < split_rank).float()
        t = max(softness * max(split_rank, 1), 1e-6)
        return torch.sigmoid((split_rank - idx - 0.5) / t)
    return 1.0 - torch.outer(keep(nrows), keep(ncols))


class DACSoftForCLIP(DACMergeForCLIP):
    def __init__(self, softness: float = 0.25, **kwargs):
        self.softness = softness
        super().__init__(**kwargs)

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
        mask = soft_keep_mask(split_rank, P.shape[0], P.shape[1], self.softness, P.device)
        total_e = task_tv.pow(2).sum()
        removed_e = (P.pow(2) * (1.0 - mask)).sum()
        P.diag().fill_(0)                       # as in OPCM / DAC-Merge (a no-op in PyTorch)
        P = P * mask
        cleaned = (u @ P @ v.T) * sc

        if self.log_diagnostics:
            self.diagnostics.append({
                "step": len(self.all_task_vector_norm) - 1, "param": param_name,
                "alpha_used": float(a), "scale_used": float(sc), "rank_used": int(split_rank),
                "softness": float(self.softness),
                "removed_fraction": float(removed_e / (total_e + 1e-12)),
            })
        new_W = pretrained_W + (self.previous_lambda_t * prev + cleaned) / self.lambda_t
        return new_W.to(dev)
