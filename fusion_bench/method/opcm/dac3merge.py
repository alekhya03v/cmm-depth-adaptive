"""
DAC-Merge v3 = DAC-Merge (depth-aware protection + strength) + two new options:
  proj_mode (Idea A, WHAT is removed):
     "full"        -> OPCM/DAC default: remove the whole overlapping block
     "keepcross"   -> remove only the aligned (diagonal) part, keep cross terms
     "keepcross_h" -> remove the aligned part, keep half of the cross terms
  ln_mode (Idea B, WHICH parameters):
     "opcm"        -> default averaging rule for LayerNorm parameters
     "pre"         -> LayerNorm parameters stay at pretrained values
     "half"        -> LayerNorm parameters get half of the task change
With proj_mode="full" and ln_mode="opcm" it is exactly DAC-Merge.
"""
import re
import torch
from torch import Tensor
from .dacmerge import DACMergeForCLIP
from .utils import svd


class DAC3MergeForCLIP(DACMergeForCLIP):
    def __init__(self, alpha: float, proj_mode: str = "full", ln_mode: str = "opcm", **kwargs):
        self.proj_mode, self.ln_mode = proj_mode, ln_mode
        super().__init__(alpha=alpha, **kwargs)

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
        k = (s.cumsum(dim=0) / s.sum() > a).float().argmax().item()
        P = u.T @ task_tv @ v
        if k > 0:
            if self.proj_mode == "full":
                P[:k, :k] = 0
            else:
                keep = 1.0 if self.proj_mode == "keepcross" else 0.5
                idx = torch.arange(k, device=P.device)
                P[:k, :k] = P[:k, :k] * keep
                P[idx, idx] = 0
        cleaned = (u @ P @ v.T) * sc
        new_W = pretrained_W + (self.previous_lambda_t * prev + cleaned) / self.lambda_t
        return new_W.to(dev)

    def merge_other_parameters(self, merged_W, pretrained_W, task_W, param_name, accelerator="cpu"):
        if self.ln_mode != "opcm" and re.search(r"norm", param_name):
            if self.ln_mode == "pre":
                return pretrained_W.data.clone()
            task_W = pretrained_W + 0.5 * (task_W - pretrained_W)
        return super().merge_other_parameters(merged_W, pretrained_W, task_W, param_name, accelerator)
