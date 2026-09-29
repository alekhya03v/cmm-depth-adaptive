"""
SyMerge: synergy-aware continual merging built on OPCM (NeurIPS 2025).
keep_synergy=False -> exactly OPCM + logs flaw diagnostics (SDR, retained energy, per-layer removal).
"""
from pathlib import Path
import torch
from torch import Tensor
from fusion_bench.utils.json import save_to_json
from .opcm import OPCMForCLIP
from .utils import svd


class SyMergeForCLIP(OPCMForCLIP):
    def __init__(self, alpha: float, keep_synergy: bool = True, gamma: float = 1.0,
                 margin: float = 0.0, log_diagnostics: bool = True, **kwargs):
        self.keep_synergy = keep_synergy
        self.gamma = gamma
        self.margin = margin
        self.log_diagnostics = log_diagnostics
        self.diagnostics = []
        super().__init__(alpha=alpha, **kwargs)

    @torch.no_grad()
    def run(self, modelpool):
        merged_model = super().run(modelpool)
        if self.log_diagnostics and self.log_dir is not None:
            save_to_json(self.diagnostics, Path(self.log_dir) / "diagnostics.json")
        return merged_model

    def merge_linear_weights(self, merged_W: Tensor, pretrained_W: Tensor, task_W: Tensor,
                             param_name: str, alpha: float, accelerator: str = "cpu"):
        original_device = merged_W.device
        merged_W = merged_W.to(accelerator)
        pretrained_W = pretrained_W.to(accelerator)
        task_W = task_W.to(accelerator)

        previous_merged_tv = merged_W - pretrained_W
        task_tv = task_W - pretrained_W

        u, s, v = svd(previous_merged_tv)
        split_rank = (s.cumsum(dim=0) / s.sum() > alpha).float().argmax().item()

        projected_task_tv = u.T @ task_tv @ v

        # diagnostics: split what OPCM removes into synergy / conflict / cross terms
        block = projected_task_tv[:split_rank, :split_rank].clone()
        coeff = block.diagonal().clone()
        thr = self.margin * coeff.abs().max() if coeff.numel() > 0 else 0.0
        syn_mask = coeff > thr
        removed_e = block.pow(2).sum()
        syn_e = coeff[syn_mask].pow(2).sum()
        conf_e = coeff[coeff < 0].pow(2).sum()
        cross_e = removed_e - coeff.pow(2).sum()
        total_e = task_tv.pow(2).sum()

        # OPCM projection (verbatim)
        projected_task_tv.diag().fill_(0)
        projected_task_tv[:split_rank, :split_rank] = 0

        # SyMerge: put back synergistic directions
        if self.keep_synergy and split_rank > 0:
            idx = torch.arange(split_rank, device=projected_task_tv.device)
            projected_task_tv[idx, idx] = self.gamma * coeff * syn_mask

        cleaned_task_tv = u @ projected_task_tv @ v.T
        kept_e = cleaned_task_tv.pow(2).sum()

        if self.log_diagnostics:
            eps = 1e-12
            self.diagnostics.append({
                "step": len(self.all_task_vector_norm) - 1,
                "param": param_name,
                "split_rank": int(split_rank),
                "full_rank": int(s.numel()),
                "SDR": float(syn_e / (removed_e + eps)),
                "conflict_ratio": float(conf_e / (removed_e + eps)),
                "cross_ratio": float(cross_e / (removed_e + eps)),
                "retained_energy": float(kept_e / (total_e + eps)),
                "removed_fraction": float(removed_e / (total_e + eps)),
                "n_synergy_dirs": int(syn_mask.sum().item()),
            })

        new_merged_W = (pretrained_W
                        + (self.previous_lambda_t * previous_merged_tv + cleaned_task_tv)
                        / self.lambda_t)
        return new_merged_W.to(original_device)
