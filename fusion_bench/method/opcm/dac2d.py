"""
DAC-2D: strength schedule over BOTH depth and task arrival order (endsem extension #2).
    s(l, t) = s_layer(l) * g(t),   g(t) = 1 + gamma * (t / (T-1) - 0.5)
t = index of the task being merged (1..T-1), T = num_tasks. gamma = 0 -> exactly DAC-Merge.
gamma > 0: later tasks get more strength; gamma < 0: earlier tasks get more.
"""
from torch import Tensor
from .dacmerge import DACMergeForCLIP


class DAC2DForCLIP(DACMergeForCLIP):
    def __init__(self, gamma: float = 0.0, num_tasks: int = 8, **kwargs):
        self.gamma = gamma
        self.num_tasks = num_tasks
        super().__init__(**kwargs)

    def _time_gain(self) -> float:
        t = len(self.all_task_vector_norm) - 1          # index of the task being merged
        return 1.0 + self.gamma * (t / max(self.num_tasks - 1, 1) - 0.5)

    def merge_linear_weights(self, merged_W: Tensor, pretrained_W: Tensor, task_W: Tensor,
                             param_name: str, alpha: float, accelerator: str = "cpu"):
        g = self._time_gain()
        sf, sl = self.scale_first, self.scale_last
        self.scale_first, self.scale_last = sf * g, sl * g
        try:
            return super().merge_linear_weights(merged_W, pretrained_W, task_W,
                                                param_name, alpha, accelerator)
        finally:
            self.scale_first, self.scale_last = sf, sl
