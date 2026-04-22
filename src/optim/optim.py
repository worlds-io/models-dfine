"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ..core import register

__all__ = ["AdamW", "SGD", "Adam", "MultiStepLR", "CosineAnnealingLR", "OneCycleLR", "LambdaLR"]


class Adam(optim.Adam):
    """Adam that defaults to fused=True on CUDA. Fused optimizers collapse N per-parameter
    update kernels into a single kernel, which matters because kernel-launch dispatch is a
    dominant per-iter cost on transformer models like D-FINE (Nsight profiling shows ~9.5k
    kernel launches per training iter, median 15us gap between them).
    """

    def __init__(self, params, *args, fused=None, **kwargs):
        if fused is None:
            fused = torch.cuda.is_available()
        super().__init__(params, *args, fused=fused, **kwargs)


class AdamW(optim.AdamW):
    """AdamW that defaults to fused=True on CUDA. See Adam above for rationale."""

    def __init__(self, params, *args, fused=None, **kwargs):
        if fused is None:
            fused = torch.cuda.is_available()
        super().__init__(params, *args, fused=fused, **kwargs)


SGD = register()(optim.SGD)
Adam = register()(Adam)
AdamW = register()(AdamW)


MultiStepLR = register()(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)
