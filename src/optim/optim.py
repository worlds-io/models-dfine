"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ..core import register

__all__ = ["AdamW", "SGD", "Adam", "MultiStepLR", "CosineAnnealingLR", "OneCycleLR", "LambdaLR"]


def _params_are_cuda(params) -> bool:
    """True iff every tensor in the param groups is on CUDA *and* they all share a single
    device. Fused optimizers require same-device params; a model sharded across cuda:0 and
    cuda:1 must opt out."""
    params = list(params)
    if not params:
        return False
    tensors = []
    for p in params:
        if isinstance(p, dict):
            tensors.extend(p.get("params", []))
        else:
            tensors.append(p)
    if not tensors or not all(isinstance(t, torch.Tensor) and t.is_cuda for t in tensors):
        return False
    return len({t.device for t in tensors}) == 1


class Adam(optim.Adam):
    """Adam that defaults to fused=True when all params are on CUDA. Fused optimizers collapse
    N per-parameter update kernels into one, which matters because kernel-launch dispatch is a
    dominant per-iter cost on transformer models like D-FINE (Nsight profiling shows ~9.5k
    kernel launches per training iter, median 15us gap between them).
    """

    def __init__(self, params, *args, fused=None, **kwargs):
        params = list(params)
        if fused is None:
            fused = _params_are_cuda(params)
        super().__init__(params, *args, fused=fused, **kwargs)


class AdamW(optim.AdamW):
    """AdamW that defaults to fused=True when all params are on CUDA. See Adam above."""

    def __init__(self, params, *args, fused=None, **kwargs):
        params = list(params)
        if fused is None:
            fused = _params_are_cuda(params)
        super().__init__(params, *args, fused=fused, **kwargs)


SGD = register()(optim.SGD)
Adam = register()(Adam)
AdamW = register()(AdamW)


MultiStepLR = register()(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)
