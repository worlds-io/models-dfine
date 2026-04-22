"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import sys
from typing import Dict, Iterable, List, Tuple

import torch
import torch.amp
from torch.amp import GradScaler
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes


class DataLoaderIterator:
    """Wraps a DataLoader in an iterator that auto-reshuffles on exhaustion.

    Step-based training needs to keep pulling batches from the loader until `max_steps` is
    hit, regardless of dataset size. Each time the underlying iterator runs out, we bump a
    "dataloader_epoch" counter (the dataset's shuffle seed) and rebuild the iterator. The
    dataloader's persistent_workers setting keeps workers alive across rebuilds
    """

    def __init__(self, loader):
        self.loader = loader
        self.dataloader_epoch = 0
        self._set_epoch(self.dataloader_epoch)
        self._iter = iter(loader)

    def _set_epoch(self, epoch: int) -> None:
        if hasattr(self.loader, "set_epoch"):
            self.loader.set_epoch(epoch)
        if (
            dist_utils.is_dist_available_and_initialized()
            and hasattr(self.loader, "sampler")
            and hasattr(self.loader.sampler, "set_epoch")
        ):
            self.loader.sampler.set_epoch(epoch)

    def next_batch(self) -> Tuple:
        try:
            return next(self._iter)
        except StopIteration:
            self.dataloader_epoch += 1
            self._set_epoch(self.dataloader_epoch)
            self._iter = iter(self.loader)
            return next(self._iter)


def train_steps(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_iter: DataLoaderIterator,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    start_step: int,
    num_steps: int,
    max_steps: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    """Train for ``num_steps`` iterations starting at ``start_step``, returning aggregated
    training metrics for the window. ``max_steps`` is used only for log headers.

    Unlike the original ``train_one_epoch``, this function:
      - pulls batches from a persistent ``DataLoaderIterator`` (dataset wraparound is
        invisible to the caller);
      - runs a caller-specified number of iterations rather than "one epoch", so the solver
        can choose when to break for validation / checkpoint;
      - frames progress logs in terms of steps out of the global budget.
    """
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    lr_scheduler = kwargs.get("lr_scheduler", None)
    grad_accum_steps = kwargs.get("gradient_accumulation_steps", 1)

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    end_step = min(start_step + num_steps, max_steps)
    header = "Step: [{}/{}]".format(start_step, max_steps)

    def step_iter():
        for step in range(start_step, end_step):
            samples, targets = data_iter.next_batch()
            yield step, (samples, targets)

    for step, (samples, targets) in metric_logger.log_every(
        step_iter(), print_freq, header, total=(end_step - start_step)
    ):
        metas = dict(epoch=data_iter.dataloader_epoch, step=step, global_step=step,
                     epoch_step=len(data_iter.loader))

        if step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # Multi-scale resize on GPU (scale chosen by collate_fn, passed via targets)
        multiscale_size = targets[0].pop('_multiscale_size', None)
        if multiscale_size is not None:
            samples = torch.nn.functional.interpolate(samples, size=multiscale_size)

        is_accum_step = (step + 1) % grad_accum_steps != 0

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                print(outputs["pred_boxes"])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    new_key = key.replace("module.", "")
                    state[new_key] = value
                new_state["model"] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values()) / grad_accum_steps
            scaler.scale(loss).backward()

            if not is_accum_step:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values()) / grad_accum_steps
            loss.backward()

            if not is_accum_step:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # EMA, warmup, and main LR scheduler all advance per optimizer step
        if not is_accum_step:
            if ema is not None:
                ema.update(model)

            if lr_warmup_scheduler is not None and not lr_warmup_scheduler.finished():
                lr_warmup_scheduler.step()
            elif lr_scheduler is not None:
                lr_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        # NaN check tied to print_freq so the sync amortizes against the logger's own sync
        if (step + 1) % print_freq == 0 and not math.isfinite(loss_value.item()):
            print("Loss is {}, stopping training".format(loss_value.item()))
            print(loss_dict_reduced)
            sys.exit(1)

        main_losses = {k: v for k, v in loss_dict_reduced.items()
                       if not any(s in k for s in ('_aux_', '_dn_', '_pre', '_enc_'))}
        metric_logger.update(loss=loss_value, **main_losses)
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])

        if writer and dist_utils.is_main_process() and step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), step)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, end_step


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    step: int,
    use_wandb: bool,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = coco_evaluator.iou_types

    output_dir = kwargs.get("output_dir", None)

    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            coco_evaluator.summarize()

        if "bbox" in iou_types:
            s = coco_evaluator.coco_eval["bbox"].stats
            if hasattr(s, 'tolist'):
                s = s.tolist()
            if s and len(s) > 0:
                names = ['map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large',
                         'mar_1', 'mar_10', 'mar_100', 'mar_small', 'mar_medium', 'mar_large']
                metrics = {names[i]: f'{s[i]:.3f}' for i in range(min(len(s), len(names)))}
                print(f"validation metrics: {metrics}")
            else:
                print("validation metrics: no detections")

    stats = {}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            s = coco_evaluator.coco_eval["bbox"].stats
            if hasattr(s, 'tolist'):
                s = s.tolist()
            if s and len(s) > 0:
                stats["coco_eval_bbox"] = s
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

        # Release the per-prediction buffers now that we've extracted .stats. Otherwise
        # eval_imgs + _evalImgs_cpp (hundreds of MB for large val sets) sit in RAM until the
        # next evaluate() call's cleanup() clears them
        coco_evaluator.img_ids = []
        coco_evaluator.eval_imgs = {k: [] for k in coco_evaluator.iou_types}
        for ce in coco_evaluator.coco_eval.values():
            if hasattr(ce, "_evalImgs_cpp"):
                ce._evalImgs_cpp = None
            if hasattr(ce, "evalImgs"):
                ce.evalImgs = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats, coco_evaluator
