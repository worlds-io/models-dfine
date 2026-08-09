"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import gc
import math
import sys
from typing import Dict, Iterable, List

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


def _release_allocator_arenas() -> None:
    """Tell glibc to return empty arenas to the OS. CPython + glibc keep freed memory in
    per-thread arenas indefinitely unless explicitly told to trim; without this call, each
    val run leaves multi-GB of "free but not returned" memory in the process, which
    eventually trips k8s eviction. Requires PYTHONMALLOC=malloc in the pod env for Python's
    small-object allocations to go through glibc (otherwise they stay in obmalloc pools
    which are not trimmable)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    # Solver hook to re-apply mode overrides that model.train() just clobbered
    # (e.g. putting frozen-backbone BatchNorm back in eval so running stats
    # don't drift while the backbone is nominally frozen).
    after_train_mode = kwargs.get("after_train_mode", None)
    if after_train_mode is not None:
        after_train_mode()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = "Epoch: [{}]".format(epoch + 1) if epochs is None else "Epoch: [{}/{}]".format(epoch + 1, epochs)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    grad_accum_steps = kwargs.get("gradient_accumulation_steps", 1)

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # Multi-scale resize on GPU (scale chosen by collate_fn, passed via targets)
        multiscale_size = targets[0].pop('_multiscale_size', None)
        if multiscale_size is not None:
            samples = torch.nn.functional.interpolate(samples, size=multiscale_size)

        is_accum_step = (i + 1) % grad_accum_steps != 0

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                print(outputs["pred_boxes"])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace("module.", "")
                    # Add the updated key-value pair to the state dictionary
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

        # ema and warmup update on every optimizer step
        if not is_accum_step:
            if ema is not None:
                ema.update(model)

            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        # NaN check on optimizer-step boundaries. Previously ran every iter with a
        # GPU->CPU sync; now gated on `not is_accum_step` so the max delay between
        # the first NaN loss and the bail-out is one grad-accumulation window (typically
        # 1 step) rather than a full `print_freq` window of optimizer updates
        if not is_accum_step and not math.isfinite(loss_value.item()):
            print("Loss is {}, stopping training".format(loss_value.item()))
            print(loss_dict_reduced)
            sys.exit(1)

        # Only log main losses for cleaner output
        main_losses = {k: v for k, v in loss_dict_reduced.items()
                       if not any(s in k for s in ('_aux_', '_dn_', '_pre', '_enc_'))}
        metric_logger.update(loss=loss_value, **main_losses)
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    # Discard any trailing partial accumulation window. When len(data_loader) is
    # not a multiple of grad_accum_steps the leftover gradients were neither
    # stepped nor zeroed, silently leaking into the first optimizer step of the
    # next epoch.
    if grad_accum_steps > 1 and len(data_loader) % grad_accum_steps != 0:
        optimizer.zero_grad(set_to_none=True)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    output_dir = kwargs.get("output_dir", None)
    max_val_images = kwargs.get("max_val_images", 0) or 0

    # Accumulate raw predictions, then run a single coco_evaluator.update() at the end.
    # The library's per-batch update() pattern runs coco_eval.evaluate() inside every call
    # and appends a per-(cat × area × batch_size) numpy array of C++ eval wrappers to
    # eval_imgs[iou] — at production scale (27k val images, 80 cats, 4 areas, bs 16) that
    # accumulates >8M wrappers over the val loop, allocated into the process heap and never
    # returned until the val ends. A single end-of-loop update() converts per-batch growth
    # into one end-of-val peak with the same total wrappers but a much shorter residency
    all_predictions: Dict = {}
    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        for target, output in zip(targets, results):
            img_id = target["image_id"].item()
            all_predictions[img_id] = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                        for k, v in output.items()}

        if max_val_images > 0 and len(all_predictions) >= max_val_images:
            break

    if coco_evaluator is not None and all_predictions:
        # Restrict COCO evaluation to just the image IDs we ran predictions on. Without this,
        # the evaluator iterates every GT image ID (params.imgIds defaults to the full val
        # set) and counts unpredicted images as all-false-negatives, pulling mAP artificially
        # low and making epoch-to-epoch deltas noisy when each epoch samples a different
        # subset via shuffle=True + max_val_images
        predicted_ids = list(all_predictions.keys())
        for ce in coco_evaluator.coco_eval.values():
            ce.params.imgIds = predicted_ids

        coco_evaluator.update(all_predictions)
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        # summarize() populates .stats — suppress its verbose table
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            coco_evaluator.summarize()

        # Print concise YOLO-style validation summary
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

        # Aggressively release per-val state. The critical retained buffers:
        #   - eval_imgs[iou]: list of per-batch np.array(_evalImgs_cpp) arrays, one per
        #     update() call — O(N_batches × cats × areas × bs) wrappers
        #   - coco_eval[iou]._evalImgs_cpp: merged eval buffer after synchronize
        #   - coco_eval[iou].gt_dataset / .dt_dataset: C++ datasets holding per-annotation
        #     Python-wrapped instances (NOT cleared by library until next _prepare() call)
        #   - coco_eval[iou].ious: dict of per-(imgId, catId) IoU ndarrays
        #   - coco_eval[iou].cocoDt: last batch's detection COCO object
        #   - coco_eval[iou]._paramsEval: deepcopy of params from last update
        # coco_eval itself is preserved so the solver can read .stats; the structures that
        # don't contribute to stats are all released here
        coco_evaluator.img_ids = []
        coco_evaluator.eval_imgs = {k: [] for k in coco_evaluator.iou_types}
        for ce in coco_evaluator.coco_eval.values():
            if hasattr(ce, "_evalImgs_cpp"):
                ce._evalImgs_cpp = None
            if hasattr(ce, "evalImgs"):
                ce.evalImgs = None
            if hasattr(ce, "gt_dataset") and hasattr(ce.gt_dataset, "clean"):
                ce.gt_dataset.clean()
            if hasattr(ce, "dt_dataset") and hasattr(ce.dt_dataset, "clean"):
                ce.dt_dataset.clean()
            ce.ious = {}
            ce.cocoDt = None
            ce._paramsEval = None

    # Drop the per-val prediction dict before trimming. Bulk CPU tensor data lives in
    # torch's caching allocator and isn't affected by malloc_trim, but the dict wrappers
    # and int image_id keys live on the Python heap (glibc arenas under PYTHONMALLOC=malloc)
    # and only get returned to the OS if freed before the trim call
    all_predictions = None
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _release_allocator_arenas()

    return stats, coco_evaluator
