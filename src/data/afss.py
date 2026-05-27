"""
Anti-Forgetting Sampling Strategy (AFSS) for D-FINE.

Port of the strategy from "Does YOLO Really Need to See Every Training Image in
Every Epoch?" (Xie et al., arXiv:2603.17684). See the YOLOv9 fork's
yolo/tools/afss.py for the full rationale; AFSSState / AFSSSampler below are the
same framework-agnostic logic. Only the scoring pass is D-FINE specific.

We use AFSS not for speedup but to train *longer on the images that still
matter*: skip already-learned ("easy") images each epoch and let the Go
orchestrator reinvest the freed per-epoch budget into more epochs + a matching
LR schedule, for better convergence at ~equal wall-clock.

Single-GPU only (the solver guards on this): under DDP, warp_loader replaces the
sampler with a DistributedSampler, which would silently disable AFSS.
"""

from __future__ import annotations

import copy
from math import ceil
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision.ops import box_convert, box_iou

EASY, MEDIUM, HARD = 0, 1, 2


def per_image_recall_precision(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_thresh: float,
) -> tuple[float, float]:
    """Greedy class-aware IoU matching for one image; returns (recall, precision).

    Mirrors the matching in src/solver/validator.py but per-image. Predictions
    must already be confidence-filtered. Degenerate cases keep unlearned images
    in the hard tier (see the YOLOv9 copy for the rationale)."""
    n_gt = int(gt_boxes.shape[0])
    n_pred = int(pred_boxes.shape[0])
    if n_gt == 0:
        return (1.0, 1.0) if n_pred == 0 else (1.0, 0.0)
    if n_pred == 0:
        return 0.0, 1.0

    iou = box_iou(pred_boxes, gt_boxes)
    same_class = pred_labels.view(-1, 1) == gt_labels.view(1, -1)
    iou = torch.where(same_class, iou, torch.zeros_like(iou))

    mask = iou >= iou_thresh
    if not bool(mask.any()):
        tp = 0
    else:
        pi, gi = torch.where(mask)
        order = torch.argsort(iou[pi, gi], descending=True)
        matched_pred: set[int] = set()
        matched_gt: set[int] = set()
        tp = 0
        for k in order.tolist():
            p, g = int(pi[k]), int(gi[k])
            if p in matched_pred or g in matched_gt:
                continue
            matched_pred.add(p)
            matched_gt.add(g)
            tp += 1

    fp = n_pred - tp
    fn = n_gt - tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    return recall, precision


class AFSSState:
    """Per-image bookkeeping that persists across the run (and resume)."""

    def __init__(self, num_images: int, easy_thresh: float, medium_thresh: float):
        self.num_images = num_images
        self.easy_thresh = easy_thresh
        self.medium_thresh = medium_thresh
        self.sufficiency = np.zeros(num_images, dtype=np.float32)
        self.last_used_epoch = np.full(num_images, -1, dtype=np.int64)
        self.tier = np.full(num_images, HARD, dtype=np.int8)
        self.last_scored_epoch = -1

    def update_sufficiency(self, scores: Dict[int, float], epoch: int) -> None:
        for idx, suff in scores.items():
            self.sufficiency[idx] = suff
        self.tier = np.where(
            self.sufficiency >= self.easy_thresh,
            EASY,
            np.where(self.sufficiency >= self.medium_thresh, MEDIUM, HARD),
        ).astype(np.int8)
        self.last_scored_epoch = epoch

    def tier_counts(self) -> Dict[str, int]:
        return {
            "easy": int((self.tier == EASY).sum()),
            "medium": int((self.tier == MEDIUM).sum()),
            "hard": int((self.tier == HARD).sum()),
        }

    def state_dict(self) -> dict:
        return {
            "sufficiency": self.sufficiency,
            "last_used_epoch": self.last_used_epoch,
            "tier": self.tier,
            "last_scored_epoch": self.last_scored_epoch,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.sufficiency = np.asarray(sd["sufficiency"], dtype=np.float32)
        self.last_used_epoch = np.asarray(sd["last_used_epoch"], dtype=np.int64)
        self.tier = np.asarray(sd["tier"], dtype=np.int8)
        self.last_scored_epoch = int(sd["last_scored_epoch"])


class AFSSSampler(Sampler):
    """Per-epoch subset sampler. __len__ returns the current epoch's selection so
    the DataLoader length (iters/epoch) shrinks once sufficiency is known."""

    def __init__(
        self,
        num_images: int,
        easy_thresh: float = 0.9,
        medium_thresh: float = 0.5,
        easy_keep_frac: float = 0.15,
        medium_keep_frac: float = 0.5,
        hard_repeat: int = 1,
        refresh_period: int = 5,
        score_only: bool = False,
        seed: int = 0,
    ):
        self.num_images = num_images
        self.easy_keep_frac = easy_keep_frac
        self.medium_keep_frac = medium_keep_frac
        self.hard_repeat = max(1, hard_repeat)
        self.refresh_period = max(1, refresh_period)
        self.score_only = score_only
        self.state = AFSSState(num_images, easy_thresh, medium_thresh)
        self._rng = np.random.default_rng(seed)
        self._epoch = 0
        self._indices: List[int] = list(range(num_images))

    def needs_refresh(self, epoch: int) -> bool:
        return self.state.last_scored_epoch < 0 or (epoch - self.state.last_scored_epoch) >= self.refresh_period

    def update_sufficiency(self, scores: Dict[int, float], epoch: int) -> None:
        self.state.update_sufficiency(scores, epoch)
        counts = self.state.tier_counts()
        print(
            f"[AFSS] epoch {epoch}: scored {len(scores)} images -> "
            f"easy={counts['easy']} medium={counts['medium']} hard={counts['hard']}"
        )

    def _select(self, epoch: int) -> List[int]:
        tier = self.state.tier
        last_used = self.state.last_used_epoch
        hard_idx = np.where(tier == HARD)[0]
        medium_idx = np.where(tier == MEDIUM)[0]
        easy_idx = np.where(tier == EASY)[0]

        selected: List[int] = []
        selected.extend(np.tile(hard_idx, self.hard_repeat).tolist())

        if medium_idx.size:
            keep = ceil(self.medium_keep_frac * medium_idx.size)
            ordered = medium_idx[np.argsort(last_used[medium_idx], kind="stable")]
            n_priority = keep // 2
            chosen = list(ordered[:n_priority])
            pool = ordered[n_priority:]
            remaining = keep - len(chosen)
            if remaining > 0 and pool.size:
                chosen.extend(self._rng.choice(pool, size=min(remaining, pool.size), replace=False).tolist())
            selected.extend(int(i) for i in chosen)

        if easy_idx.size:
            keep = ceil(self.easy_keep_frac * easy_idx.size)
            ordered = easy_idx[np.argsort(last_used[easy_idx], kind="stable")]
            selected.extend(int(i) for i in ordered[:keep])

        return selected

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        if self.score_only or self.state.last_scored_epoch < 0:
            idx = np.arange(self.num_images)
        else:
            idx = np.asarray(self._select(epoch), dtype=np.int64)
            if idx.size == 0:
                idx = np.arange(self.num_images)
        self.state.last_used_epoch[idx] = epoch
        self._rng.shuffle(idx)
        self._indices = idx.tolist()

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


# Sentinel epoch that exceeds any configured stop_epoch, so the transform Compose's
# stop_epoch policy skips the heavy augmentations (RandomPhotometricDistort/ZoomOut/
# IoUCrop) and the scoring pass sees clean, eval-style images.
_SCORING_EPOCH = 1_000_000_000


def build_scoring_loader(train_dataset, batch_size, num_workers, collate_fn, max_images, seed=0):
    """Loader over the training images with augmentation disabled, for scoring.

    Shallow-copies the dataset so setting a high epoch (to bypass the aug policy)
    does not affect the live training dataset. Each target carries its true
    dataset index in target["idx"], so capping/subsetting stays correct.
    """
    scoring_ds = copy.copy(train_dataset)
    scoring_ds.set_epoch(_SCORING_EPOCH)

    n = len(scoring_ds)
    if max_images and n > max_images:
        indices = sorted(np.random.default_rng(seed).choice(n, size=max_images, replace=False).tolist())
        scoring_ds = Subset(scoring_ds, indices)

    return DataLoader(
        scoring_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
    )


@torch.no_grad()
def score_train_set(model, postprocessor, loader, device, conf_thresh, iou_thresh) -> Dict[int, float]:
    """Run the (EMA) model over the scoring loader and return {dataset_idx: sufficiency}.

    Mirrors det_engine.evaluate()'s inference: model(samples) without targets (no
    denoising), postprocessor -> per-image xyxy boxes in orig-image pixels. GT
    (normalized cxcywh) is converted to the same space for matching.
    """
    was_training = model.training
    model.eval()
    suff: Dict[int, float] = {}
    for samples, targets in loader:
        samples = samples.to(device, non_blocking=True)
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        outputs = model(samples)
        results = postprocessor(outputs, orig_sizes)

        for target, result in zip(targets, results):
            idx = int(target["idx"].item())
            pred_boxes = result["boxes"].detach().cpu().float()
            pred_labels = result["labels"].detach().cpu()
            scores = result["scores"].detach().cpu()
            keep = scores >= conf_thresh
            pred_boxes, pred_labels = pred_boxes[keep], pred_labels[keep]

            if "boxes" in target and target["boxes"].numel() > 0:
                w, h = int(target["orig_size"][0]), int(target["orig_size"][1])
                gt_norm = torch.as_tensor(target["boxes"]).cpu().float()  # cxcywh, normalized
                gt_boxes = box_convert(gt_norm, in_fmt="cxcywh", out_fmt="xyxy")
                gt_boxes = gt_boxes * torch.tensor([w, h, w, h], dtype=torch.float32)
                gt_labels = torch.as_tensor(target["labels"]).cpu()
            else:
                gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
                gt_labels = torch.zeros((0,), dtype=torch.long)

            recall, precision = per_image_recall_precision(
                pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh
            )
            suff[idx] = min(recall, precision)
    if was_training:
        model.train()
    return suff
