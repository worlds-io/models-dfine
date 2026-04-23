"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


# Persistent thread pool for parallel scipy.linear_sum_assignment calls.
# scipy's LAP is implemented in C and releases the GIL, so threads deliver real concurrency.
# Sized to fill the CPU without over-subscribing threads that compete with the dataloader
# workers for CPU time.
_LAP_POOL_WORKERS = max(1, min((os.cpu_count() or 4) - 2, 8))
_LAP_POOL = ThreadPoolExecutor(max_workers=_LAP_POOL_WORKERS, thread_name_prefix="lap")


def _lap_one(args: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray, np.ndarray]:
    """Solve a single LAP instance. Returns (task_id, row_ind, col_ind) so the caller can
    reassemble results in order without relying on submission order."""
    task_id, C = args
    row_ind, col_ind = linear_sum_assignment(C)
    return task_id, row_ind, col_ind


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = [
        "use_focal_loss",
    ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert (
            self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def batch_forward(
        self, outputs_list: List[Dict[str, torch.Tensor]], targets
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Match multiple output heads (main + aux decoder outputs + encoder aux) against the
        same targets in a single batched pass. This preserves per-head matching semantics but
        pays kernel launch / Python overhead once instead of N times. Complements the GPU LAP
        path, where the per-call overhead is the main thing keeping it from beating scipy.

        Returns a list of per-head indices lists, each matching the shape of forward()'s
        ``{"indices": [...]}`` return value
        """
        num_heads = len(outputs_list)
        bs, num_queries = outputs_list[0]["pred_logits"].shape[:2]
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        sizes = [len(v["boxes"]) for v in targets]
        total_T = tgt_bbox.shape[0]

        # Batch with zero targets across every image: skip cost construction and return
        # per-head per-image empty CPU index pairs. Matches the early-return in
        # _auction_match_from_flat_cost() and avoids both a pointless GPU->CPU transfer and
        # a scipy call on Q×0 matrices
        if total_T == 0:
            empty = torch.empty(0, dtype=torch.int64)
            return [[(empty.clone(), empty.clone()) for _ in range(bs)] for _ in range(num_heads)]

        tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)

        # Build a stack of cost matrices on GPU: [num_heads, bs, Q, total_T]
        cost_stack = []
        for outputs in outputs_list:
            if self.use_focal_loss:
                out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
            else:
                out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
            out_bbox = outputs["pred_boxes"].flatten(0, 1)

            if self.use_focal_loss:
                op = out_prob[:, tgt_ids]
                neg = (1 - self.alpha) * (op ** self.gamma) * (-(1 - op + 1e-8).log())
                pos = self.alpha * ((1 - op) ** self.gamma) * (-(op + 1e-8).log())
                cost_class = pos - neg
            else:
                cost_class = -out_prob[:, tgt_ids]

            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), tgt_bbox_xyxy)
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            cost_stack.append(C.view(bs, num_queries, -1))

        # Single .cpu() transfer for the whole stack, then run each per-(head, image) LAP
        # concurrently through a thread pool. scipy.linear_sum_assignment releases the GIL,
        # so the threads deliver real parallelism (not pseudo-parallelism gated by the GIL).
        # The .cpu() is a blocking transfer — the main thread can't start the first LAP
        # until the GPU forward pass has produced the cost matrices — but from that point
        # on, running num_heads × bs problems in parallel instead of sequentially is a
        # direct wall-clock win proportional to min(num_heads × bs, pool_size).
        cost_all_cpu = torch.stack(cost_stack, dim=0).cpu()
        cost_all_cpu = torch.nan_to_num(cost_all_cpu, nan=1.0).numpy()

        # Build a flat task list with deterministic task_ids so we can reassemble
        # results in order regardless of completion order. Each split chunk has shape
        # [bs, Q, sizes[i]]; we select row `i` to get image `i`'s own cost matrix.
        tasks: List[Tuple[int, np.ndarray]] = []
        for h in range(num_heads):
            per_head_chunks = np.split(cost_all_cpu[h], np.cumsum(sizes)[:-1], axis=-1)
            for i, chunk in enumerate(per_head_chunks):
                if sizes[i] > 0:
                    task_id = h * bs + i
                    tasks.append((task_id, chunk[i]))

        # Submit all LAP problems to the shared thread pool. executor.map preserves the
        # iteration order of results, so we can zip them back with the (h, i) pairs
        results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for task_id, row_ind, col_ind in _LAP_POOL.map(_lap_one, tasks):
            results[task_id] = (row_ind, col_ind)

        # Reassemble per-head per-image index pairs, filling in empties for zero-target imgs
        out: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
        empty = torch.empty(0, dtype=torch.int64)
        for h in range(num_heads):
            head_indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for i in range(bs):
                task_id = h * bs + i
                if task_id not in results:
                    head_indices.append((empty.clone(), empty.clone()))
                else:
                    row_ind, col_ind = results[task_id]
                    head_indices.append((
                        torch.as_tensor(row_ind, dtype=torch.int64),
                        torch.as_tensor(col_ind, dtype=torch.int64),
                    ))
            out.append(head_indices)
        return out

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (
                (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        sizes = [len(v["boxes"]) for v in targets]

        C = C.view(bs, num_queries, -1).cpu()
        C = torch.nan_to_num(C, nan=1.0)

        if return_topk:
            # get_top_k_matches does multiple rounds of scipy on the same cost matrices and
            # needs the full C tensor for in-place masking between rounds, so keep this
            # single-threaded.
            indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            indices = [
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_pre
            ]
        else:
            # Parallel scipy via the shared thread pool — same GIL-release trick as
            # batch_forward. Single-head has only bs problems (vs num_heads × bs in
            # batch_forward), so the win here is smaller but still real.
            chunks = list(C.split(sizes, -1))
            tasks = [(i, chunks[i][i].numpy()) for i in range(bs) if sizes[i] > 0]
            results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            for task_id, row_ind, col_ind in _LAP_POOL.map(_lap_one, tasks):
                results[task_id] = (row_ind, col_ind)
            empty_np = np.empty(0, dtype=np.int64)
            indices_pre = [results.get(i, (empty_np, empty_np)) for i in range(bs)]
            indices = [
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_pre
            ]

        # Compute topk indices
        if return_topk:
            return {
                "indices_o2m": self.get_top_k_matches(
                    C, sizes=sizes, k=return_topk, initial_indices=indices_pre
                )
            }

        return {"indices": indices}  # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = (
                [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                if i > 0
                else initial_indices
            )
            indices_list.append(
                [
                    (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                    for i, j in indices_k
                ]
            )
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [
            (
                torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                torch.cat([indices_list[i][j][1] for i in range(k)], dim=0),
            )
            for j in range(len(sizes))
        ]
        # C.copy_(C_original)
        return indices_list
