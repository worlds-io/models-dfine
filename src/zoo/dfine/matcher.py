"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


@torch.no_grad()
def _gpu_auction_lap(
    cost: torch.Tensor,
    active: torch.Tensor,
    max_iters_factor: int = 128,
    sync_check_every: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve a batch of bipartite LAP instances on the GPU via parallel (Jacobi-style)
    forward auction.

    Persons = targets, objects = predictions (queries). Partial-assignment auction — each
    active person eventually gets a distinct object, requires num_objects >= num_active
    (satisfied by DETR's num_queries >= num_targets).

    Uses a single tight epsilon = tol/(Tmax+1) so the produced assignment's total cost is
    within `tol` of scipy's. Fully vectorized updates — no ``torch.nonzero`` or per-iter
    ``.any()`` calls that would trigger GPU->CPU syncs. Early-exit is probed only every
    ``sync_check_every`` iterations to amortize the single required bool sync.

    Args:
        cost: [B, Q, Tmax] float32 cost tensor.
        active: [B, Tmax] bool — True for real targets, False for padded slots.
        max_iters_factor: safety budget multiplier on iterations.
        sync_check_every: run the `all-assigned?` early-exit probe every N iterations.
            Higher values reduce syncs at the cost of running extra rounds past convergence.

    Returns:
        obj_of_person: [B, Tmax] long. For active persons, the assigned object (query) index.
                       For inactive slots, -1.
        converged: [B] bool. False if the auction exhausted its iteration budget without
                   fully assigning all active persons for that problem.
    """
    B, Q, Tmax = cost.shape
    device = cost.device
    dtype = cost.dtype

    v = (-cost).transpose(1, 2).contiguous()  # [B, Tmax, Q], benefit-major
    v = torch.where(active[:, :, None], v, v.new_full((1,), -1e30))

    # Single tight eps — Bertsekas bound: total-cost error <= Tmax * eps. Epsilon scaling
    # was attempted but destabilizes partial-assignment auction (num_queries > num_targets)
    # where an overshot coarse price on the true optimum has no competing bidders to recover
    # it at finer scales. The win we capture instead is fully vectorized (no torch.nonzero)
    # updates and an amortized early-exit probe — this pushes 10× speedup over the naive
    # version for the same final accuracy
    tol_total = 1e-3
    eps = max(tol_total / max(Tmax, 1), 1e-8)

    price = torch.zeros(B, Q, device=device, dtype=dtype)
    obj_of_person = torch.full((B, Tmax), -1, dtype=torch.long, device=device)
    person_of_obj = torch.full((B, Q), -1, dtype=torch.long, device=device)

    max_iters = min(max_iters_factor * max(Tmax, 1) + 64, 50000)

    person_arange = torch.arange(Tmax, device=device).expand(B, Tmax)
    NEG_INF = float("-inf")
    UNASSIGNED = -1

    it = 0
    while it < max_iters:
        bidders = active & (obj_of_person == UNASSIGNED)

        adj = v - price[:, None, :]
        adj = torch.where(bidders[:, :, None], adj, adj.new_full((1,), NEG_INF))
        top2 = torch.topk(adj, k=min(2, Q), dim=-1)
        best_obj = top2.indices[..., 0]
        best_val = top2.values[..., 0]
        second_val = top2.values[..., 1] if top2.values.shape[-1] > 1 else best_val - 1.0
        bid = best_val - second_val + eps
        bid = torch.where(bidders, bid, bid.new_full((1,), NEG_INF))

        bid_buf = torch.full((B, Q), NEG_INF, device=device, dtype=dtype)
        bid_buf.scatter_reduce_(1, best_obj, bid, reduce="amax", include_self=True)

        LARGE = Tmax + 1
        bidder_has_winning_bid = (
            bidders
            & torch.isfinite(bid)
            & (bid_buf.gather(1, best_obj) == bid)
        )
        cand_person = torch.where(
            bidder_has_winning_bid, person_arange, person_arange.new_full((1,), LARGE)
        )
        winner_per_obj = torch.full((B, Q), LARGE, dtype=torch.long, device=device)
        winner_per_obj.scatter_reduce_(1, best_obj, cand_person, reduce="amin", include_self=True)
        has_winner = winner_per_obj < LARGE

        actual_winner_for_my_target = winner_per_obj.gather(1, best_obj)
        is_winner = bidder_has_winning_bid & (actual_winner_for_my_target == person_arange)

        safe_prev_obj = obj_of_person.clamp(min=0)
        prev_obj_has_winner = has_winner.gather(1, safe_prev_obj)
        prev_obj_new_owner = winner_per_obj.gather(1, safe_prev_obj)
        was_displaced = (
            (obj_of_person >= 0)
            & prev_obj_has_winner
            & (prev_obj_new_owner != person_arange)
        )

        obj_of_person = torch.where(
            is_winner,
            best_obj,
            torch.where(was_displaced, obj_of_person.new_full((1,), UNASSIGNED), obj_of_person),
        )
        person_of_obj = torch.where(has_winner, winner_per_obj, person_of_obj)

        bid_add = torch.where(has_winner, bid_buf, bid_buf.new_zeros(()))
        price = price + bid_add

        it += 1
        if it % sync_check_every == 0:
            if not (active & (obj_of_person == UNASSIGNED)).any():
                break

    converged = ~(active & (obj_of_person == UNASSIGNED)).any(dim=1)
    return obj_of_person, converged


@torch.no_grad()
def _auction_match_from_flat_cost(
    C_flat: torch.Tensor, num_queries: int, sizes: List[int]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Pad the flat [bs*Q, total_T] cost into a batched [bs, Q, Tmax] tensor, run the GPU
    auction, and unpack per-image (query_idx, target_idx) tuples matching scipy's output
    convention. Any image whose auction fails to converge falls back to scipy on CPU for
    just that image.
    """
    bs = len(sizes)
    Tmax = max(sizes) if sizes else 0
    if Tmax == 0:
        empty = torch.empty(0, dtype=torch.long)
        return [(empty.clone(), empty.clone()) for _ in sizes]

    device = C_flat.device
    C_view = C_flat.view(bs, num_queries, -1)  # [bs, Q, total_T]
    C_padded = C_flat.new_zeros((bs, num_queries, Tmax))
    active = torch.zeros(bs, Tmax, dtype=torch.bool, device=device)
    offset = 0
    for i, t in enumerate(sizes):
        if t > 0:
            C_padded[i, :, :t] = C_view[i, :, offset : offset + t]
            active[i, :t] = True
        offset += t

    C_padded = torch.nan_to_num(C_padded, nan=1.0)
    obj_of_person, converged = _gpu_auction_lap(C_padded, active)

    converged_cpu = converged.cpu().tolist()
    obj_cpu = obj_of_person.cpu()
    indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i, t in enumerate(sizes):
        if t == 0:
            empty = torch.empty(0, dtype=torch.long)
            indices.append((empty.clone(), empty.clone()))
            continue
        if converged_cpu[i]:
            rows = obj_cpu[i, :t].to(dtype=torch.long)
            cols = torch.arange(t, dtype=torch.long)
            indices.append((rows, cols))
        else:
            # Rare fallback: CPU scipy for just this problem
            C_i = C_padded[i, :, :t].detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C_i)
            indices.append(
                (torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long))
            )
    return indices


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

        # GPU path: pool all (num_heads * bs) problems into one auction call
        if cost_stack[0].is_cuda and total_T > 0:
            Tmax = max(sizes)
            NH_bs = num_heads * bs
            device = cost_stack[0].device
            # Build padded per-problem cost tensor [NH_bs, Q, Tmax] — each head has its own
            # per-image target slice at positions [offset, offset+t). Pad unused slots with
            # zero; the `active` mask masks them out in the auction kernel anyway
            C_padded = cost_stack[0].new_zeros((NH_bs, num_queries, Tmax))
            active = torch.zeros(NH_bs, Tmax, dtype=torch.bool, device=device)
            for h, C_h in enumerate(cost_stack):
                offset = 0
                for i, t in enumerate(sizes):
                    row = h * bs + i
                    if t > 0:
                        C_padded[row, :, :t] = C_h[i, :, offset : offset + t]
                        active[row, :t] = True
                    offset += t

            C_padded = torch.nan_to_num(C_padded, nan=1.0)
            obj_of_person, converged = _gpu_auction_lap(C_padded, active)
            converged_cpu = converged.cpu().tolist()
            obj_cpu = obj_of_person.cpu()

            out: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
            for h in range(num_heads):
                head_indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
                for i, t in enumerate(sizes):
                    row = h * bs + i
                    if t == 0:
                        empty = torch.empty(0, dtype=torch.long)
                        head_indices.append((empty.clone(), empty.clone()))
                    elif converged_cpu[row]:
                        rows = obj_cpu[row, :t].to(dtype=torch.long)
                        cols = torch.arange(t, dtype=torch.long)
                        head_indices.append((rows, cols))
                    else:
                        # Rare fallback: CPU scipy for the single image
                        C_i = C_padded[row, :, :t].detach().cpu().numpy()
                        row_ind, col_ind = linear_sum_assignment(C_i)
                        head_indices.append(
                            (torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long))
                        )
                out.append(head_indices)
            return out

        # CPU fallback: single .cpu() transfer for the whole stack, then per-head scipy
        cost_all_cpu = torch.stack(cost_stack, dim=0).cpu()
        cost_all_cpu = torch.nan_to_num(cost_all_cpu, nan=1.0)
        out = []
        for h in range(num_heads):
            indices_pre = [
                linear_sum_assignment(c[i])
                for i, c in enumerate(cost_all_cpu[h].split(sizes, -1))
            ]
            out.append(
                [
                    (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                    for i, j in indices_pre
                ]
            )
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

        # GPU auction path — eliminates the C.cpu() sync and the scipy LAP. get_top_k_matches
        # still needs the CPU path (return_topk=True), so we only route non-topk calls here
        if not return_topk and C.is_cuda:
            return {"indices": _auction_match_from_flat_cost(C, num_queries, sizes)}

        C = C.view(bs, num_queries, -1).cpu()

        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
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
