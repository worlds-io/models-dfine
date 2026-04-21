"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

# Gated on via DFINE_GPU_MATCHER=1 — replaces the scipy LAP call (which forces a GPU->CPU sync
# per matcher call, 8+ times per training iter) with a Bertsekas forward auction run entirely
# on the GPU. Off by default for safety; flip once validated on a long training run
_GPU_MATCHER = os.environ.get("DFINE_GPU_MATCHER", "0") == "1"
_GPU_MATCHER_DEBUG = os.environ.get("DFINE_GPU_MATCHER_DEBUG", "0") == "1"


@torch.no_grad()
def _gpu_auction_lap(
    cost: torch.Tensor,
    active: torch.Tensor,
    max_iters_factor: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve a batch of bipartite LAP instances on the GPU via parallel (Jacobi-style)
    forward auction.

    Persons = targets, objects = predictions (queries). Partial-assignment auction — each
    active person eventually gets a distinct object, requires num_objects >= num_active
    (satisfied by DETR's num_queries >= num_targets).

    Uses a single tight epsilon = tol/(Tmax+1) so the produced assignment's total cost is
    within `tol` of scipy's. Scaling is a follow-up optimization once correctness lands.

    Args:
        cost: [B, Q, Tmax] float32 cost tensor.
        active: [B, Tmax] bool — True for real targets, False for padded slots.
        max_iters_factor: safety budget multiplier on iterations (upper bound scales with
            dynamic_range / eps, so adjust if costs have a very wide range).

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

    # Single-eps forward auction. Bertsekas bound: total-cost error <= Tmax * eps, so pick
    # eps tight enough for our tolerance. Iteration count scales with (dyn_range / eps), which
    # for random cost matrices of range ~5 and eps ~= 1e-5 is a few thousand iters per problem.
    # Epsilon scaling would reduce this 10-100x but requires proper ε-CS handling to stay exact;
    # left as follow-up work
    tol_total = 1e-3
    eps = max(tol_total / max(Tmax, 1), 1e-8)

    price = torch.zeros(B, Q, device=device, dtype=dtype)
    obj_of_person = torch.full((B, Tmax), -1, dtype=torch.long, device=device)
    person_of_obj = torch.full((B, Q), -1, dtype=torch.long, device=device)

    v_real = torch.where(active[:, :, None], v, v.new_full((1,), 0.0))
    dyn_range = float((v_real.amax() - v_real.amin()).item())
    max_iters = max(max_iters_factor * Tmax, int((dyn_range / eps) * 4) + 64)
    max_iters = min(max_iters, 200000)  # hard ceiling

    person_arange = torch.arange(Tmax, device=device).expand(B, Tmax)

    for _ in range(max_iters):
        bidders = active & (obj_of_person == -1)
        if not bidders.any():
            break

        adj = v - price[:, None, :]  # [B, Tmax, Q]
        adj = torch.where(bidders[:, :, None], adj, adj.new_full((1,), -float("inf")))
        top2 = torch.topk(adj, k=min(2, Q), dim=-1)
        best_obj = top2.indices[..., 0]
        best_val = top2.values[..., 0]
        second_val = top2.values[..., 1] if top2.values.shape[-1] > 1 else best_val - 1.0
        bid = best_val - second_val + eps
        bid = torch.where(bidders, bid, bid.new_full((1,), -float("inf")))

        bid_buf = torch.full((B, Q), -float("inf"), device=device, dtype=dtype)
        bid_buf.scatter_reduce_(1, best_obj, bid, reduce="amax", include_self=True)

        LARGE = Tmax + 1
        is_winner = bidders & torch.isfinite(bid) & (bid_buf.gather(1, best_obj) == bid)
        cand_person = torch.where(is_winner, person_arange, person_arange.new_full((1,), LARGE))
        winner_per_obj = torch.full((B, Q), LARGE, dtype=torch.long, device=device)
        winner_per_obj.scatter_reduce_(
            1, best_obj, cand_person, reduce="amin", include_self=True
        )
        has_winner = winner_per_obj < LARGE

        prev_owners = person_of_obj
        displaced = (prev_owners >= 0) & has_winner & (winner_per_obj != prev_owners)
        if displaced.any():
            d_b, d_q = torch.nonzero(displaced, as_tuple=True)
            obj_of_person[d_b, prev_owners[d_b, d_q]] = -1

        person_of_obj = torch.where(has_winner, winner_per_obj, prev_owners)
        w_b, w_q = torch.nonzero(has_winner, as_tuple=True)
        obj_of_person[w_b, winner_per_obj[w_b, w_q]] = w_q

        bid_add = torch.where(has_winner, bid_buf, bid_buf.new_zeros(()))
        price = price + bid_add

    converged = ~(active & (obj_of_person == -1)).any(dim=1)
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


def _assert_lap_parity(
    C_flat: torch.Tensor,
    num_queries: int,
    sizes: List[int],
    gpu_indices: List[Tuple[torch.Tensor, torch.Tensor]],
    tol: float = 1e-3,
) -> None:
    """Debug helper: run scipy LAP in parallel and assert the matched-cost sums agree with
    the GPU auction's output within ``tol``. Only runs under DFINE_GPU_MATCHER_DEBUG=1.
    """
    bs = len(sizes)
    C_view = C_flat.view(bs, num_queries, -1).detach().cpu().numpy()
    offset = 0
    for i, t in enumerate(sizes):
        if t == 0:
            offset += t
            continue
        C_i = C_view[i, :, offset : offset + t]
        row_ind, col_ind = linear_sum_assignment(np.nan_to_num(C_i, nan=1.0))
        cpu_cost = float(C_i[row_ind, col_ind].sum())
        rows, cols = gpu_indices[i]
        gpu_cost = float(C_i[rows.numpy(), cols.numpy()].sum())
        if abs(cpu_cost - gpu_cost) > tol:
            raise AssertionError(
                f"GPU auction LAP parity check failed: image {i} size {t} cpu_cost={cpu_cost:.6f} "
                f"gpu_cost={gpu_cost:.6f} (tol={tol})"
            )
        offset += t


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

        # GPU auction path — eliminates the C.cpu() sync and the scipy LAP. Off by default
        # behind DFINE_GPU_MATCHER=1 until validated end-to-end. Dead-code get_top_k_matches
        # still needs the CPU path (return_topk=True), so we only route non-topk calls
        if _GPU_MATCHER and not return_topk and C.is_cuda:
            indices = _auction_match_from_flat_cost(C, num_queries, sizes)
            if _GPU_MATCHER_DEBUG:
                _assert_lap_parity(C, num_queries, sizes, indices)
            return {"indices": indices}

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
