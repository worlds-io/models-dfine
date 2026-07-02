"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import YAMLConfig, yaml_utils
from src.zoo.dfine.dfine_decoder import MSDeformableAttention


# ── TensorRT deformable-attention plugin export ──────────────────────────────
# D-FINE's multi-scale deformable attention exports as a per-level torch grid_sample
# decomposition. TensorRT's native GridSample mishandles the out-of-bounds samples that
# deformable attention deliberately produces (padding_mode="zeros" should return 0; TRT
# returns garbage), which collapses detection scores and silently drops objects — the bug
# is precision-independent (fp32 and fp16 both wrong). We map deformable attention to the
# TensorRT MultiscaleDeformableAttnPlugin_TRT (reference CUDA impl, correct zero-padding)
# instead. The plugin ships in libnvinfer_plugin, so no custom build is needed; TensorRT
# auto-registers it at engine build and inference-engine initializes it at load.
#
# This is always on: every exported D-FINE ONNX uses the plugin. It changes only the
# exported graph — training and eager inference are untouched (the Function's forward is
# the same reference math, used only to trace shapes during export).


class _MSDeformAttnPlugin(torch.autograd.Function):
    """Emits a MultiscaleDeformableAttnPlugin_TRT node. forward = reference math (tracing only)."""

    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
        # value [bs,S,heads,dim]; sampling_locations [bs,Lq,heads,L,P,2] in [0,1];
        # attention_weights [bs,Lq,heads,L,P]
        bs, _, n_head, c = value.shape
        _, Lq, _, L, P, _ = sampling_locations.shape
        shapes = [(int(spatial_shapes[i, 0]), int(spatial_shapes[i, 1])) for i in range(L)]
        vlist = value.split([h * w for h, w in shapes], dim=1)
        grids = 2 * sampling_locations - 1
        out_lvls = []
        for lvl, (h, w) in enumerate(shapes):
            v = vlist[lvl].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
            g = grids[:, :, :, lvl].permute(0, 2, 1, 3, 4).flatten(0, 1)
            out_lvls.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
        aw = attention_weights.permute(0, 2, 1, 3, 4).reshape(bs * n_head, 1, Lq, L * P)
        out = (torch.stack(out_lvls, dim=-2).flatten(-2) * aw).sum(-1).reshape(bs, n_head, c, Lq)
        return out.permute(0, 3, 1, 2).contiguous()  # [bs,Lq,heads,dim]

    @staticmethod
    def symbolic(g, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
        return g.op("trt.plugins::MultiscaleDeformableAttnPlugin_TRT",
                    value, spatial_shapes, level_start_index, sampling_locations, attention_weights)


def _deformable_attn_core_plugin(value, value_spatial_shapes, sampling_locations,
                                 attention_weights, num_points_list, method="default"):
    """Drop-in for deformable_attention_core_func_v2 that routes to the TRT plugin.

    The stock plugin requires a uniform points-per-level count, but D-FINE uses per-level
    counts (e.g. [3,6,3]). Each level is padded up to max(num_points_list) with extra points
    carrying attention weight 0, so the padded samples contribute nothing — the output is
    mathematically identical to the native per-level sampling.
    """
    bs, n_head, c, _ = value[0].shape
    Lq = sampling_locations.shape[1]
    L = len(num_points_list)
    P = max(num_points_list)

    # value list per level [bs,heads,c,h*w] -> [bs, S, heads, c]
    val = torch.cat([v.reshape(bs, n_head, c, -1) for v in value], dim=-1).permute(0, 3, 1, 2).contiguous()

    shapes = [(int(h), int(w)) for (h, w) in value_spatial_shapes]
    spatial_shapes = torch.tensor(shapes, dtype=torch.int32, device=val.device)
    starts, acc = [], 0
    for h, w in shapes:
        starts.append(acc)
        acc += h * w
    level_start_index = torch.tensor(starts, dtype=torch.int32, device=val.device)

    loc_split = sampling_locations.split(num_points_list, dim=3)
    w_split = attention_weights.split(num_points_list, dim=3)
    loc_pad, w_pad = [], []
    for lvl in range(L):
        nl = num_points_list[lvl]
        lo, we = loc_split[lvl], w_split[lvl]
        if nl < P:
            lo = F.pad(lo, (0, 0, 0, P - nl))  # pad points dim; coords irrelevant (weight 0)
            we = F.pad(we, (0, P - nl))        # padded weight = 0 -> no contribution
        loc_pad.append(lo)
        w_pad.append(we)
    loc = torch.stack(loc_pad, dim=3)  # [bs,Lq,heads,L,P,2]
    wgt = torch.stack(w_pad, dim=3)    # [bs,Lq,heads,L,P]

    out = _MSDeformAttnPlugin.apply(val, spatial_shapes, level_start_index, loc, wgt)  # [bs,Lq,heads,c]
    return out.reshape(bs, Lq, n_head * c)


def _install_deformable_attn_plugin(model):
    """Route every MSDeformableAttention through the TRT plugin core (export-only)."""
    n = 0
    for m in model.modules():
        if isinstance(m, MSDeformableAttention):
            m.ms_deformable_attn_core = _deformable_attn_core_plugin
            n += 1
    print(f"deformable attention -> MultiscaleDeformableAttnPlugin_TRT on {n} module(s)")


def main(args):
    """main"""
    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    cfg = YAMLConfig(args.config, resume=args.resume, **update_dict)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        cfg.model.load_state_dict(state)
    else:
        print("not load model.state_dict, use default init state dict...")

    # Route deformable attention through the TensorRT plugin (always on; export-only).
    _install_deformable_attn_plugin(cfg.model)

    img_size = cfg.yaml_cfg["eval_spatial_size"]

    class Model(nn.Module):
        def __init__(self, img_size) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            # img_size is [H, W] but postprocessor expects [W, H] (matching [x, y] box order)
            self.register_buffer("orig_target_sizes", torch.tensor([[img_size[1], img_size[0]]], dtype=torch.int64))

        def forward(self, images):
            # Input: NHWC uint8-range float32 [0, 255] → NCHW float32 [0, 1]
            images = images.permute(0, 3, 1, 2)
            images = images / 255.0
            outputs = self.model(images)
            orig_target_sizes = self.orig_target_sizes.expand(images.shape[0], -1)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return tuple(o.to(torch.float32) for o in outputs)

    model = Model(img_size)

    # Input: NHWC float32, values in [0, 255]
    data = torch.randint(0, 256, (1, *img_size, 3), dtype=torch.float32)
    _ = model(data)

    dynamic_axes = {"images": {0: "N"}, "labels": {0: "N"}, "boxes": {0: "N"}, "scores": {0: "N"}}

    import os
    if args.output:
        output_file = args.output
    elif args.resume:
        output_file = os.path.splitext(args.resume)[0] + ".onnx"
    else:
        output_file = "model.onnx"

    torch.onnx.export(
        model,
        (data,),
        output_file,
        input_names=["images"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        verbose=False,
        do_constant_folding=True,
    )

    import onnx
    import onnxsim
    onnx_model = onnx.load(output_file)
    onnx_model, check = onnxsim.simplify(onnx_model)
    onnx.save(onnx_model, output_file)
    print(f"Simplify onnx model: {check}")

    if args.check:
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("Check export onnx model done...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/dfine/dfine_hgnetv2_l_coco.yml", type=str)
    parser.add_argument("--resume", "-r", type=str)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("-o", "--output", type=str, help="output onnx file path")
    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")
    args = parser.parse_args()
    main(args)
