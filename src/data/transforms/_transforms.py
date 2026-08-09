"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from ...core import register
from .._misc import (
    BoundingBoxes,
    Image,
    Mask,
    SanitizeBoundingBoxes as _TVSanitizeBoundingBoxes,
    Video,
    _boxes_keys,
    convert_to_tv_tensor,
)

torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


def _find_distill_target(inputs):
    """Locate the target dict inside a transform sample (img, target, dataset)."""
    if isinstance(inputs, dict):
        return inputs
    if isinstance(inputs, (tuple, list)):
        for item in inputs:
            if isinstance(item, dict) and "labels" in item:
                return item
    return None


def _labels_and_weights_getter(inputs):
    """``labels_getter`` for SanitizeBoundingBoxes that also keeps the per-detection
    confidence ``weights`` tensor row-aligned with boxes/labels when degenerate
    boxes are dropped. Falls back to labels-only (stock behaviour) when no weights
    are present. SanitizeBoundingBoxes filters every returned tensor by identity."""
    target = _find_distill_target(inputs)
    if target is None:
        return None
    if "weights" in target:
        return (target["labels"], target["weights"])
    return target["labels"]


@register(name="SanitizeBoundingBoxes")
class SanitizeBoundingBoxes(_TVSanitizeBoundingBoxes):
    """torchvision ``SanitizeBoundingBoxes`` whose default ``labels_getter`` also
    filters the distillation ``weights`` tensor in lockstep with boxes/labels, so a
    per-detection confidence weight never de-syncs when augmentation drops boxes
    (e.g. after RandomIoUCrop zeroes out-of-crop boxes). Behaviour is unchanged for
    datasets without a ``weights`` key."""

    def __init__(self, *args, labels_getter=_labels_and_weights_getter, **kwargs):
        super().__init__(*args, labels_getter=labels_getter, **kwargs)


@register()
class EmptyTransform(T.Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class RandomServeShape(T.Transform):
    """Simulate the production capture chain on high-resolution training frames.

    At serve time the sign downscales the native camera frame so its long side is
    ``long_dim`` px (SWS_AREA) and re-encodes it as JPEG (quality ~85–100) before
    the detector ever sees it; the frames persisted for training are the pristine
    native-resolution captures. Without this transform the model trains on detail
    the production pipeline can never deliver.

    The image is BOX-downscaled to ``long_dim`` on its long side, JPEG
    round-tripped at a random quality in [q_min, q_max] (fixed ``q`` when
    ``deterministic``), then BILINEAR-upscaled back to its original size — so
    geometry (boxes, later crops/resizes) is untouched and only the information
    content matches serve conditions.

    No-ops when the image's long side is already <= ``long_dim`` (e.g. the
    service path pre-shrinks images to model resolution), when ``long_dim <= 0``,
    or when the env kill-switch ``TRAIN_SERVE_SHAPE=0`` is set.
    """

    def __init__(self, long_dim=1147, q_min=85, q_max=100, p=1.0,
                 q=93, deterministic=False, enabled=True) -> None:
        super().__init__()
        import os as _os
        self.long_dim = int(long_dim)
        self.q_min = int(q_min)
        self.q_max = int(q_max)
        self.p = float(p)
        self.q = int(q)
        self.deterministic = bool(deterministic)
        self.enabled = bool(enabled) and _os.environ.get("TRAIN_SERVE_SHAPE", "1") != "0"

    def _degrade(self, img: PIL.Image.Image) -> PIL.Image.Image:
        import io as _io

        w, h = img.size
        long_side = max(w, h)
        if long_side <= self.long_dim:
            return img
        scale = self.long_dim / long_side
        small = (max(1, round(w * scale)), max(1, round(h * scale)))
        if self.deterministic:
            quality = self.q
        else:
            quality = int(torch.randint(self.q_min, self.q_max + 1, (1,)).item())
        resampling = getattr(PIL.Image, "Resampling", PIL.Image)
        rgb = img if img.mode == "RGB" else img.convert("RGB")
        down = rgb.resize(small, resampling.BOX)
        buf = _io.BytesIO()
        down.save(buf, format="JPEG", quality=quality, subsampling=2)
        buf.seek(0)
        decoded = PIL.Image.open(buf)
        decoded.load()
        return decoded.resize((w, h), resampling.BILINEAR)

    def forward(self, *inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]
        if not self.enabled or self.long_dim <= 0:
            return sample
        if not self.deterministic and self.p < 1.0 and torch.rand(1).item() >= self.p:
            return sample
        if isinstance(sample, PIL.Image.Image):
            return self._degrade(sample)
        if isinstance(sample, (tuple, list)) and sample and isinstance(sample[0], PIL.Image.Image):
            out = (self._degrade(sample[0]),) + tuple(sample[1:])
            return out if isinstance(sample, tuple) else list(out)
        return sample


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params["padding"]
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]["padding"] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(
                inpt, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size
            )

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.0

        inpt = Image(inpt)

        return inpt
