from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


_EPS = 1.0e-6
_POLLUTANT_CHANNELS = 2


def clone_data(data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in data.items()}


def ensure_finite(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(tensor).all():
        return tensor
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def candidate_name(spec: Mapping[str, Any]) -> str:
    return str(spec.get("name") or "unnamed")


def make_identity_affine(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size = int(reference.shape[0])
    channel_count = int(reference.shape[-1])
    return {
        "offset": torch.zeros(batch_size, channel_count, device=reference.device, dtype=reference.dtype),
        "scale": torch.ones(batch_size, channel_count, device=reference.device, dtype=reference.dtype),
    }


def make_channel_affine(values: torch.Tensor, *, center: bool, scale: bool) -> dict[str, torch.Tensor]:
    batch_size = int(values.shape[0])
    channel_count = int(values.shape[-1])
    flat = values.reshape(batch_size, -1, channel_count)
    mean = flat.mean(dim=1) if center else torch.zeros(batch_size, channel_count, device=values.device, dtype=values.dtype)
    std = flat.std(dim=1, unbiased=False).clamp_min(_EPS) if scale else torch.ones(
        batch_size,
        channel_count,
        device=values.device,
        dtype=values.dtype,
    )
    return {"offset": mean, "scale": std}


def _affine_view(values: torch.Tensor, ndim: int) -> torch.Tensor:
    return values.reshape(int(values.shape[0]), *([1] * (ndim - 2)), int(values.shape[-1]))


def apply_affine(tensor: torch.Tensor, affine: Mapping[str, torch.Tensor]) -> torch.Tensor:
    offset = _affine_view(affine["offset"].to(device=tensor.device, dtype=tensor.dtype), tensor.ndim)
    scale = _affine_view(affine["scale"].to(device=tensor.device, dtype=tensor.dtype), tensor.ndim)
    return (tensor - offset) / scale.clamp_min(_EPS)


def invert_affine(tensor: torch.Tensor, affine: Mapping[str, torch.Tensor]) -> torch.Tensor:
    offset = _affine_view(affine["offset"].to(device=tensor.device, dtype=tensor.dtype), tensor.ndim)
    scale = _affine_view(affine["scale"].to(device=tensor.device, dtype=tensor.dtype), tensor.ndim)
    return tensor * scale + offset


def visible_condition_values(data: Mapping[str, torch.Tensor]) -> torch.Tensor:
    demo_cond = data["demo_cond_v"]
    quest_cond = data["quest_cond_v"]
    batch_size = int(demo_cond.shape[0])
    channel_count = int(demo_cond.shape[-1])
    return torch.cat(
        [
            demo_cond.reshape(batch_size, -1, channel_count),
            quest_cond.reshape(batch_size, -1, channel_count),
        ],
        dim=1,
    )


def pollutant_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    if int(prediction.shape[-1]) >= _POLLUTANT_CHANNELS:
        prediction = prediction[..., -_POLLUTANT_CHANNELS:]
        target = target[..., -_POLLUTANT_CHANNELS:]
    return torch.mean((prediction - target) ** 2, dim=tuple(range(1, prediction.ndim)))
