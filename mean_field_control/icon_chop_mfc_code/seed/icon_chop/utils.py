"""Shared tensor utilities for ICON-CHOP seed candidates.

This module is intentionally small and deterministic.  It gives evolution a
safe library of low-complexity primitives while keeping the default seed equal
to raw ICON.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

EPS = 1.0e-6

DATA_KEYS = (
    "demo_input_coords",
    "demo_input_vals",
    "demo_target_coords",
    "demo_target_vals",
    "query_input_coords",
    "query_input_vals",
    "query_target_coords",
)


def clone_data(data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Shallow-copy the semantic data dictionary."""

    return {key: data[key] for key in DATA_KEYS if key in data}


def batch_size_from_data(data: Mapping[str, torch.Tensor]) -> int:
    return int(data["query_input_vals"].shape[0])


def query_target_shape(data: Mapping[str, torch.Tensor], channels: int = 1) -> tuple[int, int, int]:
    """Return the evaluator-required output shape ``(B, N, C)``.

    The MFC evaluator provides ``query_target_coords`` as ``(B, 1, N, K)`` but
    scores predictions as ``(B, N, 1)``.
    """

    coords = data["query_target_coords"]
    batch = int(coords.shape[0])
    n_target = int(coords.shape[-2])
    return batch, n_target, int(channels)


def _broadcast_shape(value: torch.Tensor, stat: torch.Tensor) -> tuple[int, ...]:
    return (int(value.shape[0]),) + (1,) * (value.dim() - 1)


def make_identity_affine(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-batch scalar identity affine transform."""

    batch = int(reference.shape[0])
    zeros = reference.new_zeros(batch, 1)
    ones = reference.new_ones(batch, 1)
    return {"mean": zeros, "std": ones}


def make_scalar_affine(value: torch.Tensor, *, center: bool = True, scale: bool = True) -> dict[str, torch.Tensor]:
    """Make a per-batch scalar affine transform from visible values.

    Statistics are intentionally low-capacity: one mean/std per batch item,
    shared across demos, points, and channels.  This is safer than fitting many
    per-point or per-demo parameters to the few-shot context.
    """

    dims = tuple(range(1, value.dim()))
    if center:
        mean = value.mean(dim=dims, keepdim=False).reshape(value.shape[0], 1)
    else:
        mean = value.new_zeros(value.shape[0], 1)
    if scale:
        std = value.std(dim=dims, unbiased=False, keepdim=False).reshape(value.shape[0], 1)
        std = std.clamp_min(EPS)
    else:
        std = value.new_ones(value.shape[0], 1)
    return {"mean": mean, "std": std}


def merge_visible_inputs(data: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate demo/query input values for input-side statistics."""

    return torch.cat([data["demo_input_vals"], data["query_input_vals"]], dim=1)


def apply_affine(value: torch.Tensor, affine: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mean = affine["mean"].to(device=value.device, dtype=value.dtype)
    std = affine["std"].to(device=value.device, dtype=value.dtype).clamp_min(EPS)
    shape = _broadcast_shape(value, mean)
    return (value - mean.view(shape)) / std.view(shape)


def invert_affine(value: torch.Tensor, affine: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mean = affine["mean"].to(device=value.device, dtype=value.dtype)
    std = affine["std"].to(device=value.device, dtype=value.dtype).clamp_min(EPS)
    shape = _broadcast_shape(value, mean)
    return value * std.view(shape) + mean.view(shape)


def nearest_interpolate(
    source_coords: torch.Tensor,
    source_vals: torch.Tensor,
    target_coords: torch.Tensor,
) -> torch.Tensor:
    """Batched nearest-neighbor interpolation from source points to target points.

    Shapes:
    - source_coords: ``(B, D, Ns, K)``
    - source_vals:   ``(B, D, Ns, C)``
    - target_coords: ``(B, D, Nt, K)``
    - return:        ``(B, D, Nt, C)``

    This is a cheap deterministic operator.  It is not used by the neutral seed
    but is a useful mutation primitive for residualized chains.
    """

    if source_coords.shape[:2] != source_vals.shape[:2]:
        raise ValueError("source coordinates/values have inconsistent batch-demo axes")
    if source_coords.shape[:2] != target_coords.shape[:2]:
        raise ValueError("source/target coordinates have inconsistent batch-demo axes")
    diff = target_coords.unsqueeze(-2) - source_coords.unsqueeze(-3)
    dist2 = torch.sum(diff * diff, dim=-1)
    nearest = torch.argmin(dist2, dim=-1)
    gather_index = nearest.unsqueeze(-1).expand(*nearest.shape, source_vals.shape[-1])
    return torch.gather(source_vals, dim=2, index=gather_index)


def mean_project(source_vals: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
    """Project each sample/demo to its input-value mean at every target point."""

    mean = source_vals.mean(dim=-2, keepdim=True)
    return mean.expand(*target_coords.shape[:-1], source_vals.shape[-1])


def easy_operator(
    source_coords: torch.Tensor,
    source_vals: torch.Tensor,
    target_coords: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Cheap visible-input -> target-coordinate operator.

    Supported modes are intentionally simple and interpretable.  Evolution can
    add modes here, but avoid high-capacity fitting to demo targets.
    """

    if mode == "none":
        return torch.zeros(*target_coords.shape[:-1], source_vals.shape[-1], device=source_vals.device, dtype=source_vals.dtype)
    if mode == "nearest":
        return nearest_interpolate(source_coords, source_vals, target_coords)
    if mode == "mean":
        return mean_project(source_vals, target_coords)
    raise ValueError(f"Unknown easy_operator mode: {mode!r}")


def normalize_to_query_shape(
    prediction: Any,
    data: Mapping[str, torch.Tensor],
    *,
    channels: int = 1,
) -> torch.Tensor:
    """Convert ICON output to the exact evaluator shape ``(B, N, C)``.

    This function is deliberately conservative.  It accepts common ICON-shaped
    outputs such as ``(B, 1, N, C)`` or already-correct ``(B, N, C)`` but raises
    for ambiguous outputs rather than silently dropping a batch dimension.
    """

    device = data["query_input_vals"].device
    if isinstance(prediction, torch.Tensor):
        out = prediction.to(device=device)
    else:
        out = torch.as_tensor(prediction, device=device)

    batch, n_target, out_channels = query_target_shape(data, channels=channels)

    # Remove only the singleton query/demo axis.  Avoid blind squeeze().
    if out.dim() == 4 and out.shape[1] == 1:
        out = out[:, 0]

    if out.dim() == 2:
        out = out.unsqueeze(-1)

    if out.dim() != 3:
        raise ValueError(f"Prediction must be rank 3 after shape normalization, got {tuple(out.shape)}")
    if int(out.shape[0]) != batch:
        raise ValueError(f"Prediction batch mismatch: expected {batch}, got {int(out.shape[0])}")
    if int(out.shape[1]) != n_target:
        raise ValueError(f"Prediction target-length mismatch: expected {n_target}, got {int(out.shape[1])}")

    # The current evaluator targets one output channel.  If a model emits extra
    # channels, keep the first rather than returning the wrong shape.
    if int(out.shape[2]) != out_channels:
        if int(out.shape[2]) > out_channels:
            out = out[..., :out_channels]
        else:
            raise ValueError(f"Prediction channel mismatch: expected {out_channels}, got {int(out.shape[2])}")
    return out


def ensure_finite(output: torch.Tensor) -> torch.Tensor:
    """Reject invalid tensors before the evaluator failure path is triggered."""

    if not torch.isfinite(output).all():
        raise ValueError("Prediction contains NaN or Inf")
    return output


def per_sample_rel_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-batch relative L2 in original target scale."""

    pred = prediction.reshape(prediction.shape[0], -1)
    true = target.reshape(target.shape[0], -1)
    numer = torch.linalg.vector_norm(pred - true, dim=1)
    denom = torch.linalg.vector_norm(true, dim=1).clamp_min(EPS)
    return numer / denom


def candidate_name(spec: Mapping[str, Any]) -> str:
    return str(spec.get("name") or "unnamed")
