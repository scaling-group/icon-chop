"""Minimal postprocessing stage G: G_scale (F_value inverse) then G_residual.

Output = G_scale(ICON(F_value(x))) + alpha * residual_transfer(x*),
where alpha and residual_transfer are estimated in preprocess.py from one
batched leave-one-demo-out ICON probe.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .utils import ensure_finite, normalize_to_query_shape


def _batch_view(value: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    return value.to(device=output.device, dtype=output.dtype).reshape(
        int(output.shape[0]),
        *([1] * (output.dim() - 1)),
    )


def _apply_value_inverse(output: torch.Tensor, state: Mapping[str, Any]) -> torch.Tensor:
    """G_scale: invert the target-side F_value gauge."""

    scale = state.get("value_scale")
    center = state.get("value_center")
    if scale is None:
        return output
    decoded = output * _batch_view(scale, output)
    if center is not None:
        decoded = decoded + _batch_view(center, output)
    return decoded


def _apply_residual(output: torch.Tensor, state: Mapping[str, Any]) -> torch.Tensor:
    """G_residual: add alpha * similarity-weighted LOO residual transfer."""

    correction = state.get("residual_correction")
    alpha = state.get("residual_alpha")
    if correction is None or alpha is None:
        return output
    correction_tensor = correction.to(device=output.device, dtype=output.dtype)
    if correction_tensor.shape != output.shape:
        correction_tensor = correction_tensor.reshape_as(output)
    return output + _batch_view(alpha, output) * correction_tensor


def _batch_mask(value: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    return value.to(device=output.device, dtype=torch.bool).reshape(
        int(output.shape[0]),
        *([1] * (output.dim() - 1)),
    )


def postprocess_prediction(prediction: Any, processed: Mapping[str, Any]) -> torch.Tensor:
    """Decode the ICON prediction with G_scale, add G_residual, fall back to raw."""

    output = normalize_to_query_shape(prediction, processed["data"])
    state = processed.get("chain_state") or {}
    output = _apply_value_inverse(output, state)
    output = _apply_residual(output, state)

    chain_used = state.get("chain_used")
    raw_query = state.get("raw_query_prediction")
    if chain_used is not None and raw_query is not None:
        raw_tensor = raw_query.to(device=output.device, dtype=output.dtype)
        if raw_tensor.shape != output.shape:
            raw_tensor = raw_tensor.reshape_as(output)
        output = torch.where(_batch_mask(chain_used, output), output, raw_tensor)

    return ensure_finite(output)
