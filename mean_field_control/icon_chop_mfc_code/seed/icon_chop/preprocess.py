"""Minimal preprocessing stage for the MFC operator chain.

Pipeline:

    F_value  ->  ICON  ->  G_scale (= F_value^-1)  ->  G_residual

F_value is a reversible per-batch value gauge: input and target sides share
one (center, scale) pair estimated from all visible prompt values.  Samples
whose visible values are non-negative use center=0 to preserve positive-field
structure; sign-changing samples use the shared visible mean.
G_residual transfers leave-one-demo-out residuals from one batched ICON probe
to the query target coordinates by input-similarity-weighted nearest-neighbor
interpolation; the scalar mixing alpha is fitted by demo cross-fit and gated
by visible LOO MSE.

All quantities derive only from visible prompt data (demo pairs + query
input).  Hidden query targets are never used.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .utils import EPS, clone_data, nearest_interpolate, normalize_to_query_shape


def _as_query_axis(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() >= 3 and int(tensor.shape[1]) == 1:
        return tensor
    return tensor.unsqueeze(1)


def _batch_scalar_view(stat: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return stat.to(device=value.device, dtype=value.dtype).reshape(
        int(value.shape[0]),
        *([1] * (value.dim() - 1)),
    )


def _flatten_fields(*values: torch.Tensor) -> torch.Tensor:
    batch = int(values[0].shape[0])
    return torch.cat([value.reshape(batch, -1) for value in values], dim=1)


def _shared_value_gauge(data: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate one reversible scalar gauge from all visible prompt values.

    This mirrors ``references/best_program2/icon_chop_unified``: demo inputs,
    demo targets, and query input are pooled, then one per-sample pair is used
    for both condition-side and target-side values.  For non-negative fields,
    the center is pinned to zero so the transform remains scale-only.
    """

    batch = int(data["demo_input_vals"].shape[0])
    visible = _flatten_fields(
        data["demo_input_vals"],
        data["demo_target_vals"],
        data["query_input_vals"],
    )
    eps = torch.finfo(visible.dtype).eps
    mean = visible.mean(dim=1, keepdim=True)
    std = visible.std(dim=1, keepdim=True)
    rms = torch.sqrt((visible * visible).mean(dim=1, keepdim=True).clamp_min(eps))
    is_nonnegative = visible.amin(dim=1, keepdim=True) >= -eps
    center = torch.where(is_nonnegative, torch.zeros_like(mean), mean).reshape(batch, 1, 1)
    scale = torch.where(is_nonnegative, rms, std).clamp_min(eps).reshape(batch, 1, 1)
    return center, scale


def _apply_f_transforms(
    data: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Apply F_value (one shared affine gauge) to a complete episode."""

    processed = clone_data(data)
    value_center, value_scale = _shared_value_gauge(data)

    for key in ("demo_input_vals", "query_input_vals", "demo_target_vals"):
        processed[key] = (
            data[key] - _batch_scalar_view(value_center, data[key])
        ) / _batch_scalar_view(value_scale, data[key])

    return processed, {
        "input_value_center": value_center,
        "input_value_scale": value_scale,
        "target_value_center": value_center,
        "target_value_scale": value_scale,
        "value_center": value_center,
        "value_scale": value_scale,
    }


# ---------------------------------------------------------------------------
# Batched leave-one-demo-out probe.
# ---------------------------------------------------------------------------


def _leave_one_context_indices(num_demos: int, device: torch.device) -> torch.Tensor:
    rows: list[list[int]] = []
    for held_out in range(num_demos):
        keep = [idx for idx in range(num_demos) if idx != held_out]
        while len(keep) < num_demos:
            keep.append(keep[0])
        rows.append(keep[:num_demos])
    return torch.tensor(rows, dtype=torch.long, device=device)


def _select_probe_context(demo_tensor: torch.Tensor, context_indices: torch.Tensor) -> torch.Tensor:
    batch, num_demos = int(demo_tensor.shape[0]), int(demo_tensor.shape[1])
    selected = demo_tensor[:, context_indices.to(device=demo_tensor.device)]
    return selected.reshape(batch * num_demos, num_demos, *demo_tensor.shape[2:])


def _heldout_query_tensor(demo_tensor: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    batch, num_demos = int(demo_tensor.shape[0]), int(demo_tensor.shape[1])
    held = demo_tensor.reshape(batch * num_demos, *demo_tensor.shape[2:])
    if tuple(held.shape[1:]) == tuple(template.shape[1:]):
        return held
    return held.unsqueeze(1)


def _build_probe_data(data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    num_demos = int(data["demo_input_vals"].shape[1])
    context_indices = _leave_one_context_indices(num_demos, data["demo_input_vals"].device)
    probe = clone_data(data)
    for key in ("demo_input_coords", "demo_input_vals", "demo_target_coords", "demo_target_vals"):
        probe[key] = _select_probe_context(data[key], context_indices)
    probe["query_input_coords"] = _heldout_query_tensor(
        data["demo_input_coords"], data["query_input_coords"]
    )
    probe["query_input_vals"] = _heldout_query_tensor(
        data["demo_input_vals"], data["query_input_vals"]
    )
    probe["query_target_coords"] = _heldout_query_tensor(
        data["demo_target_coords"], data["query_target_coords"]
    )
    return probe


def _heldout_targets(data: Mapping[str, torch.Tensor]) -> torch.Tensor:
    batch, num_demos = int(data["demo_target_vals"].shape[0]), int(data["demo_target_vals"].shape[1])
    target = data["demo_target_vals"].reshape(batch * num_demos, *data["demo_target_vals"].shape[2:])
    if target.dim() == 4 and int(target.shape[1]) == 1:
        target = target[:, 0]
    if target.dim() == 2:
        target = target.unsqueeze(-1)
    if target.dim() != 3:
        raise ValueError(f"Held-out targets must normalize to rank 3, got {tuple(target.shape)}")
    return target


@torch.no_grad()
def _loo_probe_predictions(
    model: Any,
    data: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one batched LOO probe through F_value -> ICON -> G_scale.

    Returns predictions and held-out targets in the original target scale.
    """

    batch = int(data["demo_input_vals"].shape[0])
    num_demos = int(data["demo_input_vals"].shape[1])
    probe_data = _build_probe_data(data)
    probe_processed, probe_state = _apply_f_transforms(probe_data)
    raw = model.predict(data=probe_processed)
    prediction = normalize_to_query_shape(raw, probe_processed)
    prediction = (
        prediction * _batch_scalar_view(probe_state["value_scale"], prediction)
        + _batch_scalar_view(probe_state["value_center"], prediction)
    )
    target = _heldout_targets(data)
    return (
        prediction.reshape(batch, num_demos, *prediction.shape[1:]),
        target.reshape(batch, num_demos, *target.shape[1:]),
    )


@torch.no_grad()
def _raw_loo_probe_predictions(
    model: Any,
    data: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Run one batched LOO probe through raw ICON (no F_value transform)."""

    batch = int(data["demo_input_vals"].shape[0])
    num_demos = int(data["demo_input_vals"].shape[1])
    probe_data = _build_probe_data(data)
    raw = model.predict(data=probe_data)
    prediction = normalize_to_query_shape(raw, probe_data)
    return prediction.reshape(batch, num_demos, *prediction.shape[1:])


@torch.no_grad()
def _raw_query_prediction(
    model: Any,
    data: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Run raw ICON on the actual query (no F_value transform).  Used as the
    per-sample safety fallback when the chain LOO MSE exceeds the raw LOO MSE.
    """

    raw = model.predict(data=data)
    return normalize_to_query_shape(raw, data)


# ---------------------------------------------------------------------------
# G_residual: similarity-weighted LOO residual transfer with alpha gate.
# ---------------------------------------------------------------------------


def _similarity_weights(
    source_coords: torch.Tensor,
    source_vals: torch.Tensor,
    target_coords: torch.Tensor,
    target_vals: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Softmax weights from visible input similarity with context-scaled distance."""

    source_on_target = nearest_interpolate(source_coords, source_vals, target_coords)
    if target_vals.dim() == 3:
        target_vals = target_vals.unsqueeze(1)
    diff = (source_on_target - target_vals) / _batch_scalar_view(scale, source_on_target)
    dist = (diff * diff).mean(dim=tuple(range(2, diff.dim())))
    temperature = dist.mean(dim=1, keepdim=True).clamp_min(EPS)
    return torch.softmax(-dist / temperature, dim=1)


def _weighted_sum(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights.unsqueeze(-1).unsqueeze(-1), dim=1)


def _fit_alpha(crossfit: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(1, crossfit.dim()))
    numerator = (crossfit * residuals).sum(dim=reduce_dims)
    denominator = (crossfit * crossfit).sum(dim=reduce_dims).clamp_min(EPS)
    return (numerator / denominator).clamp(0.0, 1.0).reshape(-1, 1, 1)


def _mse_by_batch(error: torch.Tensor) -> torch.Tensor:
    return (error * error).mean(dim=tuple(range(1, error.dim())))


def _estimate_residual_state(
    data: Mapping[str, torch.Tensor],
    probe_prediction: torch.Tensor,
    probe_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Transfer visible LOO residuals to the query by input-similarity weights."""

    batch, num_demos = int(probe_prediction.shape[0]), int(probe_prediction.shape[1])
    residuals = probe_target - probe_prediction
    _, scale = _shared_value_gauge(data)

    query_target_coords = data["query_target_coords"].expand(-1, num_demos, -1, -1)
    query_residual_fields = nearest_interpolate(
        data["demo_target_coords"],
        residuals,
        query_target_coords,
    )
    query_input_coords = data["query_input_coords"].expand(-1, num_demos, -1, -1)
    query_weights = _similarity_weights(
        data["demo_input_coords"],
        data["demo_input_vals"],
        query_input_coords,
        data["query_input_vals"],
        scale,
    )
    query_correction = _weighted_sum(query_residual_fields, query_weights)

    crossfit_parts: list[torch.Tensor] = []
    for hold in range(num_demos):
        keep = [idx for idx in range(num_demos) if idx != hold]
        held_target_coords = data["demo_target_coords"][:, hold : hold + 1].expand(
            -1, len(keep), -1, -1
        )
        projected = nearest_interpolate(
            data["demo_target_coords"][:, keep],
            residuals[:, keep],
            held_target_coords,
        )
        held_input_coords = data["demo_input_coords"][:, hold : hold + 1].expand(
            -1, len(keep), -1, -1
        )
        weights = _similarity_weights(
            data["demo_input_coords"][:, keep],
            data["demo_input_vals"][:, keep],
            held_input_coords,
            data["demo_input_vals"][:, hold : hold + 1],
            scale,
        )
        crossfit_parts.append(_weighted_sum(projected, weights))

    crossfit = torch.stack(crossfit_parts, dim=1)
    alpha = _fit_alpha(crossfit, residuals)
    raw_mse = _mse_by_batch(residuals)
    fitted_mse = _mse_by_batch(residuals - alpha.unsqueeze(1) * crossfit)
    used = torch.isfinite(fitted_mse) & (fitted_mse < raw_mse)
    alpha = torch.where(used.reshape(batch, 1, 1), alpha, torch.zeros_like(alpha))
    query_correction = torch.where(
        used.reshape(batch, 1, 1),
        query_correction,
        torch.zeros_like(query_correction),
    )
    return {
        "residual_correction": query_correction,
        "residual_alpha": alpha,
        "residual_used": used.reshape(batch, 1, 1),
        "residual_raw_mse": raw_mse.reshape(batch, 1, 1),
        "residual_fitted_mse": fitted_mse.reshape(batch, 1, 1),
    }


def _estimate_g_state(
    model: Any,
    data: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Estimate G_residual state and the raw-ICON safety fallback.

    Compares per-sample LOO MSE of the chain (F_value -> ICON -> G_scale ->
    G_residual) against raw ICON.  Samples where the chain LOO MSE exceeds the
    raw LOO MSE fall back to raw ICON on the actual query.
    """

    batch = int(data["demo_input_vals"].shape[0])
    n_target = int(data["query_target_coords"].shape[-2])
    channels = int(data["demo_target_vals"].shape[-1])
    num_demos = int(data["demo_input_vals"].shape[1])

    if num_demos < 2:
        return {
            "residual_correction": data["query_input_vals"].new_zeros(batch, n_target, channels),
            "residual_alpha": data["query_input_vals"].new_zeros(batch, 1, 1),
            "residual_used": data["query_input_vals"].new_zeros(batch, 1, 1, dtype=torch.bool),
            "chain_used": data["query_input_vals"].new_zeros(batch, 1, 1, dtype=torch.bool),
            "raw_query_prediction": _raw_query_prediction(model, data),
            "auxiliary_icon_calls": 1,
        }

    chain_probe_prediction, probe_target = _loo_probe_predictions(model, data)
    state = _estimate_residual_state(data, chain_probe_prediction, probe_target)

    chain_loo_mse = state["residual_fitted_mse"].reshape(batch)
    raw_probe_prediction = _raw_loo_probe_predictions(model, data)
    raw_loo_mse = _mse_by_batch(raw_probe_prediction - probe_target).reshape(batch)
    chain_used = torch.isfinite(chain_loo_mse) & (chain_loo_mse < raw_loo_mse)

    state["chain_used"] = chain_used.reshape(batch, 1, 1)
    state["raw_loo_mse"] = raw_loo_mse.reshape(batch, 1, 1)
    state["raw_query_prediction"] = _raw_query_prediction(model, data)
    state["auxiliary_icon_calls"] = 3
    return state


def preprocess_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare the ICON-compatible episode and visible-data G parameters."""

    raw_data = clone_data(context["data"])
    for key in ("query_input_coords", "query_input_vals", "query_target_coords"):
        raw_data[key] = _as_query_axis(raw_data[key])

    processed_data, f_state = _apply_f_transforms(raw_data)
    g_state = _estimate_g_state(context["model"], raw_data)

    return {
        "data": processed_data,
        "model": context["model"],
        "raw_data": raw_data,
        "chain_state": {
            "name": "minimal_residual_chain",
            "operators": ("F_value", "G_scale", "G_residual"),
            **f_state,
            **g_state,
        },
    }
