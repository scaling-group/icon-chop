"""Editable postprocessing stage G for GICON-CHOP (final).

Applies the convex combination solved in the preprocessing stage: per
pollutant channel the output is sum_k a_k * f_k over the named atom
forecasts, with the weights on the probability simplex. The GICON-relative
atoms (bias corrections) are completed here because the query GICON forecast
only exists after the mandatory main call.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .utils import ensure_finite


def _expected_prediction_shape(quest_cond: torch.Tensor) -> tuple[int, int, int, int]:
    return (
        int(quest_cond.shape[0]),
        1,
        int(quest_cond.shape[-2]),
        int(quest_cond.shape[-1]),
    )


def _atom_forecast(
    name: str,
    channel: int,
    output: torch.Tensor,
    combo: Mapping[str, Any],
) -> torch.Tensor:
    if name == combo["base_atom"]:
        return output[..., channel]
    if name == "bias_corrected":
        return output[..., channel] + combo["query_residual"][..., channel]
    if name == "bias_corrected_smooth":
        return output[..., channel] + combo["query_residual_smooth"][..., channel]
    prior = combo["query_priors"][name]
    if tuple(prior.shape) != tuple(output.shape):
        raise ValueError(f"Atom {name!r} shape {tuple(prior.shape)} != output shape {tuple(output.shape)}")
    return prior[..., channel]


def _apply_convex_combination(output: torch.Tensor, combo: Mapping[str, Any]) -> torch.Tensor:
    atoms = list(combo["atoms"])
    weights = combo["weights"]
    weights_t = weights.to(device=output.device, dtype=output.dtype) if isinstance(weights, torch.Tensor) else torch.as_tensor(
        weights,
        device=output.device,
        dtype=output.dtype,
    )
    if weights_t.ndim != 3 or int(weights_t.shape[0]) != int(output.shape[0]):
        raise ValueError(
            "Expected per-sample weights shaped (batch, channels, atoms), "
            f"got {tuple(weights_t.shape)} for output batch {int(output.shape[0])}"
        )
    start_channel = int(combo["start_channel"])

    result = output.clone()
    for offset in range(int(weights_t.shape[1])):
        channel = start_channel + offset
        if channel < 0 or channel >= int(output.shape[-1]):
            continue
        combined = torch.zeros_like(output[..., channel])
        for atom_index, name in enumerate(atoms):
            weight = weights_t[:, offset, atom_index].reshape(int(output.shape[0]), 1, 1)
            combined = combined + weight * _atom_forecast(name, channel, output, combo)
        result[..., channel] = combined
    return result


def postprocess_prediction(prediction: Any, processed: Mapping[str, Any]) -> torch.Tensor:
    """Complete and apply the probe-validated convex combination."""

    output = prediction if isinstance(prediction, torch.Tensor) else torch.as_tensor(prediction)
    quest_cond = processed["data"]["quest_cond_v"]
    expected_shape = _expected_prediction_shape(quest_cond)
    if tuple(output.shape) != expected_shape:
        raise ValueError(f"Expected GICON output shape {expected_shape}, got {tuple(output.shape)}")

    state = processed.get("chain_state") or {}
    combo = state.get("convex_combination") or {}
    if combo.get("enabled"):
        output = _apply_convex_combination(output, combo)
    return ensure_finite(output)
