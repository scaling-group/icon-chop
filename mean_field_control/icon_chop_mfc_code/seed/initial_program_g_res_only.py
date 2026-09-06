"""G_residual-only ablation for the ICON-CHOP MFC operator chain.

This entry point intentionally skips the F_value/G_scale path used by
``initial_program.py``.  It runs raw ICON on the query, estimates residual
transfer from raw leave-one-demo-out ICON probes, and applies only G_residual.
"""

from __future__ import annotations

import torch

from icon_chop.preprocess import (
    _as_query_axis,
    _estimate_residual_state,
    _heldout_targets,
    _raw_loo_probe_predictions,
)
from icon_chop.utils import clone_data, ensure_finite, normalize_to_query_shape


@torch.no_grad()
def run_icon_chop(context: dict[str, object]) -> torch.Tensor:
    model = context["model"]
    raw_data = clone_data(context["data"])
    for key in ("query_input_coords", "query_input_vals", "query_target_coords"):
        raw_data[key] = _as_query_axis(raw_data[key])

    raw_query_prediction = normalize_to_query_shape(model.predict(data=raw_data), raw_data)
    raw_probe_prediction = _raw_loo_probe_predictions(model, raw_data)
    probe_target = _heldout_targets(raw_data)
    residual_state = _estimate_residual_state(raw_data, raw_probe_prediction, probe_target)

    correction = residual_state["residual_correction"].to(
        device=raw_query_prediction.device,
        dtype=raw_query_prediction.dtype,
    )
    alpha = residual_state["residual_alpha"].to(
        device=raw_query_prediction.device,
        dtype=raw_query_prediction.dtype,
    )
    output = raw_query_prediction + alpha.reshape(
        int(raw_query_prediction.shape[0]),
        *([1] * (raw_query_prediction.dim() - 1)),
    ) * correction.reshape_as(raw_query_prediction)
    return ensure_finite(output)


run_icon_agent = run_icon_chop

__all__ = ["run_icon_chop", "run_icon_agent"]
