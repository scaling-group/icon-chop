"""Single fixed operator chain for the 1D periodic cubic conservation law.

Pipeline:

    F_scale o F_shift  ->  ICON  ->  G_unshift  ->  G_unscale  ->  G_mass

All three F/G operators are exact symmetries of the PDE family:

    F_scale  : v = (u - mu) / sigma         (closed under cubic flux family)
    F_shift  : integer cyclic translation   (exact translation symmetry)
    G_mass   : orthogonal L^2 projection onto the conserved-mean affine subspace

There is no selector and no parametric residual estimator: every parameter
(mu, sigma, integer s) is a sufficient statistic of the visible prompt, and
the chain is unconditionally beneficial under the LOO probe used during
development.
"""

from __future__ import annotations

import torch

from .operators import (
    ScaleTransform,
    fit_scale,
    fit_shift,
    project_mass,
    shift_data_forward,
    shift_output_backward,
)

DEMO_INPUT_T = (0, 1, 2, 3, 4)
DEMO_TARGET_T = (1, 2, 3, 4, 5)
QUERY_INPUT_T = (5,)
QUERY_TARGET_T = 6


def chain_zscore_shift_mass(data: dict[str, torch.Tensor], model) -> torch.Tensor:
    """F_scale o F_shift -> ICON -> G_unshift -> G_unscale -> G_mass."""

    shift_state = fit_shift(data)
    rolled = shift_data_forward(
        data,
        shift_state,
        demo_input_t=DEMO_INPUT_T,
        demo_target_t=DEMO_TARGET_T,
        query_input_t=QUERY_INPUT_T,
    )
    scale_state = fit_scale(rolled)
    prediction = model.predict(transform=ScaleTransform(scale_state), data=rolled)
    prediction = shift_output_backward(prediction, shift_state, query_target_t=QUERY_TARGET_T)
    return project_mass(prediction, data["query_input_vals"])
