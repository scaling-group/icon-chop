"""Single fixed operator chain for the 1D periodic cubic conservation law.

Pipeline:

    F_scale o F_shift  ->  ICON  ->  G_unshift  ->  G_unscale  ->  G_mass

All three operators are exact symmetries of the PDE family.  There is no
selector and no parametric residual estimator: every parameter is a
sufficient statistic of the visible prompt.
"""

from __future__ import annotations

import torch

from .chains import chain_zscore_shift_mass


def run_icon_chop(context: dict[str, object]) -> torch.Tensor:
    """Execute the single fixed chain on a batch and return predictions [B, P, C]."""

    return chain_zscore_shift_mass(context["data"], context["model"])


__all__ = ["run_icon_chop"]
