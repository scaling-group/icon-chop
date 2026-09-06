"""Atomic F / G operators for 1D periodic conservation laws.

The benchmark PDE family is

    d_t u + d_x f(u) = 0,    f(u) = a u^3 + b u^2 + c u,

with periodic boundary conditions on [0, 1) and a, b, c randomly drawn in
[-1, 1].  Three structural facts about this family motivate three operators:

  (i) Closure under affine value-substitution.  Setting v = (u - mu)/sigma
      yields the same cubic conservation law in v with coefficients
        (a', b', c') = (a sigma^2,  (3 a mu + b) sigma,  3 a mu^2 + 2 b mu + c),
      so F_scale maps the task into another member of the family ICON is
      already trained on.

 (ii) Translation invariance.  u(x, t) -> u(x - x_0, t) is an exact
      symmetry of the PDE.  Working in an integer co-moving frame is an
      exact cyclic permutation of the spatial grid.

(iii) Mass conservation.  Under periodic BC, the integral
      int_0^1 u(x, t) dx is conserved exactly in time.  G_mass is the
      orthogonal L^2 projection onto the affine subspace of fields whose
      spatial mean equals <u_5>.

Every operator below is either an exact bijection (F_scale, F_shift and
their inverses) or an exact orthogonal projection (G_mass).  The only
tunable quantity is the integer search bound max_shift = P // 4 used inside
the shift estimator; it is an upper bound on per-step characteristic
displacement and does not appear elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


# ---------------------------------------------------------------------------
# F_scale: per-sample affine substitution v = (u - mu) / sigma.
# Closure within the cubic flux family makes this a Lie-type symmetry of the
# multi-operator family ICON is trained on; it is not the Lie scaling of a
# single fixed PDE.
# ---------------------------------------------------------------------------


@dataclass
class ScaleState:
    mu: torch.Tensor   # [B]
    sigma: torch.Tensor  # [B]


def fit_scale(data: dict[str, torch.Tensor]) -> ScaleState:
    """Estimate (mu, sigma) from the pooled visible prompt values."""

    batch = int(data["demo_input_vals"].shape[0])
    visible = torch.cat(
        [
            data["demo_input_vals"].reshape(batch, -1),
            data["demo_target_vals"].reshape(batch, -1),
            data["query_input_vals"].reshape(batch, -1),
        ],
        dim=1,
    )
    mu = visible.mean(dim=1)
    sigma = visible.std(dim=1).clamp_min(torch.finfo(visible.dtype).eps)
    return ScaleState(mu=mu, sigma=sigma)


def _broadcast(stat: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Broadcast per-sample stat [B] to match value shape [B*k, ...] for any k."""

    stat = stat.view(-1)
    bs, bv = int(stat.shape[0]), int(value.shape[0])
    if bv == bs:
        expanded = stat
    elif bv % bs == 0:
        expanded = stat.repeat_interleave(bv // bs)
    else:
        raise ValueError(f"Batch size mismatch: stat={bs}, value={bv}")
    return expanded.to(value.device).view(-1, *([1] * (value.dim() - 1)))


def scale_forward(value: torch.Tensor, state: ScaleState) -> torch.Tensor:
    mu = _broadcast(state.mu, value)
    sigma = _broadcast(state.sigma, value)
    return (value - mu) / sigma


def scale_backward(value: torch.Tensor, state: ScaleState) -> torch.Tensor:
    mu = _broadcast(state.mu, value)
    sigma = _broadcast(state.sigma, value)
    return value * sigma + mu


class ScaleTransform:
    """Value-wise wrapper compatible with the IconWrapper transform API."""

    def __init__(self, state: ScaleState):
        self.state = state

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return scale_forward(value, self.state)

    def backward(self, value: torch.Tensor) -> torch.Tensor:
        return scale_backward(value, self.state)


# ---------------------------------------------------------------------------
# F_shift: cyclic translation in space.  For state index k along the trajectory,
# the operator applies roll(., -k * s), placing the entire prompt in an integer
# co-moving frame.  Inverse is roll(., +k * s) applied to the prediction at the
# query target time index.
# ---------------------------------------------------------------------------


@dataclass
class ShiftState:
    per_step: torch.Tensor  # [B] integer cells per time step


def estimate_integer_shift(
    last_input: torch.Tensor,
    last_target: torch.Tensor,
    max_shift: int,
) -> torch.Tensor:
    """Per-sample argmin over s in [-L, L] of ||roll(last_input, s) - last_target||^2.

    s = 0 is always inside the search set, so the estimator returns zero whenever
    no nonzero shift strictly reduces the alignment error.  L = max_shift acts as
    a CFL-style upper bound on per-step displacement and only affects compute.
    """

    points = int(last_input.shape[-2])
    shift_limit = max(1, min(int(max_shift), points // 2))
    shifts = torch.arange(-shift_limit, shift_limit + 1, device=last_input.device, dtype=torch.long)

    inputs = last_input - last_input.mean(dim=1, keepdim=True)
    targets = last_target - last_target.mean(dim=1, keepdim=True)

    errors = []
    for s in shifts.tolist():
        rolled = torch.roll(inputs, shifts=s, dims=1)
        errors.append(((rolled - targets) ** 2).mean(dim=(1, 2)))
    errors = torch.stack(errors, dim=1)
    return shifts[errors.argmin(dim=1)]


def fit_shift(data: dict[str, torch.Tensor]) -> ShiftState:
    """Estimate per-step shift from the most recent demo pair (u_{D-1} -> u_D)."""

    last_input = data["demo_input_vals"][:, -1]
    last_target = data["demo_target_vals"][:, -1]
    points = int(last_input.shape[-2])
    s = estimate_integer_shift(last_input, last_target, max_shift=points // 4)
    return ShiftState(per_step=s)


def _roll_per_time(values: torch.Tensor, per_step: torch.Tensor, time_indices: Sequence[int]) -> torch.Tensor:
    """For each state k in time_indices, apply roll(values_k, -k * per_step)."""

    batch, time, points, channels = values.shape
    if len(time_indices) != time:
        raise ValueError(f"Expected {time} time indices, got {len(time_indices)}")
    base = torch.arange(points, device=values.device, dtype=torch.long)
    per_step = per_step.to(device=values.device, dtype=torch.long).view(batch, 1)
    t_offsets = torch.as_tensor(time_indices, device=values.device, dtype=torch.long).view(1, time)
    shifts = -t_offsets * per_step
    gather_index = (base.view(1, 1, points) - shifts.view(batch, time, 1)) % points
    gather_index = gather_index.view(batch, time, points, 1).expand(batch, time, points, channels)
    return torch.gather(values, dim=-2, index=gather_index)


def shift_data_forward(
    data: dict[str, torch.Tensor],
    state: ShiftState,
    *,
    demo_input_t: Sequence[int],
    demo_target_t: Sequence[int],
    query_input_t: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Place the entire trajectory in the co-moving frame defined by `state`."""

    out = dict(data)
    s = state.per_step
    out["demo_input_vals"] = _roll_per_time(data["demo_input_vals"], s, demo_input_t)
    out["demo_target_vals"] = _roll_per_time(data["demo_target_vals"], s, demo_target_t)
    out["query_input_vals"] = _roll_per_time(data["query_input_vals"], s, query_input_t)
    return out


def shift_output_backward(prediction: torch.Tensor, state: ShiftState, query_target_t: int) -> torch.Tensor:
    """Invert F_shift on the ICON output at time index `query_target_t`."""

    batch, points, channels = prediction.shape
    s = state.per_step.to(prediction.device).long() * int(query_target_t)
    base = torch.arange(points, device=prediction.device, dtype=torch.long)
    gather_index = (base.view(1, points) - s.view(batch, 1)) % points
    gather_index = gather_index.view(batch, points, 1).expand(batch, points, channels)
    return torch.gather(prediction, dim=-2, index=gather_index)


# ---------------------------------------------------------------------------
# G_mass: orthogonal L^2 projection onto { v : <v> = <u_query_input> }.
# Non-expansive when the target itself lies in this affine subspace, which is
# exactly the case for periodic mass-conserving evolution.
# ---------------------------------------------------------------------------


def project_mass(prediction: torch.Tensor, query_input_vals: torch.Tensor) -> torch.Tensor:
    target_mean = query_input_vals.reshape(prediction.shape[0], -1).mean(dim=1)
    current_mean = prediction.reshape(prediction.shape[0], -1).mean(dim=1)
    correction = (target_mean - current_mean).view(-1, *([1] * (prediction.dim() - 1)))
    return prediction + correction
