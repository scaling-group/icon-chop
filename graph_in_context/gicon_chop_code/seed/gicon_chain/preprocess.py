"""Editable preprocessing stage F for GICON-CHOP (final).

The final forecast is a single convex combination of named candidate
forecasts ("atoms"), each a parameter-free one-line formula:

    gicon                  frozen GICON forecast of the query
    persistence_latest     last visible frame of the query window
    persistence_start      first visible frame of the query window
    demo_mean              mean of the visible demo targets (climatology)
    nearest_analog         target of the most similar demo (Lorenz analog)
    bias_corrected         gicon + mean demo residual (MOS bias correction)
    bias_corrected_smooth  gicon + graph-neighbor-averaged mean demo residual

Per pollutant channel the weights are solved by constrained least squares on
the probability simplex, using error second moments measured on
leave-one-demo-out probes (the held-out demo stands in for the query), then
shrunk toward the pure GICON forecast by the derived factor
lambda = signal / (signal + noise), where signal is the probe-estimated loss
reduction of the solved combination and noise is tr(M V) with V the
jackknife covariance of the solved weights over probes. Exactly the
probe-validated combination is applied to the query.

Classical continuous parameters are spanned, not tuned: every damped
persistence prior latest - d*(latest - start), d in [0, 1], is a convex
combination of the two persistence atoms, and every partial graph smoothing
of the bias correction is a convex combination of the raw and fully averaged
bias atoms - damping and smoothing therefore emerge from the same least
squares. There are no weight grids, thresholds, margins, kernel
temperatures, mode selections, or per-channel special cases anywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .utils import clone_data, ensure_finite

# Data schema and compute batching. Not behavioral tuning values.
_POLLUTANT_CHANNELS = 2
_MAX_CV_PROBE_CONTEXTS = 16

# Numerical settings of the projected-gradient QP solver (convex problem;
# these control convergence precision only, not behavior).
_SOLVER_MAX_ITERS = 512
_SOLVER_TOL = 1.0e-9

_ATOMS = (
    "gicon",
    "persistence_latest",
    "persistence_start",
    "demo_mean",
    "nearest_analog",
    "bias_corrected",
    "bias_corrected_smooth",
)
_BASE_ATOM = "gicon"

DEFAULT_CANDIDATE: dict[str, Any] = {
    "name": "convex_combination_of_named_forecasts",
    "input_affine": "identity",
    "target_affine": "identity",
}


# ---------------------------------------------------------------------------
# Leave-one-demo-out probes
# ---------------------------------------------------------------------------


def _make_holdout_data(data: Mapping[str, torch.Tensor], hold: int) -> dict[str, torch.Tensor]:
    demo_count = int(data["demo_cond_v"].shape[1])
    keep = [idx for idx in range(demo_count) if idx != hold]
    return {
        "demo_cond_v": data["demo_cond_v"][:, keep, ...],
        "demo_qoi_v": data["demo_qoi_v"][:, keep, ...],
        "quest_cond_v": data["demo_cond_v"][:, hold : hold + 1, ...],
    }


def _concat_holdout_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = items[0].keys()
    return {key: torch.cat([item[key] for item in items], dim=0).contiguous() for key in keys}


@torch.no_grad()
def _demo_holdout_predictions(
    model: Any,
    graph: Any,
    data: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Frozen-GICON forecasts of each demo from the remaining demos, (D, B, 1, N, C)."""

    demo_count = int(data["demo_cond_v"].shape[1])
    batch_size = int(data["demo_cond_v"].shape[0])

    preds: list[torch.Tensor] = []
    contexts_per_call = max(1, min(demo_count, _MAX_CV_PROBE_CONTEXTS // max(batch_size, 1)))
    for chunk_start in range(0, demo_count, contexts_per_call):
        hold_indices = range(chunk_start, min(demo_count, chunk_start + contexts_per_call))
        holdout_items = [_make_holdout_data(data, hold) for hold in hold_indices]
        cv_data = _concat_holdout_batch(holdout_items)
        pred = model.predict(data=cv_data, graph=graph)
        preds.append(ensure_finite(pred).reshape(len(holdout_items), batch_size, *pred.shape[1:]))
    return torch.cat(preds, dim=0)


# ---------------------------------------------------------------------------
# Atom forecasts
# ---------------------------------------------------------------------------


def _condition_distances(demo_cond: torch.Tensor, quest_cond: torch.Tensor) -> torch.Tensor:
    """Per-channel std-normalized MSE between condition windows, (B, D).

    The per-channel scale only harmonizes units across physically different
    channels; the distance has no bandwidth or temperature.
    """

    batch_size = int(demo_cond.shape[0])
    channel_count = int(demo_cond.shape[-1])
    values = torch.cat(
        [
            demo_cond.reshape(batch_size, -1, channel_count),
            quest_cond.reshape(batch_size, -1, channel_count),
        ],
        dim=1,
    )
    scale = values.std(dim=1, unbiased=False).clamp_min(1.0e-4).reshape(batch_size, 1, 1, 1, channel_count)
    diff = (demo_cond - quest_cond) / scale
    return torch.mean(diff * diff, dim=tuple(range(2, diff.ndim)))


def _nearest_analog(
    demo_cond: torch.Tensor,
    demo_qoi: torch.Tensor,
    quest_cond: torch.Tensor,
) -> torch.Tensor:
    nearest = _condition_distances(demo_cond, quest_cond).argmin(dim=1)
    batch_index = torch.arange(int(demo_qoi.shape[0]), device=demo_qoi.device)
    return demo_qoi[batch_index, nearest].unsqueeze(1)


def _prior_atoms(
    demo_cond: torch.Tensor,
    demo_qoi: torch.Tensor,
    quest_cond: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Model-free atom forecasts for one case (query or held-out demo)."""

    if int(quest_cond.shape[-1]) != int(demo_qoi.shape[-1]):
        raise ValueError(
            "Persistence atoms require condition channels to match target "
            f"channels, got {int(quest_cond.shape[-1])} vs {int(demo_qoi.shape[-1])}"
        )
    return {
        "persistence_latest": quest_cond[:, :, -1, :, :],
        "persistence_start": quest_cond[:, :, 0, :, :],
        "demo_mean": demo_qoi.mean(dim=1, keepdim=True),
        "nearest_analog": _nearest_analog(demo_cond, demo_qoi, quest_cond),
    }


def _mean_other_residuals(residuals: torch.Tensor) -> torch.Tensor:
    demo_count = int(residuals.shape[0])
    if demo_count <= 1:
        return residuals
    total = residuals.sum(dim=0, keepdim=True)
    return (total - residuals) / float(demo_count - 1)


def _graph_edge_index(graph: Any, *, device: torch.device) -> torch.Tensor | None:
    edge_index: Any = None
    if isinstance(graph, Mapping):
        for key in ("edge_index", "edges", "edge_indices"):
            if key in graph:
                edge_index = graph[key]
                break
    else:
        for key in ("edge_index", "edges", "edge_indices"):
            if hasattr(graph, key):
                edge_index = getattr(graph, key)
                break
    if edge_index is None:
        return None

    if not isinstance(edge_index, torch.Tensor):
        try:
            edge_index = torch.as_tensor(edge_index)
        except (TypeError, ValueError):
            return None
    if edge_index.ndim != 2:
        return None
    if int(edge_index.shape[0]) != 2 and int(edge_index.shape[1]) == 2:
        edge_index = edge_index.t()
    if int(edge_index.shape[0]) != 2:
        return None
    return edge_index.to(device=device, dtype=torch.long)


def _neighbor_average(values: torch.Tensor, graph: Any) -> torch.Tensor:
    """Full graph-neighbor average; nodes without neighbors keep their value."""

    node_count = int(values.shape[-2])
    edge_index = _graph_edge_index(graph, device=values.device)
    if edge_index is None or int(edge_index.numel()) == 0 or node_count <= 1:
        return values

    src, dst = edge_index[0], edge_index[1]
    valid = (src >= 0) & (src < node_count) & (dst >= 0) & (dst < node_count)
    src = src[valid]
    dst = dst[valid]
    if int(src.numel()) == 0:
        return values
    src, dst = torch.cat([src, dst], dim=0), torch.cat([dst, src], dim=0)

    channel_count = int(values.shape[-1])
    flat = values.reshape(-1, node_count, channel_count)
    aggregate = torch.zeros_like(flat)
    aggregate.index_add_(1, dst, flat[:, src, :])
    degree = torch.zeros(node_count, device=values.device, dtype=values.dtype)
    degree.index_add_(0, dst, torch.ones_like(dst, dtype=values.dtype))

    neighbor_mean = aggregate / degree.clamp_min(1.0).reshape(1, node_count, 1)
    has_neighbors = (degree > 0).reshape(1, node_count, 1)
    neighbor_mean = torch.where(has_neighbors, neighbor_mean, flat)
    return neighbor_mean.reshape_as(values)


def _probe_atom_stack(
    data: Mapping[str, torch.Tensor],
    graph: Any,
    holdout_pred: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """All atom forecasts evaluated on the LOO probes, each (D, B, 1, N, C)."""

    demo_count = int(data["demo_cond_v"].shape[1])
    prior_stacks: dict[str, list[torch.Tensor]] = {}
    for hold in range(demo_count):
        holdout = _make_holdout_data(data, hold)
        priors = _prior_atoms(holdout["demo_cond_v"], holdout["demo_qoi_v"], holdout["quest_cond_v"])
        for name, value in priors.items():
            prior_stacks.setdefault(name, []).append(value)

    atoms = {name: torch.stack(stack, dim=0) for name, stack in prior_stacks.items()}
    target = data["demo_qoi_v"].transpose(0, 1).unsqueeze(2).contiguous()
    loo_residual = _mean_other_residuals(target - holdout_pred)
    atoms["gicon"] = holdout_pred
    atoms["bias_corrected"] = holdout_pred + loo_residual
    atoms["bias_corrected_smooth"] = holdout_pred + _neighbor_average(loo_residual, graph)
    return atoms


# ---------------------------------------------------------------------------
# Constrained least squares on the simplex, with derived shrinkage
# ---------------------------------------------------------------------------


def _project_to_simplex(vector: torch.Tensor) -> torch.Tensor:
    """Exact Euclidean projection onto the probability simplex (Duchi et al. 2008)."""

    sorted_v, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(sorted_v, dim=0) - 1.0
    counts = torch.arange(1, int(vector.numel()) + 1, device=vector.device, dtype=vector.dtype)
    support = sorted_v - cumulative / counts > 0
    rho = int(support.nonzero().max().item())
    theta = cumulative[rho] / float(rho + 1)
    return (vector - theta).clamp_min(0.0)


def _solve_simplex_weights(moment: torch.Tensor) -> torch.Tensor:
    """argmin_{a >= 0, sum a = 1} a^T M a via projected gradient (convex)."""

    atom_count = int(moment.shape[0])
    step = 1.0 / (2.0 * float(torch.linalg.eigvalsh(moment)[-1].clamp_min(1.0e-12)))
    weights = torch.full((atom_count,), 1.0 / atom_count, dtype=moment.dtype, device=moment.device)
    for _ in range(_SOLVER_MAX_ITERS):
        updated = _project_to_simplex(weights - step * 2.0 * (moment @ weights))
        if float((updated - weights).abs().max()) <= _SOLVER_TOL:
            weights = updated
            break
        weights = updated
    return weights


def _shrunk_simplex_weights(
    probe_moments: torch.Tensor,
    base_index: int,
) -> tuple[torch.Tensor, float]:
    """Solve the combination and shrink it toward the base atom.

    ``probe_moments`` is (D, K, K): per-probe second-moment matrices of the
    atom errors for one channel. With ``a_hat`` solved from the pooled moment
    M and V the jackknife covariance of the solution over probes, the
    expected loss of applying ``a_base + lambda * (a_hat - a_base)`` to the
    exactly quadratic loss is minimized at
    ``lambda = delta^T M delta / (delta^T M delta + tr(M V))`` with
    ``delta = a_hat - a_base`` - the vector generalization of the scalar SNR
    shrinkage w^2/(w^2+V). Conservatism is derived from probe noise, not
    tuned, and the shrunk weights stay on the simplex.
    """

    probes = int(probe_moments.shape[0])
    pooled = probe_moments.mean(dim=0)
    solved = _solve_simplex_weights(pooled)

    base = torch.zeros_like(solved)
    base[base_index] = 1.0
    delta = solved - base

    if probes > 1:
        total = probe_moments.sum(dim=0)
        loo_solutions = torch.stack(
            [_solve_simplex_weights((total - probe_moments[d]) / float(probes - 1)) for d in range(probes)],
            dim=0,
        )
        centered = loo_solutions - loo_solutions.mean(dim=0)
        jackknife_cov = centered.t() @ centered * float(probes - 1) / float(probes)
        signal = float(delta @ pooled @ delta)
        noise = float(torch.trace(pooled @ jackknife_cov))
        shrink = signal / (signal + noise) if signal + noise > 0 else 0.0
    else:
        shrink = 0.0

    return base + shrink * delta, shrink


# ---------------------------------------------------------------------------
# Assemble the combination state
# ---------------------------------------------------------------------------


def _solve_convex_combination(
    data: Mapping[str, torch.Tensor],
    model: Any,
    graph: Any,
) -> dict[str, Any]:
    demo_count = int(data["demo_cond_v"].shape[1])
    if demo_count < 2:
        return {"enabled": False, "reason": "not_enough_demos"}

    holdout_pred = _demo_holdout_predictions(model, graph, data)
    target = data["demo_qoi_v"].transpose(0, 1).unsqueeze(2).contiguous()
    channel_count = int(target.shape[-1])
    pollutant_count = min(_POLLUTANT_CHANNELS, channel_count)
    start_channel = channel_count - pollutant_count

    probe_atoms = _probe_atom_stack(data, graph, holdout_pred)
    errors = torch.stack(
        [probe_atoms[name][..., start_channel:] - target[..., start_channel:] for name in _ATOMS],
        dim=0,
    )
    # Keep the evaluation-batch axis: each query must solve its combination
    # from its own demonstrations.  Pooling ``b`` here makes a prediction
    # depend on whichever unrelated queries happen to share its runtime batch.
    elements = float(errors.shape[3] * errors.shape[4])
    probe_moments = torch.einsum("kdbxnc,jdbxnc->dbckj", errors, errors) / elements

    base_index = _ATOMS.index(_BASE_ATOM)
    batch_size = int(errors.shape[2])
    weights = holdout_pred.new_zeros(batch_size, pollutant_count, len(_ATOMS))
    shrink_factors: list[list[float]] = []
    for batch_index in range(batch_size):
        sample_shrink: list[float] = []
        for channel in range(pollutant_count):
            channel_weights, shrink = _shrunk_simplex_weights(
                probe_moments[:, batch_index, channel],
                base_index,
            )
            weights[batch_index, channel] = channel_weights
            sample_shrink.append(shrink)
        shrink_factors.append(sample_shrink)

    pooled = probe_moments.mean(dim=0)
    base_loss = [
        [float(pooled[b, c, base_index, base_index]) for c in range(pollutant_count)]
        for b in range(batch_size)
    ]
    predicted_loss = [
        [float(weights[b, c] @ pooled[b, c] @ weights[b, c]) for c in range(pollutant_count)]
        for b in range(batch_size)
    ]

    query_priors = _prior_atoms(data["demo_cond_v"], data["demo_qoi_v"], data["quest_cond_v"])
    query_residual = (target - holdout_pred).mean(dim=0)
    enabled = bool(torch.any(weights[..., base_index] < 1.0).detach().cpu().item())

    return {
        "enabled": enabled,
        "start_channel": start_channel,
        "atoms": list(_ATOMS),
        "base_atom": _BASE_ATOM,
        "weights": weights,
        "shrink": shrink_factors,
        "base_loss": base_loss,
        "predicted_loss": predicted_loss,
        "query_priors": query_priors,
        "query_residual": query_residual,
        "query_residual_smooth": _neighbor_average(query_residual, graph),
    }


def preprocess_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare data for the mandatory main GICON call."""

    raw_data = clone_data(context["data"])
    model = context["model"]
    graph = context["graph"]
    state: dict[str, Any] = {
        "candidate": dict(DEFAULT_CANDIDATE),
        "candidate_name": DEFAULT_CANDIDATE["name"],
        "convex_combination": _solve_convex_combination(raw_data, model, graph),
    }

    return {
        "data": raw_data,
        "graph": graph,
        "model": model,
        "raw_data": raw_data,
        "combined_mean": context.get("combined_mean"),
        "combined_std": context.get("combined_std"),
        "metadata": context.get("metadata"),
        "chain_state": state,
    }
