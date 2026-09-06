from __future__ import annotations

from typing import Any

import numpy as np
import torch


POLLUTANT_NAMES = ("pm25", "o3")


def _batch_rel_l2(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    flat_target = target.reshape(target.shape[0], -1)
    flat_prediction = prediction.reshape(prediction.shape[0], -1)
    numerator = torch.linalg.vector_norm(flat_target - flat_prediction, dim=1)
    denominator = torch.linalg.vector_norm(flat_target, dim=1)
    rel_l2 = torch.empty_like(numerator)
    nonzero = denominator != 0
    rel_l2[nonzero] = numerator[nonzero] / denominator[nonzero]
    rel_l2[~nonzero] = torch.where(
        numerator[~nonzero] == 0,
        torch.zeros_like(numerator[~nonzero]),
        torch.full_like(numerator[~nonzero], float("inf")),
    )
    return rel_l2


def _pollutant_channels(value: torch.Tensor) -> torch.Tensor:
    return value[..., -len(POLLUTANT_NAMES) :]


def denormalize_pollutants(
    value: torch.Tensor,
    combined_mean: torch.Tensor,
    combined_std: torch.Tensor,
) -> torch.Tensor:
    selected = _pollutant_channels(value)
    mean = combined_mean[:, None, None, -len(POLLUTANT_NAMES) :].to(device=value.device, dtype=value.dtype)
    std = combined_std[:, None, None, -len(POLLUTANT_NAMES) :].to(device=value.device, dtype=value.dtype)
    return selected * std + mean


def compute_prediction_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    combined_mean: torch.Tensor,
    combined_std: torch.Tensor,
) -> dict[str, np.ndarray]:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape mismatch: expected {tuple(target.shape)}, got {tuple(prediction.shape)}")

    norm_diff = prediction - target
    pred_norm_pol = _pollutant_channels(prediction)
    target_norm_pol = _pollutant_channels(target)
    pred_pol = denormalize_pollutants(prediction, combined_mean, combined_std)
    target_pol = denormalize_pollutants(target, combined_mean, combined_std)
    diff = pred_pol - target_pol

    rmse = torch.sqrt(torch.mean(diff * diff, dim=(1, 2)))
    mae = torch.mean(torch.abs(diff), dim=(1, 2))
    bias = torch.mean(diff, dim=(1, 2))
    return {
        "norm_mse": torch.mean(norm_diff * norm_diff, dim=tuple(range(1, norm_diff.dim()))).detach().cpu().numpy(),
        "pollutant_norm_mse": torch.mean((pred_norm_pol - target_norm_pol) ** 2, dim=(1, 2, 3))
        .detach()
        .cpu()
        .numpy(),
        "rel_l2": _batch_rel_l2(target_norm_pol, pred_norm_pol).detach().cpu().numpy(),
        "rmse": rmse.detach().cpu().numpy(),
        "mae": mae.detach().cpu().numpy(),
        "bias": bias.detach().cpu().numpy(),
    }


def rows_from_batch(
    *,
    metadata: list[dict[str, Any]],
    baseline_metrics: dict[str, np.ndarray],
    chain_metrics: dict[str, np.ndarray],
    success: bool,
    error: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch_size = len(metadata)
    for index in range(batch_size):
        row: dict[str, Any] = {
            "sample_index": index,
            "query_time": metadata[index]["query_time"],
            "target_time": metadata[index]["target_time"],
            "baseline_rel_l2": float(baseline_metrics["rel_l2"][index]),
            "chain_rel_l2": float(chain_metrics["rel_l2"][index]),
            "baseline_norm_mse": float(baseline_metrics["norm_mse"][index]),
            "chain_norm_mse": float(chain_metrics["norm_mse"][index]),
            "baseline_pollutant_norm_mse": float(baseline_metrics["pollutant_norm_mse"][index]),
            "chain_pollutant_norm_mse": float(chain_metrics["pollutant_norm_mse"][index]),
            "baseline_pollutant_rmse": float(np.mean(baseline_metrics["rmse"][index])),
            "chain_pollutant_rmse": float(np.mean(chain_metrics["rmse"][index])),
            "baseline_pollutant_mae": float(np.mean(baseline_metrics["mae"][index])),
            "chain_pollutant_mae": float(np.mean(chain_metrics["mae"][index])),
            "success": bool(success),
            "error": error,
            "metadata": metadata[index],
        }
        for channel, name in enumerate(POLLUTANT_NAMES):
            row[f"baseline_{name}_rmse"] = float(baseline_metrics["rmse"][index, channel])
            row[f"chain_{name}_rmse"] = float(chain_metrics["rmse"][index, channel])
            row[f"baseline_{name}_mae"] = float(baseline_metrics["mae"][index, channel])
            row[f"chain_{name}_mae"] = float(chain_metrics["mae"][index, channel])
            row[f"baseline_{name}_bias"] = float(baseline_metrics["bias"][index, channel])
            row[f"chain_{name}_bias"] = float(chain_metrics["bias"][index, channel])
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "score": 0.0,
            "baseline_rel_l2": float("nan"),
            "chain_rel_l2": float("nan"),
            "success_rate": 0.0,
            "failure_count": 0,
        }

    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    baseline_rel_l2 = mean("baseline_rel_l2")
    chain_rel_l2 = mean("chain_rel_l2")
    score = baseline_rel_l2 / chain_rel_l2 if chain_rel_l2 != 0.0 else float("inf")
    success_flags = np.asarray([bool(row["success"]) for row in rows], dtype=bool)
    summary = {
        "score": float(score),
        "baseline_rel_l2": baseline_rel_l2,
        "chain_rel_l2": chain_rel_l2,
        "baseline_norm_mse": mean("baseline_norm_mse"),
        "chain_norm_mse": mean("chain_norm_mse"),
        "baseline_pollutant_norm_mse": mean("baseline_pollutant_norm_mse"),
        "chain_pollutant_norm_mse": mean("chain_pollutant_norm_mse"),
        "baseline_pollutant_rmse": mean("baseline_pollutant_rmse"),
        "chain_pollutant_rmse": mean("chain_pollutant_rmse"),
        "baseline_pollutant_mae": mean("baseline_pollutant_mae"),
        "chain_pollutant_mae": mean("chain_pollutant_mae"),
        "success_rate": float(np.mean(success_flags)),
        "failure_count": int(np.sum(~success_flags)),
    }
    for name in POLLUTANT_NAMES:
        summary[f"baseline_{name}_rmse"] = mean(f"baseline_{name}_rmse")
        summary[f"chain_{name}_rmse"] = mean(f"chain_{name}_rmse")
        summary[f"baseline_{name}_mae"] = mean(f"baseline_{name}_mae")
        summary[f"chain_{name}_mae"] = mean(f"chain_{name}_mae")
    return summary
