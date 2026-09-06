from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
GICON_CODE_ROOT = REPO_ROOT / "graph_in_context" / "gicon_chop_code"
DEFAULTS_ENV = Path(__file__).resolve().with_name("defaults.env")
POLLUTANTS = ("pm25", "o3")

PER_SAMPLE_METRIC_KEYS = (
    "score",
    "rel_l2_score",
    "baseline_rel_l2",
    "chain_rel_l2",
    "baseline_norm_mse",
    "chain_norm_mse",
    "baseline_pollutant_norm_mse",
    "chain_pollutant_norm_mse",
    "baseline_pollutant_rmse",
    "chain_pollutant_rmse",
    "baseline_pollutant_mae",
    "chain_pollutant_mae",
    "pm25_score",
    "baseline_pm25_rmse",
    "chain_pm25_rmse",
    "baseline_pm25_mae",
    "chain_pm25_mae",
    "baseline_pm25_bias",
    "chain_pm25_bias",
    "o3_score",
    "baseline_o3_rmse",
    "chain_o3_rmse",
    "baseline_o3_mae",
    "chain_o3_mae",
    "baseline_o3_bias",
    "chain_o3_bias",
)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("inf") if numerator > 0.0 else 1.0
    return float(numerator / denominator)


def _per_sample_rel_l2(diff: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch_size = int(target.shape[0])
    numerator = torch.linalg.vector_norm(diff.reshape(batch_size, -1), dim=1)
    denominator = torch.linalg.vector_norm(target.reshape(batch_size, -1), dim=1)
    result = torch.empty_like(numerator)
    nonzero = denominator != 0
    result[nonzero] = numerator[nonzero] / denominator[nonzero]
    result[~nonzero] = torch.where(
        numerator[~nonzero] == 0,
        torch.zeros_like(numerator[~nonzero]),
        torch.full_like(numerator[~nonzero], float("inf")),
    )
    return result


@dataclass(frozen=True)
class DirectYearConfig:
    program_path: Path
    data_dir: Path
    ckpt_path: Path
    station_file: Path
    altitude_file: Path
    model_region: str
    eval_region: str
    query_start: str
    query_end: str
    context_start: str
    context_end: str
    data_start: str
    data_end: str
    delta_t: int
    batch_size: int
    seed: int
    demo_num: int
    window_size: int
    context_strategy: str
    retrieval_index_path: Path | None
    similar_len: int
    device_name: str | None
    use_altitude: bool = True


class MetricAccumulator:
    def __init__(self) -> None:
        self.sample_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.baseline_norm_sse = 0.0
        self.chain_norm_sse = 0.0
        self.norm_count = 0
        self.baseline_pollutant_norm_sse = 0.0
        self.chain_pollutant_norm_sse = 0.0
        self.pollutant_norm_count = 0
        self.baseline_rel_num_sse = 0.0
        self.chain_rel_num_sse = 0.0
        self.rel_den_sse = 0.0
        self.baseline_pollutant_sse = 0.0
        self.chain_pollutant_sse = 0.0
        self.pollutant_count = 0
        self.baseline_channel_sse = np.zeros(len(POLLUTANTS), dtype=np.float64)
        self.chain_channel_sse = np.zeros(len(POLLUTANTS), dtype=np.float64)
        self.baseline_channel_abs = np.zeros(len(POLLUTANTS), dtype=np.float64)
        self.chain_channel_abs = np.zeros(len(POLLUTANTS), dtype=np.float64)
        self.baseline_channel_bias = np.zeros(len(POLLUTANTS), dtype=np.float64)
        self.chain_channel_bias = np.zeros(len(POLLUTANTS), dtype=np.float64)
        self.channel_count = 0
        self.per_sample_metrics: list[dict[str, Any]] = []

    def update(
        self,
        *,
        target: torch.Tensor,
        baseline_prediction: torch.Tensor,
        chain_prediction: torch.Tensor,
        combined_mean: torch.Tensor,
        combined_std: torch.Tensor,
        success_count: int,
        failure_count: int,
        metadata: list[dict[str, Any]],
        error: str = "",
    ) -> None:
        if baseline_prediction.shape != target.shape:
            raise ValueError(
                f"Baseline shape mismatch: expected {tuple(target.shape)}, got {tuple(baseline_prediction.shape)}"
            )
        if chain_prediction.shape != target.shape:
            raise ValueError(
                f"Chain shape mismatch: expected {tuple(target.shape)}, got {tuple(chain_prediction.shape)}"
            )

        batch_size = int(target.shape[0])
        if len(metadata) != batch_size:
            raise ValueError(f"Metadata batch mismatch: expected {batch_size}, got {len(metadata)}")
        if success_count + failure_count != batch_size:
            raise ValueError(
                "Success/failure count mismatch: "
                f"{success_count}+{failure_count} != {batch_size}"
            )
        sample_offset = self.sample_count
        self.sample_count += batch_size
        self.success_count += int(success_count)
        self.failure_count += int(failure_count)

        baseline_norm_diff = baseline_prediction - target
        chain_norm_diff = chain_prediction - target
        self.baseline_norm_sse += float(torch.sum(baseline_norm_diff * baseline_norm_diff).detach().cpu())
        self.chain_norm_sse += float(torch.sum(chain_norm_diff * chain_norm_diff).detach().cpu())
        self.norm_count += int(target.numel())

        target_norm_pol = target[..., -len(POLLUTANTS) :]
        baseline_norm_pol = baseline_prediction[..., -len(POLLUTANTS) :]
        chain_norm_pol = chain_prediction[..., -len(POLLUTANTS) :]
        baseline_norm_pol_diff = baseline_norm_pol - target_norm_pol
        chain_norm_pol_diff = chain_norm_pol - target_norm_pol
        self.baseline_pollutant_norm_sse += float(
            torch.sum(baseline_norm_pol_diff * baseline_norm_pol_diff).detach().cpu()
        )
        self.chain_pollutant_norm_sse += float(torch.sum(chain_norm_pol_diff * chain_norm_pol_diff).detach().cpu())
        self.pollutant_norm_count += int(target_norm_pol.numel())
        self.baseline_rel_num_sse += float(torch.sum(baseline_norm_pol_diff * baseline_norm_pol_diff).detach().cpu())
        self.chain_rel_num_sse += float(torch.sum(chain_norm_pol_diff * chain_norm_pol_diff).detach().cpu())
        self.rel_den_sse += float(torch.sum(target_norm_pol * target_norm_pol).detach().cpu())

        mean = combined_mean[:, None, None, -len(POLLUTANTS) :].to(device=target.device, dtype=target.dtype)
        std = combined_std[:, None, None, -len(POLLUTANTS) :].to(device=target.device, dtype=target.dtype)
        target_pol = target_norm_pol * std + mean
        baseline_pol = baseline_norm_pol * std + mean
        chain_pol = chain_norm_pol * std + mean
        baseline_diff = baseline_pol - target_pol
        chain_diff = chain_pol - target_pol

        self.baseline_pollutant_sse += float(torch.sum(baseline_diff * baseline_diff).detach().cpu())
        self.chain_pollutant_sse += float(torch.sum(chain_diff * chain_diff).detach().cpu())
        self.pollutant_count += int(target_pol.numel())
        self.channel_count += int(target_pol[..., 0].numel())
        self.baseline_channel_sse += torch.sum(baseline_diff * baseline_diff, dim=(0, 1, 2)).detach().cpu().numpy()
        self.chain_channel_sse += torch.sum(chain_diff * chain_diff, dim=(0, 1, 2)).detach().cpu().numpy()
        self.baseline_channel_abs += torch.sum(torch.abs(baseline_diff), dim=(0, 1, 2)).detach().cpu().numpy()
        self.chain_channel_abs += torch.sum(torch.abs(chain_diff), dim=(0, 1, 2)).detach().cpu().numpy()
        self.baseline_channel_bias += torch.sum(baseline_diff, dim=(0, 1, 2)).detach().cpu().numpy()
        self.chain_channel_bias += torch.sum(chain_diff, dim=(0, 1, 2)).detach().cpu().numpy()

        sample_dims = tuple(range(1, target.ndim))
        pollutant_dims = tuple(range(1, target_norm_pol.ndim))
        channel_dims = tuple(range(1, target_pol.ndim - 1))
        baseline_sample_rel_l2 = _per_sample_rel_l2(baseline_norm_pol_diff, target_norm_pol)
        chain_sample_rel_l2 = _per_sample_rel_l2(chain_norm_pol_diff, target_norm_pol)
        sample_values = {
            "baseline_rel_l2": baseline_sample_rel_l2,
            "chain_rel_l2": chain_sample_rel_l2,
            "baseline_norm_mse": torch.mean(baseline_norm_diff * baseline_norm_diff, dim=sample_dims),
            "chain_norm_mse": torch.mean(chain_norm_diff * chain_norm_diff, dim=sample_dims),
            "baseline_pollutant_norm_mse": torch.mean(
                baseline_norm_pol_diff * baseline_norm_pol_diff, dim=pollutant_dims
            ),
            "chain_pollutant_norm_mse": torch.mean(
                chain_norm_pol_diff * chain_norm_pol_diff, dim=pollutant_dims
            ),
            "baseline_pollutant_rmse": torch.sqrt(torch.mean(baseline_diff * baseline_diff, dim=pollutant_dims)),
            "chain_pollutant_rmse": torch.sqrt(torch.mean(chain_diff * chain_diff, dim=pollutant_dims)),
            "baseline_pollutant_mae": torch.mean(torch.abs(baseline_diff), dim=pollutant_dims),
            "chain_pollutant_mae": torch.mean(torch.abs(chain_diff), dim=pollutant_dims),
            "baseline_channel_rmse": torch.sqrt(torch.mean(baseline_diff * baseline_diff, dim=channel_dims)),
            "chain_channel_rmse": torch.sqrt(torch.mean(chain_diff * chain_diff, dim=channel_dims)),
            "baseline_channel_mae": torch.mean(torch.abs(baseline_diff), dim=channel_dims),
            "chain_channel_mae": torch.mean(torch.abs(chain_diff), dim=channel_dims),
            "baseline_channel_bias": torch.mean(baseline_diff, dim=channel_dims),
            "chain_channel_bias": torch.mean(chain_diff, dim=channel_dims),
        }
        sample_values = {key: value.detach().cpu().numpy() for key, value in sample_values.items()}
        batch_succeeded = success_count == batch_size
        for index, sample_metadata in enumerate(metadata):
            baseline_rel_l2 = float(sample_values["baseline_rel_l2"][index])
            chain_rel_l2 = float(sample_values["chain_rel_l2"][index])
            baseline_pollutant_rmse = float(sample_values["baseline_pollutant_rmse"][index])
            chain_pollutant_rmse = float(sample_values["chain_pollutant_rmse"][index])
            row: dict[str, Any] = {
                "sample_index": sample_offset + index,
                "query_idx": int(sample_metadata["query_idx"]),
                "query_time": str(sample_metadata["query_time"]),
                "target_idx": int(sample_metadata["target_idx"]),
                "target_time": str(sample_metadata["target_time"]),
                "success": batch_succeeded,
                "error": error,
                "score": _ratio(baseline_pollutant_rmse, chain_pollutant_rmse),
                "rel_l2_score": _ratio(baseline_rel_l2, chain_rel_l2),
                "baseline_rel_l2": baseline_rel_l2,
                "chain_rel_l2": chain_rel_l2,
                "baseline_norm_mse": float(sample_values["baseline_norm_mse"][index]),
                "chain_norm_mse": float(sample_values["chain_norm_mse"][index]),
                "baseline_pollutant_norm_mse": float(
                    sample_values["baseline_pollutant_norm_mse"][index]
                ),
                "chain_pollutant_norm_mse": float(sample_values["chain_pollutant_norm_mse"][index]),
                "baseline_pollutant_rmse": baseline_pollutant_rmse,
                "chain_pollutant_rmse": chain_pollutant_rmse,
                "baseline_pollutant_mae": float(sample_values["baseline_pollutant_mae"][index]),
                "chain_pollutant_mae": float(sample_values["chain_pollutant_mae"][index]),
            }
            for channel, name in enumerate(POLLUTANTS):
                baseline_rmse = float(sample_values["baseline_channel_rmse"][index, channel])
                chain_rmse = float(sample_values["chain_channel_rmse"][index, channel])
                row[f"{name}_score"] = _ratio(baseline_rmse, chain_rmse)
                row[f"baseline_{name}_rmse"] = baseline_rmse
                row[f"chain_{name}_rmse"] = chain_rmse
                row[f"baseline_{name}_mae"] = float(sample_values["baseline_channel_mae"][index, channel])
                row[f"chain_{name}_mae"] = float(sample_values["chain_channel_mae"][index, channel])
                row[f"baseline_{name}_bias"] = float(sample_values["baseline_channel_bias"][index, channel])
                row[f"chain_{name}_bias"] = float(sample_values["chain_channel_bias"][index, channel])
            self.per_sample_metrics.append(row)

    def metrics(self) -> dict[str, float]:
        baseline_rel_l2 = float(np.sqrt(self.baseline_rel_num_sse) / np.sqrt(self.rel_den_sse))
        chain_rel_l2 = float(np.sqrt(self.chain_rel_num_sse) / np.sqrt(self.rel_den_sse))
        baseline_pollutant_rmse = float(np.sqrt(self.baseline_pollutant_sse / self.pollutant_count))
        chain_pollutant_rmse = float(np.sqrt(self.chain_pollutant_sse / self.pollutant_count))
        payload: dict[str, float] = {
            "score": _ratio(baseline_pollutant_rmse, chain_pollutant_rmse),
            "baseline_rel_l2": baseline_rel_l2,
            "chain_rel_l2": chain_rel_l2,
            "rel_l2_score": _ratio(baseline_rel_l2, chain_rel_l2),
            "baseline_norm_mse": float(self.baseline_norm_sse / self.norm_count),
            "chain_norm_mse": float(self.chain_norm_sse / self.norm_count),
            "baseline_pollutant_norm_mse": float(self.baseline_pollutant_norm_sse / self.pollutant_norm_count),
            "chain_pollutant_norm_mse": float(self.chain_pollutant_norm_sse / self.pollutant_norm_count),
            "baseline_pollutant_rmse": baseline_pollutant_rmse,
            "chain_pollutant_rmse": chain_pollutant_rmse,
            "baseline_pollutant_mae": float(np.mean(self.baseline_channel_abs / self.channel_count)),
            "chain_pollutant_mae": float(np.mean(self.chain_channel_abs / self.channel_count)),
            "success_rate": float(self.success_count / self.sample_count) if self.sample_count else 0.0,
            "failure_count": float(self.failure_count),
        }
        for index, name in enumerate(POLLUTANTS):
            baseline_rmse = float(np.sqrt(self.baseline_channel_sse[index] / self.channel_count))
            chain_rmse = float(np.sqrt(self.chain_channel_sse[index] / self.channel_count))
            payload[f"baseline_{name}_rmse"] = baseline_rmse
            payload[f"chain_{name}_rmse"] = chain_rmse
            payload[f"{name}_score"] = _ratio(baseline_rmse, chain_rmse)
            payload[f"baseline_{name}_mae"] = float(self.baseline_channel_abs[index] / self.channel_count)
            payload[f"chain_{name}_mae"] = float(self.chain_channel_abs[index] / self.channel_count)
            payload[f"baseline_{name}_bias"] = float(self.baseline_channel_bias[index] / self.channel_count)
            payload[f"chain_{name}_bias"] = float(self.chain_channel_bias[index] / self.channel_count)
        payload["failure_count"] = int(self.failure_count)
        return payload

    def per_sample_metric_summary(self) -> dict[str, Any]:
        metrics: dict[str, dict[str, float]] = {}
        for key in PER_SAMPLE_METRIC_KEYS:
            values = np.asarray([float(row[key]) for row in self.per_sample_metrics], dtype=np.float64)
            metrics[key] = {
                "mean": float(np.mean(values)) if values.size else float("nan"),
                "std": float(np.std(values, ddof=0)) if values.size else float("nan"),
            }
        return {
            "count": len(self.per_sample_metrics),
            "std_ddof": 0,
            "metrics": metrics,
        }


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_defaults_env(path: Path = DEFAULTS_ENV) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    pattern = re.compile(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$')
    fallback_pattern = re.compile(r'^\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}$')
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = pattern.match(stripped)
        if not match:
            continue
        key, raw_value = match.groups()
        value = _strip_quotes(raw_value)
        fallback = fallback_pattern.match(value)
        if fallback:
            fallback_key, fallback_value = fallback.groups()
            value = os.environ.get(fallback_key, fallback_value)
        env[key] = value
    return env


def env_with_defaults(overrides: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in load_defaults_env().items():
        env.setdefault(key, value)
    env.update(overrides)
    return env


def _env_path(env: dict[str, str], key: str) -> Path:
    value = env.get(key)
    if not value:
        raise ValueError(f"Missing required environment value: {key}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = GICON_CODE_ROOT / path
    return path.resolve()


def _repo_relative_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def config_from_env(program_path: Path, overrides: dict[str, str]) -> DirectYearConfig:
    env = env_with_defaults(overrides)
    return DirectYearConfig(
        program_path=program_path.resolve(),
        data_dir=_env_path(env, "GICON_TESTBED_DATA_DIR"),
        ckpt_path=_env_path(env, "GICON_TESTBED_CKPT"),
        station_file=_env_path(env, "GICON_TESTBED_STATION_FILE"),
        altitude_file=_env_path(env, "GICON_TESTBED_ALTITUDE_FILE"),
        model_region=env.get("GICON_TESTBED_MODEL_REGION", "bthsa").lower(),
        eval_region=env.get("GICON_TESTBED_EVAL_REGION", "yrd").lower(),
        query_start=env.get("GICON_TESTBED_QUERY_START", env.get("GICON_TESTBED_TARGET_START", "")),
        query_end=env.get("GICON_TESTBED_QUERY_END", env.get("GICON_TESTBED_TARGET_END", "")),
        context_start=env["GICON_TESTBED_CONTEXT_START"],
        context_end=env["GICON_TESTBED_CONTEXT_END"],
        data_start=env["GICON_TESTBED_DATA_START"],
        data_end=env["GICON_TESTBED_DATA_END"],
        delta_t=int(env.get("GICON_TESTBED_DELTA_T", "24")),
        batch_size=int(env.get("GICON_TESTBED_BATCH_SIZE", "16")),
        seed=int(env.get("GICON_TESTBED_SEED", "42")),
        demo_num=int(env.get("GICON_TESTBED_DEMO_NUM", "5")),
        window_size=int(env.get("GICON_TESTBED_WINDOW_SIZE", "24")),
        context_strategy=env.get("GICON_TESTBED_CONTEXT_STRATEGY", "random").lower(),
        retrieval_index_path=(
            _repo_relative_path(env["GICON_TESTBED_RETRIEVAL_INDEX"])
            if env.get("GICON_TESTBED_RETRIEVAL_INDEX")
            else None
        ),
        similar_len=int(env.get("GICON_TESTBED_SIMILAR_LEN", env.get("GICON_TESTBED_WINDOW_SIZE", "24"))),
        device_name=env.get("GICON_TESTBED_DEVICE") or None,
        use_altitude=env.get("GICON_TESTBED_USE_ALTITUDE", "1").lower() not in {"0", "false", "no"},
    )


def _bootstrap_runtime() -> None:
    root = str(GICON_CODE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _select_direct_demo_times(
    *,
    dataset: Any,
    query_idx: int,
    demo_num: int,
    seed: int,
) -> np.ndarray:
    valid_demo_times = np.asarray(
        [
            time_idx
            for time_idx in range(dataset.context_start_idx, dataset.context_end_idx + 1)
            if time_idx >= dataset.window_size - 1
            and time_idx + dataset.delta_t <= dataset.context_end_idx
            and time_idx + dataset.delta_t < query_idx
        ],
        dtype=np.int64,
    )
    if len(valid_demo_times) < demo_num:
        raise ValueError(
            f"Need {demo_num} direct demos for query_idx={query_idx}, delta_t={dataset.delta_t}, "
            f"got {len(valid_demo_times)}"
        )
    return np.sort(np.random.default_rng(seed).choice(valid_demo_times, size=demo_num, replace=False))


class RetrievalDemoIndex:
    def __init__(self, path: Path, *, demo_num: int) -> None:
        payload = np.load(path, allow_pickle=False)
        query_indices = payload["query_indices"].astype(np.int64, copy=False)
        demo_indices = payload["demo_indices"].astype(np.int64, copy=False)
        if demo_indices.ndim != 2:
            raise ValueError(f"Retrieval demo_indices must be 2D, got shape={demo_indices.shape}")
        if demo_indices.shape[0] != query_indices.shape[0]:
            raise ValueError(
                "Retrieval query/demo row mismatch: "
                f"{query_indices.shape[0]} queries vs {demo_indices.shape[0]} demo rows"
            )
        if demo_indices.shape[1] < demo_num:
            raise ValueError(f"Retrieval index has {demo_indices.shape[1]} demos, need {demo_num}")
        self.path = path
        self.demo_num = demo_num
        self._demos = {
            int(query_idx): np.asarray(demo_indices[row, :demo_num], dtype=np.int64)
            for row, query_idx in enumerate(query_indices)
        }
        similarities = payload["similarities"].astype(np.float32, copy=False) if "similarities" in payload else None
        self._similarities = (
            {
                int(query_idx): np.asarray(similarities[row, :demo_num], dtype=np.float32)
                for row, query_idx in enumerate(query_indices)
            }
            if similarities is not None
            else {}
        )

    def demos_for(self, query_idx: int) -> np.ndarray:
        try:
            return self._demos[int(query_idx)]
        except KeyError as exc:
            raise KeyError(f"Retrieval index has no demo row for query_idx={query_idx}") from exc

    def similarities_for(self, query_idx: int) -> np.ndarray | None:
        return self._similarities.get(int(query_idx))


def _build_direct_episode(
    *,
    dataset: Any,
    query_idx: int,
    demo_num: int,
    seed: int,
    context_strategy: str,
    retrieval_index: RetrievalDemoIndex | None,
) -> dict[str, Any]:
    target_idx = int(query_idx + dataset.delta_t)
    if target_idx >= len(dataset.combined_data):
        raise ValueError(f"Target exceeds data: query_idx={query_idx}, target_idx={target_idx}")
    if context_strategy == "random":
        demo_times = _select_direct_demo_times(
            dataset=dataset,
            query_idx=query_idx,
            demo_num=demo_num,
            seed=seed,
        )
        retrieval_similarities = None
    elif context_strategy == "retrieval":
        if retrieval_index is None:
            raise ValueError("context_strategy=retrieval requires a retrieval index")
        demo_times = retrieval_index.demos_for(query_idx)
        retrieval_similarities = retrieval_index.similarities_for(query_idx)
    else:
        raise ValueError(f"Unknown context strategy: {context_strategy}")

    if len(demo_times) != demo_num:
        raise ValueError(f"Expected {demo_num} demos for query_idx={query_idx}, got {len(demo_times)}")
    for time_idx in demo_times:
        if time_idx < dataset.window_size - 1:
            raise ValueError(f"Demo time lacks full context window: {time_idx}")
        if time_idx + dataset.delta_t > dataset.context_end_idx:
            raise ValueError(f"Demo target exceeds context range: {time_idx}+{dataset.delta_t}")
        if time_idx + dataset.delta_t >= query_idx:
            raise ValueError(f"Demo target is not before query: demo={time_idx}, query={query_idx}")

    demo_cond_v = np.stack([dataset.get_window_data(int(time_idx)) for time_idx in demo_times], axis=0)
    demo_qoi_v = np.stack([dataset.combined_data[int(time_idx) + dataset.delta_t] for time_idx in demo_times], axis=0)
    quest_cond_v = dataset.get_window_data(query_idx)
    label = dataset.combined_data[target_idx]
    data = {
        "demo_cond_v": torch.tensor(demo_cond_v[None, ...], dtype=torch.float32),
        "demo_qoi_v": torch.tensor(demo_qoi_v[None, ...], dtype=torch.float32),
        "quest_cond_v": torch.tensor(quest_cond_v[None, None, ...], dtype=torch.float32),
    }
    return {
        "data": data,
        "label": torch.tensor(label[None, None, ...], dtype=torch.float32),
        "combined_mean": torch.tensor(dataset.combined_mean[None, ...], dtype=torch.float32),
        "combined_std": torch.tensor(dataset.combined_std[None, ...], dtype=torch.float32),
        "metadata": {
            "query_idx": int(query_idx),
            "query_time": str(dataset.timetable[query_idx]),
            "target_idx": int(target_idx),
            "target_time": str(dataset.timetable[target_idx]),
            "delta_t_hours": int(dataset.delta_t),
            "context_strategy": context_strategy,
            "demo_times": [
                {
                    "cond_end_idx": int(time_idx),
                    "cond_end_time": str(dataset.timetable[int(time_idx)]),
                    "qoi_idx": int(time_idx + dataset.delta_t),
                    "qoi_time": str(dataset.timetable[int(time_idx) + dataset.delta_t]),
                }
                for time_idx in demo_times
            ],
            "retrieval_similarities": (
                [float(value) for value in retrieval_similarities]
                if retrieval_similarities is not None
                else None
            ),
        },
    }


def _direct_batches(
    *,
    dataset: Any,
    query_indices: Iterable[int],
    demo_num: int,
    seed: int,
    context_strategy: str,
    retrieval_index: RetrievalDemoIndex | None,
    batch_size: int,
    device: torch.device,
) -> Iterable[dict[str, Any]]:
    from runtime.data import collate_episodes

    pending: list[dict[str, Any]] = []
    for offset, query_idx in enumerate(query_indices):
        pending.append(
            _build_direct_episode(
                dataset=dataset,
                query_idx=int(query_idx),
                demo_num=demo_num,
                seed=seed + offset,
                context_strategy=context_strategy,
                retrieval_index=retrieval_index,
            )
        )
        if len(pending) == batch_size:
            yield collate_episodes(pending, device)
            pending = []
    if pending:
        yield collate_episodes(pending, device)


def _summary(metrics: dict[str, float], diagnostics: dict[str, Any]) -> str:
    return (
        f"query_year_direct_score={metrics['score']:.6f}; "
        f"samples={diagnostics['sample_count']}; "
        f"delta_t={diagnostics['delta_t_hours']}h; "
        f"baseline_pollutant_rmse={metrics['baseline_pollutant_rmse']:.6f}; "
        f"chain_pollutant_rmse={metrics['chain_pollutant_rmse']:.6f}; "
        f"success_rate={metrics['success_rate']:.2%}"
    )


def evaluate_direct_year(config: DirectYearConfig) -> dict[str, Any]:
    _bootstrap_runtime()

    from runtime.air_dataset import GiconFramesQueryDataset
    from runtime.data import METEO_VARS, build_graph
    from runtime.eval_core import load_program_module
    from runtime.model import GiconWrapper, load_gicon_model, resolve_device

    module = load_program_module(config.program_path)
    if not hasattr(module, "run_gicon_chain"):
        raise AttributeError("Program must define `run_gicon_chain(context)`")
    if config.context_strategy not in {"random", "retrieval"}:
        raise ValueError("GICON_TESTBED_CONTEXT_STRATEGY must be 'random' or 'retrieval'")

    device = resolve_device(config.device_name)
    model = load_gicon_model(config.ckpt_path, device=device)
    wrapper = GiconWrapper(model)
    graph = build_graph(config.station_file, config.altitude_file, use_altitude=config.use_altitude)
    dataset = GiconFramesQueryDataset(
        path=config.data_dir,
        dataset_name=config.eval_region,
        quest_start_time=config.query_start,
        quest_end_time=config.query_end,
        context_start_time=config.context_start,
        context_end_time=config.context_end,
        meteo_var=METEO_VARS,
        meteo_use=METEO_VARS,
        data_start=config.data_start,
        data_end=config.data_end,
        delta_t=config.delta_t,
        window_size=config.window_size,
    )

    query_indices = [
        query_idx
        for query_idx in range(dataset.quest_start_idx, dataset.quest_end_idx + 1)
        if query_idx >= dataset.window_size - 1 and query_idx + dataset.delta_t < len(dataset.combined_data)
    ]
    if not query_indices:
        raise ValueError("Direct-year query range produced no valid query samples")

    retrieval_index = None
    if config.context_strategy == "retrieval":
        if config.retrieval_index_path is None:
            raise ValueError("GICON_TESTBED_RETRIEVAL_INDEX is required for retrieval context strategy")
        retrieval_index = RetrievalDemoIndex(config.retrieval_index_path, demo_num=config.demo_num)

    accumulator = MetricAccumulator()
    errors: list[str] = []
    with torch.no_grad():
        for batch in _direct_batches(
            dataset=dataset,
            query_indices=query_indices,
            demo_num=config.demo_num,
            seed=config.seed,
            context_strategy=config.context_strategy,
            retrieval_index=retrieval_index,
            batch_size=config.batch_size,
            device=device,
        ):
            target = batch["label"].to(device)
            combined_mean = batch["combined_mean"].to(device)
            combined_std = batch["combined_std"].to(device)
            baseline_prediction = wrapper.predict(data=batch["data"], graph=graph)
            try:
                context = {
                    "data": batch["data"],
                    "graph": graph,
                    "model": wrapper,
                    "combined_mean": combined_mean,
                    "combined_std": combined_std,
                    "metadata": batch["metadata"],
                }
                chain_prediction = module.run_gicon_chain(context).to(device)
                success_count = int(target.shape[0])
                failure_count = 0
                error_message = ""
            except Exception as exc:  # noqa: BLE001
                error_message = str(exc)
                if len(errors) < 5:
                    errors.append(error_message)
                chain_prediction = target + 2.0 * (baseline_prediction - target)
                success_count = 0
                failure_count = int(target.shape[0])

            accumulator.update(
                target=target,
                baseline_prediction=baseline_prediction,
                chain_prediction=chain_prediction,
                combined_mean=combined_mean,
                combined_std=combined_std,
                success_count=success_count,
                failure_count=failure_count,
                metadata=batch["metadata"],
                error=error_message,
            )

    metrics = accumulator.metrics()
    per_sample_metric_summary = accumulator.per_sample_metric_summary()
    diagnostics: dict[str, Any] = {
        "mode": "query_year_direct",
        "query_start": config.query_start,
        "query_end": config.query_end,
        "effective_query_start": str(dataset.timetable[int(query_indices[0])]),
        "effective_query_end": str(dataset.timetable[int(query_indices[-1])]),
        "effective_target_start": str(dataset.timetable[int(query_indices[0] + dataset.delta_t)]),
        "effective_target_end": str(dataset.timetable[int(query_indices[-1] + dataset.delta_t)]),
        "sample_count": int(len(query_indices)),
        "delta_t_hours": int(dataset.delta_t),
        "context_start": config.context_start,
        "context_end": config.context_end,
        "context_strategy": config.context_strategy,
        "retrieval_index": str(config.retrieval_index_path) if config.retrieval_index_path is not None else None,
        "similar_len": int(config.similar_len),
        "batch_size": int(config.batch_size),
        "demo_num": int(config.demo_num),
        "seed": int(config.seed),
        "device": str(device),
        "pollutants": list(POLLUTANTS),
        "graph": {
            "nodes": int(graph.node_num),
            "edges": int(graph.edge_num),
            "use_altitude": bool(config.use_altitude),
            "station_file": str(config.station_file),
            "altitude_file": str(config.altitude_file),
        },
    }
    if errors:
        diagnostics["errors"] = errors

    payload: dict[str, Any] = {
        "score": float(metrics["score"]),
        "pm25_score": float(metrics["pm25_score"]),
        "o3_score": float(metrics["o3_score"]),
        "summary": _summary(metrics, diagnostics),
        "program_path": str(config.program_path),
        "checkpoint_path": str(config.ckpt_path),
        "data_dir": str(config.data_dir),
        "dataset_name": config.eval_region,
        "model_region": config.model_region,
        "eval_region": config.eval_region,
        "metrics": metrics,
        "per_sample_metric_summary": per_sample_metric_summary,
        "per_sample_metrics": accumulator.per_sample_metrics,
        "diagnostics": diagnostics,
        "pollutants": list(POLLUTANTS),
    }
    payload["feedback"] = "\n".join(
        [
            payload["summary"],
            (
                f"baseline_pm25_rmse={metrics['baseline_pm25_rmse']:.6f} | "
                f"chain_pm25_rmse={metrics['chain_pm25_rmse']:.6f}"
            ),
            (
                f"baseline_o3_rmse={metrics['baseline_o3_rmse']:.6f} | "
                f"chain_o3_rmse={metrics['chain_o3_rmse']:.6f}"
            ),
        ]
    )
    return payload


def write_direct_outputs(payload: dict[str, Any], raw_json_path: Path, score_path: Path) -> None:
    raw_json_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    raw_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = dict(payload["metrics"])
    compact: dict[str, Any] = {
        "score": payload["score"],
        "pm25_score": payload["pm25_score"],
        "o3_score": payload["o3_score"],
        "summary": payload["summary"],
    }
    for key in (
        "baseline_pollutant_rmse",
        "chain_pollutant_rmse",
        "baseline_pm25_rmse",
        "chain_pm25_rmse",
        "baseline_o3_rmse",
        "chain_o3_rmse",
        "success_rate",
        "failure_count",
    ):
        compact[key] = metrics[key]
    if "per_sample_metric_summary" in payload:
        compact["per_sample_metric_summary"] = payload["per_sample_metric_summary"]
    import yaml

    score_path.write_text(yaml.safe_dump(compact, sort_keys=False), encoding="utf-8")

