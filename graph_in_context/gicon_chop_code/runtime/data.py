from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .air_dataset import GiconFramesQueryDataset
from .graph import Graph

import numpy as np
import torch


METEO_VARS = ["t2m", "d2m", "tp", "sp", "blh", "msdwswrf", "u100", "v100"]


@dataclass(frozen=True)
class GiconDataConfig:
    data_dir: Path
    dataset_name: str = "bthsa"
    query_start: str = "2023-01-01T00:00:00"
    query_end: str = "2023-01-03T00:00:00"
    context_start: str = "2016-01-01T00:00:00"
    context_end: str = "2022-12-31T00:00:00"
    data_start: str = "2016-01-01T00:00:00"
    data_end: str = "2023-12-31T00:00:00"
    delta_t: int = 24
    window_size: int = 24
    demo_num: int = 5
    query_sample_mode: str = "random"
    seed: int = 42
    context_strategy: str = "random"
    retrieval_index_path: Path | None = None
    similar_len: int = 24


def default_checkpoint_path(dataset_name: str) -> Path:
    return Path("checkpoints") / dataset_name / "multiple_demo_num5_best" / "step_90000.ckpt"


def default_station_path(data_dir: Path, dataset_name: str) -> Path:
    return data_dir / f"stations_{dataset_name}.csv"


def default_altitude_path(data_dir: Path) -> Path:
    return data_dir / "altitude.npy"


def build_graph(station_file: Path, altitude_file: Path, use_altitude: bool = True) -> Graph:
    return Graph(str(station_file), str(altitude_file), use_altitude=use_altitude)


def build_query_dataset(config: GiconDataConfig) -> GiconFramesQueryDataset:
    return GiconFramesQueryDataset(
        path=str(config.data_dir),
        dataset_name=config.dataset_name,
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


def valid_query_indices(dataset: GiconFramesQueryDataset) -> list[int]:
    return [
        quest_time
        for quest_time in range(dataset.quest_start_idx, dataset.quest_end_idx + 1)
        if quest_time + dataset.delta_t < len(dataset.combined_data)
    ]


def select_query_indices(
    query_indices: list[int],
    num_samples: int,
    seed: int,
    sample_mode: str,
) -> list[int]:
    if num_samples <= 0 or num_samples >= len(query_indices):
        return query_indices

    mode = sample_mode.lower()
    count = min(num_samples, len(query_indices))
    if mode == "random":
        selected = np.random.default_rng(seed).choice(np.asarray(query_indices), size=count, replace=False)
        return sorted(int(query_idx) for query_idx in selected)
    if mode in {"sequential", "first"}:
        return query_indices[:count]
    raise ValueError("query_sample_mode must be 'random' or 'sequential'")


def select_demo_times(
    dataset: GiconFramesQueryDataset,
    query_idx: int,
    demo_num: int,
    seed: int,
) -> np.ndarray:
    valid_demo_times = np.asarray(
        [
            time_idx
            for time_idx in range(dataset.context_start_idx, dataset.context_end_idx + 1)
            if time_idx >= dataset.window_size - 1 and time_idx + dataset.delta_t < query_idx
        ],
        dtype=np.int64,
    )
    if len(valid_demo_times) < demo_num:
        raise ValueError(f"Need {demo_num} demos before query index {query_idx}, got {len(valid_demo_times)}")
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


def validate_retrieval_demo_times(
    dataset: GiconFramesQueryDataset,
    query_idx: int,
    demo_times: np.ndarray,
    demo_num: int,
) -> None:
    if len(demo_times) != demo_num:
        raise ValueError(f"Expected {demo_num} demos for query_idx={query_idx}, got {len(demo_times)}")
    for time_idx in demo_times:
        if time_idx < dataset.window_size - 1:
            raise ValueError(f"Demo time lacks full context window: {time_idx}")
        if time_idx + dataset.delta_t > dataset.context_end_idx:
            raise ValueError(f"Demo target exceeds context range: {time_idx}+{dataset.delta_t}")
        if time_idx + dataset.delta_t >= query_idx:
            raise ValueError(f"Demo target is not before query: demo={time_idx}, query={query_idx}")


def build_episode(
    dataset: GiconFramesQueryDataset,
    query_idx: int,
    demo_num: int,
    seed: int,
    context_strategy: str = "random",
    retrieval_index: RetrievalDemoIndex | None = None,
) -> dict[str, Any]:
    if context_strategy == "random":
        demo_times = select_demo_times(dataset, query_idx=query_idx, demo_num=demo_num, seed=seed)
        retrieval_similarities = None
    elif context_strategy == "retrieval":
        if retrieval_index is None:
            raise ValueError("context_strategy=retrieval requires a retrieval index")
        demo_times = retrieval_index.demos_for(query_idx)
        validate_retrieval_demo_times(dataset, query_idx, demo_times, demo_num)
        retrieval_similarities = retrieval_index.similarities_for(query_idx)
    else:
        raise ValueError("context_strategy must be 'random' or 'retrieval'")

    demo_cond_v = np.stack([dataset.get_window_data(int(time_idx)) for time_idx in demo_times], axis=0)
    demo_qoi_v = np.stack([dataset.combined_data[int(time_idx) + dataset.delta_t] for time_idx in demo_times], axis=0)
    quest_cond_v = dataset.get_window_data(query_idx)
    label = dataset.combined_data[query_idx + dataset.delta_t]

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
            "target_idx": int(query_idx + dataset.delta_t),
            "target_time": str(dataset.timetable[query_idx + dataset.delta_t]),
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


def collate_episodes(episodes: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    data_keys = ["demo_cond_v", "demo_qoi_v", "quest_cond_v"]
    return {
        "data": {
            key: torch.cat([episode["data"][key] for episode in episodes], dim=0).to(device)
            for key in data_keys
        },
        "label": torch.cat([episode["label"] for episode in episodes], dim=0).to(device),
        "combined_mean": torch.cat([episode["combined_mean"] for episode in episodes], dim=0).to(device),
        "combined_std": torch.cat([episode["combined_std"] for episode in episodes], dim=0).to(device),
        "metadata": [episode["metadata"] for episode in episodes],
    }


def build_episode_batches(
    config: GiconDataConfig,
    batch_size: int,
    num_samples: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    dataset = build_query_dataset(config)
    context_strategy = config.context_strategy.lower()
    if context_strategy not in {"random", "retrieval"}:
        raise ValueError("context_strategy must be 'random' or 'retrieval'")
    retrieval_index = None
    if context_strategy == "retrieval":
        if config.retrieval_index_path is None:
            raise ValueError("retrieval_index_path is required when context_strategy='retrieval'")
        retrieval_index = RetrievalDemoIndex(config.retrieval_index_path, demo_num=config.demo_num)

    query_indices = valid_query_indices(dataset)
    if not query_indices:
        raise ValueError("No valid query samples were produced by the query dataset")
    query_indices = select_query_indices(
        query_indices=query_indices,
        num_samples=num_samples,
        seed=config.seed,
        sample_mode=config.query_sample_mode,
    )

    batches: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for sample_offset, query_idx in enumerate(query_indices):
        pending.append(
            build_episode(
                dataset=dataset,
                query_idx=query_idx,
                demo_num=config.demo_num,
                seed=config.seed + sample_offset,
                context_strategy=context_strategy,
                retrieval_index=retrieval_index,
            )
        )
        if len(pending) == batch_size:
            batches.append(collate_episodes(pending, device))
            pending = []
    if pending:
        batches.append(collate_episodes(pending, device))
    return batches
