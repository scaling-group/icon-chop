from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GICON_CODE_ROOT = REPO_ROOT / "graph_in_context" / "gicon_chop_code"
DEFAULT_INDEX_ROOT = Path(__file__).resolve().parent / "cache" / "retrieval_indices"


@dataclass(frozen=True)
class RetrievalBuildConfig:
    output_path: Path
    data_dir: Path
    eval_region: str
    query_start: str
    query_end: str
    context_start: str
    context_end: str
    data_start: str
    data_end: str
    delta_t: int
    demo_num: int
    window_size: int
    similar_len: int
    seed: int
    backend: str


def _bootstrap_runtime() -> None:
    root = str(GICON_CODE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _window_similarity_vector(dataset: Any, time_idx: int, similar_len: int) -> np.ndarray:
    window = dataset.get_window_data(int(time_idx))
    if similar_len > window.shape[0]:
        raise ValueError(f"similar_len={similar_len} exceeds window length={window.shape[0]}")
    window = window[-similar_len:, :, :]
    return np.ascontiguousarray(window.transpose(1, 0, 2).reshape(1, -1), dtype=np.float32)


def _valid_context_indices(dataset: Any, first_query_idx: int) -> np.ndarray:
    return np.asarray(
        [
            time_idx
            for time_idx in range(dataset.context_start_idx, dataset.context_end_idx + 1)
            if time_idx >= dataset.window_size - 1
            and time_idx + dataset.delta_t <= dataset.context_end_idx
            and time_idx + dataset.delta_t < first_query_idx
        ],
        dtype=np.int64,
    )


def _query_indices(dataset: Any) -> np.ndarray:
    return np.asarray(
        [
            query_idx
            for query_idx in range(dataset.quest_start_idx, dataset.quest_end_idx + 1)
            if query_idx >= dataset.window_size - 1 and query_idx + dataset.delta_t < len(dataset.combined_data)
        ],
        dtype=np.int64,
    )


def _build_feature_matrix(dataset: Any, context_indices: np.ndarray, similar_len: int) -> np.ndarray:
    matrix = np.empty(
        (
            int(context_indices.shape[0]),
            int(dataset.num_stations * similar_len * dataset.combined_data.shape[-1]),
        ),
        dtype=np.float32,
    )
    for row, time_idx in enumerate(context_indices):
        matrix[row] = _window_similarity_vector(dataset, int(time_idx), similar_len)[0]
    return _l2_normalize(matrix)


def _select_with_numpy(
    *,
    dataset: Any,
    feature_matrix: np.ndarray,
    context_indices: np.ndarray,
    query_indices: np.ndarray,
    demo_num: int,
    similar_len: int,
    query_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.empty((int(query_indices.shape[0]), demo_num), dtype=np.int64)
    scores = np.empty((int(query_indices.shape[0]), demo_num), dtype=np.float32)
    for start in range(0, int(query_indices.shape[0]), query_chunk_size):
        stop = min(start + query_chunk_size, int(query_indices.shape[0]))
        query_matrix = np.concatenate(
            [_window_similarity_vector(dataset, int(query_idx), similar_len) for query_idx in query_indices[start:stop]],
            axis=0,
        )
        query_matrix = _l2_normalize(query_matrix)
        similarities = query_matrix @ feature_matrix.T
        top_pos = np.argpartition(-similarities, kth=demo_num - 1, axis=1)[:, :demo_num]
        top_scores = np.take_along_axis(similarities, top_pos, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_pos = np.take_along_axis(top_pos, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        selected[start:stop] = context_indices[top_pos]
        scores[start:stop] = top_scores.astype(np.float32)
        print(f"selected retrieval demos for queries {start + 1}-{stop}/{len(query_indices)}", flush=True)
    return selected, scores


def _select_with_faiss(
    *,
    dataset: Any,
    feature_matrix: np.ndarray,
    context_indices: np.ndarray,
    query_indices: np.ndarray,
    demo_num: int,
    similar_len: int,
    query_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import faiss  # type: ignore[import-not-found]

    index = faiss.IndexFlatIP(feature_matrix.shape[1])
    index.add(feature_matrix)
    selected = np.empty((int(query_indices.shape[0]), demo_num), dtype=np.int64)
    scores = np.empty((int(query_indices.shape[0]), demo_num), dtype=np.float32)
    for start in range(0, int(query_indices.shape[0]), query_chunk_size):
        stop = min(start + query_chunk_size, int(query_indices.shape[0]))
        query_matrix = np.concatenate(
            [_window_similarity_vector(dataset, int(query_idx), similar_len) for query_idx in query_indices[start:stop]],
            axis=0,
        )
        query_matrix = _l2_normalize(query_matrix).astype(np.float32, copy=False)
        chunk_scores, chunk_positions = index.search(query_matrix, demo_num)
        selected[start:stop] = context_indices[chunk_positions]
        scores[start:stop] = chunk_scores.astype(np.float32)
        print(f"selected retrieval demos for queries {start + 1}-{stop}/{len(query_indices)}", flush=True)
    return selected, scores


def build_retrieval_index(
    config: RetrievalBuildConfig,
    *,
    query_chunk_size: int = 128,
) -> dict[str, Any]:
    _bootstrap_runtime()

    from runtime.air_dataset import GiconFramesQueryDataset
    from runtime.data import METEO_VARS

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
    query_indices = _query_indices(dataset)
    if len(query_indices) == 0:
        raise ValueError("Retrieval split produced no valid query samples")
    context_indices = _valid_context_indices(dataset, int(query_indices[0]))
    if len(context_indices) < config.demo_num:
        raise ValueError(f"Need {config.demo_num} context demos, got {len(context_indices)}")

    print(
        "building retrieval feature matrix: "
        f"context={len(context_indices)} dim={dataset.num_stations * config.similar_len * dataset.combined_data.shape[-1]}",
        flush=True,
    )
    feature_matrix = _build_feature_matrix(dataset, context_indices, config.similar_len)

    backend = config.backend
    if backend == "auto":
        try:
            import faiss  # noqa: F401

            backend = "faiss"
        except Exception:  # noqa: BLE001
            backend = "numpy"

    print(f"selecting retrieval demos with backend={backend}", flush=True)
    if backend == "faiss":
        demo_indices, similarities = _select_with_faiss(
            dataset=dataset,
            feature_matrix=feature_matrix,
            context_indices=context_indices,
            query_indices=query_indices,
            demo_num=config.demo_num,
            similar_len=config.similar_len,
            query_chunk_size=query_chunk_size,
        )
    elif backend == "numpy":
        demo_indices, similarities = _select_with_numpy(
            dataset=dataset,
            feature_matrix=feature_matrix,
            context_indices=context_indices,
            query_indices=query_indices,
            demo_num=config.demo_num,
            similar_len=config.similar_len,
            query_chunk_size=query_chunk_size,
        )
    else:
        raise ValueError(f"Unknown retrieval backend: {backend}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        **asdict(config),
        "output_path": str(config.output_path),
        "data_dir": str(config.data_dir),
        "backend_used": backend,
        "query_count": int(len(query_indices)),
        "context_count": int(len(context_indices)),
        "effective_query_start": str(dataset.timetable[int(query_indices[0])]),
        "effective_query_end": str(dataset.timetable[int(query_indices[-1])]),
        "effective_target_start": str(dataset.timetable[int(query_indices[0] + dataset.delta_t)]),
        "effective_target_end": str(dataset.timetable[int(query_indices[-1] + dataset.delta_t)]),
        "context_candidate_start": str(dataset.timetable[int(context_indices[0])]),
        "context_candidate_end": str(dataset.timetable[int(context_indices[-1])]),
        "selection": "Best",
        "similarity": "cosine",
    }
    np.savez_compressed(
        config.output_path,
        query_indices=query_indices,
        demo_indices=demo_indices,
        similarities=similarities,
        context_indices=context_indices,
        metadata=json.dumps(metadata, sort_keys=True),
    )
    sidecar = config.output_path.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def default_index_path(
    *,
    eval_region: str,
    query_start: str,
    delta_t: int,
    demo_num: int,
    window_size: int,
    similar_len: int,
) -> Path:
    name = (
        f"{eval_region}-{query_start[:4]}_multiframe_dt{delta_t}_"
        f"demo{demo_num}_win{window_size}_sim{similar_len}.npz"
    )
    return DEFAULT_INDEX_ROOT / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached GICON-CHOP retrieval demo indices without GPU.")
    parser.add_argument("--preset", choices=("none", "yrd-2023"), default="yrd-2023")
    parser.add_argument("--query-start", default=None)
    parser.add_argument("--query-end", default=None)
    parser.add_argument("--context-start", default=None)
    parser.add_argument("--context-end", default=None)
    parser.add_argument("--data-start", default=None)
    parser.add_argument("--data-end", default=None)
    parser.add_argument("--eval-region", default=None)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--demo-num", type=int, default=None)
    parser.add_argument("--delta-t", type=int, nargs="+", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--similar-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backend", choices=("auto", "faiss", "numpy"), default="auto")
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def _values_from_args(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {
        "query_start": args.query_start,
        "query_end": args.query_end,
        "context_start": args.context_start,
        "context_end": args.context_end,
        "data_start": args.data_start,
        "data_end": args.data_end,
        "eval_region": args.eval_region,
        "demo_num": args.demo_num,
        "window_size": args.window_size,
        "similar_len": args.similar_len,
        "seed": args.seed,
        "delta_t": args.delta_t,
    }
    if args.preset == "yrd-2023":
        preset = {
            "query_start": "2023-01-01T00:00:00",
            "query_end": "2023-12-31T23:00:00",
            "context_start": "2016-01-01T00:00:00",
            "context_end": "2022-12-31T23:00:00",
            "data_start": "2016-01-01T00:00:00",
            "data_end": "2023-12-31T23:00:00",
            "eval_region": "yrd",
            "demo_num": 5,
            "window_size": 24,
            "seed": 42,
            "delta_t": [24],
        }
        for key, value in preset.items():
            if values[key] is None:
                values[key] = value
    required = (
        "query_start",
        "query_end",
        "context_start",
        "context_end",
        "data_start",
        "data_end",
        "eval_region",
        "demo_num",
        "window_size",
        "seed",
        "delta_t",
    )
    missing = [key for key in required if values[key] is None]
    if missing:
        raise ValueError(f"Missing retrieval settings: {', '.join(missing)}")
    if values["similar_len"] is None:
        values["similar_len"] = values["window_size"]
    return values


def main() -> int:
    args = parse_args()
    values = _values_from_args(args)
    delta_ts = [int(value) for value in values["delta_t"]]
    if args.output_path and len(delta_ts) != 1:
        raise ValueError("--output-path can only be used with one --delta-t value")
    data_dir = Path(args.data_dir).expanduser().resolve()
    for delta_t in delta_ts:
        output_path = (
            Path(args.output_path).expanduser()
            if args.output_path
            else default_index_path(
                eval_region=str(values["eval_region"]),
                query_start=str(values["query_start"]),
                delta_t=delta_t,
                demo_num=int(values["demo_num"]),
                window_size=int(values["window_size"]),
                similar_len=int(values["similar_len"]),
            )
        )
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        config = RetrievalBuildConfig(
            output_path=output_path,
            data_dir=data_dir,
            eval_region=str(values["eval_region"]),
            query_start=str(values["query_start"]),
            query_end=str(values["query_end"]),
            context_start=str(values["context_start"]),
            context_end=str(values["context_end"]),
            data_start=str(values["data_start"]),
            data_end=str(values["data_end"]),
            delta_t=delta_t,
            demo_num=int(values["demo_num"]),
            window_size=int(values["window_size"]),
            similar_len=int(values["similar_len"]),
            seed=int(values["seed"]),
            backend=args.backend,
        )
        metadata = build_retrieval_index(config, query_chunk_size=args.query_chunk_size)
        print(f"retrieval index written to {config.output_path}")
        print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
