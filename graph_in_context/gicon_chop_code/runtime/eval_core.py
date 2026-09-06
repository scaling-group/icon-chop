from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from .data import GiconDataConfig, build_episode_batches, build_graph

import numpy as np
import torch

from .metrics import POLLUTANT_NAMES, compute_prediction_metrics, rows_from_batch, summarize_rows
from .model import GiconWrapper, load_gicon_model, resolve_device


def load_program_module(program_path: str | Path):
    resolved_path = Path(program_path).resolve()
    module_path = resolved_path / "__init__.py" if resolved_path.is_dir() else resolved_path
    if not module_path.exists():
        raise FileNotFoundError(f"Program file not found: {module_path}")

    module_name = f"gicon_operator_chain_program_{abs(hash(str(resolved_path)))}"
    if module_path.name == "__init__.py":
        spec = importlib.util.spec_from_file_location(
            module_name,
            module_path,
            submodule_search_locations=[str(module_path.parent)],
        )
    else:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load program module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    program_dir = str(module_path.parent)
    if program_dir not in sys.path:
        sys.path.insert(0, program_dir)
    spec.loader.exec_module(module)
    return module


def _failure_metrics_like(baseline_metrics: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    failed: dict[str, np.ndarray] = {}
    for key, value in baseline_metrics.items():
        failed[key] = np.where(value == 0.0, 1.0, value * 2.0)
    return failed


def evaluate_program_file(
    *,
    program_path: str | Path,
    ckpt_path: str | Path,
    data_config: GiconDataConfig,
    station_file: str | Path,
    altitude_file: str | Path,
    use_altitude: bool,
    batch_size: int,
    num_samples: int,
    device_name: str | None,
) -> dict[str, Any]:
    module = load_program_module(program_path)
    if not hasattr(module, "run_gicon_chain"):
        raise AttributeError("Program must define `run_gicon_chain(context)`")

    device = resolve_device(device_name)
    model = load_gicon_model(ckpt_path, device=device)
    wrapper = GiconWrapper(model)
    graph = build_graph(Path(station_file), Path(altitude_file), use_altitude=use_altitude)
    batches = build_episode_batches(
        config=data_config,
        batch_size=batch_size,
        num_samples=num_samples,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    for batch in batches:
        target = batch["label"].to(device)
        combined_mean = batch["combined_mean"].to(device)
        combined_std = batch["combined_std"].to(device)

        baseline_prediction = wrapper.predict(data=batch["data"], graph=graph)
        baseline_metrics = compute_prediction_metrics(
            target=target,
            prediction=baseline_prediction,
            combined_mean=combined_mean,
            combined_std=combined_std,
        )

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
            chain_metrics = compute_prediction_metrics(
                target=target,
                prediction=chain_prediction,
                combined_mean=combined_mean,
                combined_std=combined_std,
            )
            success = True
            error = ""
        except Exception as exc:  # noqa: BLE001
            chain_metrics = _failure_metrics_like(baseline_metrics)
            success = False
            error = str(exc)

        rows.extend(
            rows_from_batch(
                metadata=batch["metadata"],
                baseline_metrics=baseline_metrics,
                chain_metrics=chain_metrics,
                success=success,
                error=error,
            )
        )

    return {
        "metrics": summarize_rows(rows),
        "diagnostics": {
            "num_batches": len(batches),
            "num_samples": len(rows),
            "device": str(device),
            "pollutants": list(POLLUTANT_NAMES),
            "context_strategy": data_config.context_strategy,
            "retrieval_index": (
                str(data_config.retrieval_index_path)
                if data_config.retrieval_index_path is not None
                else None
            ),
            "similar_len": int(data_config.similar_len),
            "graph": {
                "nodes": int(graph.node_num),
                "edges": int(graph.edge_num),
                "use_altitude": bool(use_altitude),
                "station_file": str(Path(station_file).resolve()),
                "altitude_file": str(Path(altitude_file).resolve()),
            },
        },
    }
