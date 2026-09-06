from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mfc.mfc_raw_operator import MfcRawOperatorDataset
from model.icon.loader import load_model
from model.icon.wrapper import IconWrapper

_MODEL_CACHE: dict[tuple[str, str, int], torch.nn.Module] = {}

SEMANTIC_KEYS = [
    "demo_input_coords",
    "demo_input_vals",
    "demo_target_coords",
    "demo_target_vals",
    "query_input_coords",
    "query_input_vals",
    "query_target_coords",
]


def resolve_device(device_name: str | None = None) -> torch.device:
    if device_name:
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"Requested device {device_name!r} but CUDA is unavailable")
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(ckpt_path: str, device: torch.device, in_features: int) -> torch.nn.Module:
    cache_key = (str(Path(ckpt_path).resolve()), str(device), int(in_features))
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = load_model(ckpt_path, device=device, in_features=in_features)
    return _MODEL_CACHE[cache_key]


def load_program_module(program_path: str):
    resolved_path = Path(program_path).resolve()
    module_name = f"icon_chop_mfc_program_{abs(hash(str(resolved_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load program module from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    program_dir = str(resolved_path.parent)
    if program_dir not in sys.path:
        sys.path.insert(0, program_dir)
    spec.loader.exec_module(module)
    return module


def _make_torch_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    rng = torch.Generator()
    rng.manual_seed(int(seed))
    return rng


def _randperm(length: int, seed: int | None) -> torch.Tensor:
    rng = _make_torch_generator(seed)
    if rng is None:
        return torch.randperm(length)
    return torch.randperm(length, generator=rng)


def _sample_id(dataset: MfcRawOperatorDataset, index: int) -> str:
    file_path, group_name = dataset.indices[index]
    return f"{Path(file_path).stem}:{group_name}"


def _shape_signature(sample: dict[str, object]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    data = sample["data"]
    return tuple((key, tuple(data[key].shape[1:])) for key in SEMANTIC_KEYS)


def _raw_collate(samples: list[dict[str, object]], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    data_dict = {key: torch.cat([sample["data"][key] for sample in samples], dim=0).to(device) for key in SEMANTIC_KEYS}
    target = torch.cat([sample["label"][:, 0] for sample in samples], dim=0).to(device)
    return data_dict, target


def _split_fixed_demo_query_sample(
    sample: dict[str, object],
    split_seed: int | None,
    demo_num: int,
) -> dict[str, object]:
    data = sample["data"]
    pair_count = int(data["candidate_input_vals"].shape[1])
    if pair_count < demo_num + 1:
        raise ValueError(f"Expected at least {demo_num + 1} candidate pairs, but found {pair_count}.")

    permutation = _randperm(pair_count, split_seed)
    demo_indices = permutation[:demo_num]
    query_indices = permutation[demo_num : demo_num + 1]
    query_index = int(query_indices[0].item())

    def _gather_pairs(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return tensor[:, indices]

    query_target_vals = _gather_pairs(data["candidate_target_vals"], query_indices)
    return {
        "description": sample["description"],
        "data": {
            "demo_input_coords": _gather_pairs(data["candidate_input_coords"], demo_indices),
            "demo_input_vals": _gather_pairs(data["candidate_input_vals"], demo_indices),
            "demo_target_coords": _gather_pairs(data["candidate_target_coords"], demo_indices),
            "demo_target_vals": _gather_pairs(data["candidate_target_vals"], demo_indices),
            "query_input_coords": _gather_pairs(data["candidate_input_coords"], query_indices),
            "query_input_vals": _gather_pairs(data["candidate_input_vals"], query_indices),
            "query_target_coords": _gather_pairs(data["candidate_target_coords"], query_indices),
        },
        "label": query_target_vals,
        "demo_indices": demo_indices.clone(),
        "query_index": query_index,
    }


def load_raw_operator_batches(
    data_source: str,
    batch_size: int,
    num_samples: int | None,
    seed: int,
    device: torch.device,
    max_len: int | None,
    k_dim: int,
    demo_num: int,
    time_subsample: int | None = None,
    space_subsample: int | None = None,
) -> list[dict[str, object]]:
    dataset = MfcRawOperatorDataset(
        file_paths=data_source,
        max_len=max_len,
        k_dim=k_dim,
        base_seed=seed,
        time_subsample=time_subsample,
        space_subsample=space_subsample,
    )
    num_total = len(dataset)
    if num_total == 0:
        raise ValueError(f"Resolved MFC dataset is empty: {data_source}")

    indices = np.arange(num_total)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(indices)
    if num_samples is not None and num_samples > 0:
        indices = indices[: min(num_samples, num_total)]

    batches: list[dict[str, object]] = []
    pending_samples: list[dict[str, object]] = []
    pending_ids: list[str] = []
    pending_source_indices: list[int] = []
    pending_signature = None

    def _flush_pending() -> None:
        nonlocal pending_samples, pending_ids, pending_source_indices, pending_signature
        if not pending_samples:
            return
        data_dict, target = _raw_collate(pending_samples, device)
        batches.append(
            {
                "data_dict": data_dict,
                "target": target,
                "sample_ids": pending_ids[:],
                "source_indices": pending_source_indices[:],
            }
        )
        pending_samples = []
        pending_ids = []
        pending_source_indices = []
        pending_signature = None

    for index in [int(idx) for idx in indices]:
        split_seed = seed + index if seed is not None else None
        sample = _split_fixed_demo_query_sample(dataset[index], split_seed, demo_num=demo_num)
        signature = _shape_signature(sample)
        if pending_samples and (len(pending_samples) >= batch_size or signature != pending_signature):
            _flush_pending()
        pending_samples.append(sample)
        pending_ids.append(_sample_id(dataset, index))
        pending_source_indices.append(index)
        pending_signature = signature

    _flush_pending()
    return batches


def load_single_raw_operator_batch(
    data_source: str,
    sample_index: int,
    seed: int,
    device: torch.device,
    max_len: int | None,
    k_dim: int,
    demo_num: int,
    time_subsample: int | None = None,
    space_subsample: int | None = None,
) -> dict[str, object]:
    dataset = MfcRawOperatorDataset(
        file_paths=data_source,
        max_len=max_len,
        k_dim=k_dim,
        base_seed=seed,
        time_subsample=time_subsample,
        space_subsample=space_subsample,
    )
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"Sample index {sample_index} is out of range for dataset {data_source}.")

    split_seed = seed + sample_index if seed is not None else None
    sample = _split_fixed_demo_query_sample(dataset[sample_index], split_seed, demo_num=demo_num)
    data_dict, target = _raw_collate([sample], device)
    return {
        "data_dict": data_dict,
        "target": target,
        "sample_ids": [_sample_id(dataset, sample_index)],
        "source_indices": [sample_index],
    }


def _as_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def _batch_rel_l2(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    flat_target = target.reshape(target.shape[0], -1)
    flat_prediction = prediction.reshape(prediction.shape[0], -1)
    numer = torch.linalg.vector_norm(flat_target - flat_prediction, dim=1)
    denom = torch.linalg.vector_norm(flat_target, dim=1)

    rel_l2 = torch.empty_like(numer)
    nonzero = denom != 0
    rel_l2[nonzero] = numer[nonzero] / denom[nonzero]
    rel_l2[~nonzero] = torch.where(
        numer[~nonzero] == 0,
        torch.zeros_like(numer[~nonzero]),
        torch.full_like(numer[~nonzero], float("inf")),
    )
    return rel_l2


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        if numerator == 0.0:
            return 1.0
        return float("inf")
    return numerator / denominator


def _relative_improvement(baseline: np.ndarray, chain: np.ndarray) -> np.ndarray:
    improvement = np.zeros_like(baseline)
    nonzero = baseline != 0.0
    improvement[nonzero] = (baseline[nonzero] - chain[nonzero]) / baseline[nonzero]
    regression_from_zero = (~nonzero) & (chain != 0.0)
    improvement[regression_from_zero] = -1.0
    return improvement


def _relative_improvement_scalar(baseline: float, chain: float) -> float:
    return float(_relative_improvement(np.asarray([baseline], dtype=np.float64), np.asarray([chain], dtype=np.float64))[0])


def _compute_errors(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, np.ndarray]:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape mismatch: expected {tuple(target.shape)}, got {tuple(prediction.shape)}")
    mse = torch.mean((target - prediction) ** 2, dim=tuple(range(1, target.dim())))
    rel_l2 = _batch_rel_l2(target, prediction)
    return {
        "mse": mse.detach().cpu().numpy().astype(np.float64),
        "rel_l2": rel_l2.detach().cpu().numpy().astype(np.float64),
    }


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {
            "score": 0.0,
            "chain_rel_l2": float("nan"),
            "baseline_rel_l2": float("nan"),
            "improvement_ratio": 0.0,
            "mean_relative_improve": 0.0,
            "var_rel_improve": 0.0,
            "var_chain_rel_l2": 0.0,
            "improve_ratio": 0.0,
            "regress_ratio": 0.0,
            "success_rate": 0.0,
            "failure_count": 0,
            "chain_mse": float("nan"),
            "baseline_mse": float("nan"),
        }

    baseline_rel_l2 = np.asarray([float(row["baseline_rel_l2"]) for row in rows], dtype=np.float64)
    chain_rel_l2 = np.asarray([float(row["chain_rel_l2"]) for row in rows], dtype=np.float64)
    baseline_mse = np.asarray([float(row["baseline_mse"]) for row in rows], dtype=np.float64)
    chain_mse = np.asarray([float(row["chain_mse"]) for row in rows], dtype=np.float64)
    success_flags = np.asarray([bool(row["success"]) for row in rows], dtype=bool)

    improvement_ratio = _ratio(float(np.mean(baseline_rel_l2)), float(np.mean(chain_rel_l2)))
    rel_improve = _relative_improvement(baseline_rel_l2, chain_rel_l2)

    return {
        "score": float(improvement_ratio),
        "chain_rel_l2": float(np.mean(chain_rel_l2)),
        "baseline_rel_l2": float(np.mean(baseline_rel_l2)),
        "improvement_ratio": float(improvement_ratio),
        "mean_relative_improve": float(np.mean(rel_improve)),
        "var_rel_improve": float(np.var(rel_improve)),
        "var_chain_rel_l2": float(np.var(chain_rel_l2)),
        "improve_ratio": float(np.mean(chain_rel_l2 <= baseline_rel_l2)),
        "regress_ratio": float(np.mean(chain_rel_l2 > baseline_rel_l2)),
        "success_rate": float(np.mean(success_flags)),
        "failure_count": int(np.sum(~success_flags)),
        "chain_mse": float(np.mean(chain_mse)),
        "baseline_mse": float(np.mean(baseline_mse)),
    }


def evaluate_dataset_source(
    module: Any,
    wrapper: IconWrapper,
    data_source: str,
    batch_size: int,
    num_samples: int | None,
    seed: int,
    demo_num: int,
    device: torch.device,
    max_len: int | None,
    k_dim: int,
    time_subsample: int | None = None,
    space_subsample: int | None = None,
) -> dict[str, object]:
    dataset_name = Path(data_source).stem
    dataset_rows: list[dict[str, object]] = []
    batches = load_raw_operator_batches(
        data_source=data_source,
        batch_size=batch_size,
        num_samples=num_samples,
        seed=seed,
        device=device,
        max_len=max_len,
        k_dim=k_dim,
        demo_num=demo_num,
        time_subsample=time_subsample,
        space_subsample=space_subsample,
    )

    for batch in batches:
        sample_ids = batch["sample_ids"]
        source_indices = batch["source_indices"]
        target = batch["target"].to(device)

        baseline_prediction = _as_tensor(wrapper.predict(data=batch["data_dict"]), device)
        baseline_errors = _compute_errors(target, baseline_prediction)

        try:
            context = {"data": batch["data_dict"], "model": wrapper}
            chain_prediction = _as_tensor(module.run_icon_chop(context), device)
            chain_errors = _compute_errors(target, chain_prediction)
            success_flags = [True] * len(sample_ids)
            error_messages = [""] * len(sample_ids)
        except Exception as exc:
            chain_errors = {
                "mse": np.where(baseline_errors["mse"] == 0.0, 1.0, baseline_errors["mse"] * 2.0),
                "rel_l2": np.where(baseline_errors["rel_l2"] == 0.0, 1.0, baseline_errors["rel_l2"] * 2.0),
            }
            success_flags = [False] * len(sample_ids)
            error_messages = [str(exc)] * len(sample_ids)

        for index, sample_id in enumerate(sample_ids):
            baseline_rel_l2 = float(baseline_errors["rel_l2"][index])
            chain_rel_l2 = float(chain_errors["rel_l2"][index])
            row = {
                "dataset_name": dataset_name,
                "sample_id": sample_id,
                "source_index": int(source_indices[index]),
                "baseline_rel_l2": baseline_rel_l2,
                "chain_rel_l2": chain_rel_l2,
                "baseline_mse": float(baseline_errors["mse"][index]),
                "chain_mse": float(chain_errors["mse"][index]),
                "improvement_ratio": float(_ratio(baseline_rel_l2, chain_rel_l2)),
                "relative_improve": _relative_improvement_scalar(baseline_rel_l2, chain_rel_l2),
                "success": success_flags[index],
                "error": error_messages[index],
            }
            dataset_rows.append(row)

    best_sample = None
    successful_rows = [row for row in dataset_rows if row["success"]]
    if successful_rows:
        best_sample = max(
            successful_rows,
            key=lambda row: (
                float(row["relative_improve"]),
                -float(row["chain_rel_l2"]),
                row["sample_id"],
            ),
        )

    return {
        "dataset_name": dataset_name,
        "data_path": str(Path(data_source)),
        "num_samples": len(dataset_rows),
        "metrics": summarize_rows(dataset_rows),
        "rows": dataset_rows,
        "best_sample": best_sample,
    }


def evaluate_program_file(
    program_path: str,
    ckpt_path: str,
    data_sources: list[str],
    batch_size: int,
    num_samples: int,
    seed: int,
    demo_num: int,
    max_len: int,
    k_dim: int,
    model_in_features: int,
    device_name: str | None,
) -> dict[str, Any]:
    module = load_program_module(program_path)
    if not hasattr(module, "run_icon_chop"):
        raise AttributeError("Program must define `run_icon_chop(context)`")

    device = resolve_device(device_name)
    model = get_model(ckpt_path=ckpt_path, device=device, in_features=model_in_features)
    wrapper = IconWrapper(model)

    all_rows: list[dict[str, object]] = []
    dataset_summaries: list[dict[str, object]] = []

    for data_source in data_sources:
        dataset_result = evaluate_dataset_source(
            module=module,
            wrapper=wrapper,
            data_source=data_source,
            batch_size=batch_size,
            num_samples=num_samples,
            seed=seed,
            demo_num=demo_num,
            device=device,
            max_len=max_len,
            k_dim=k_dim,
        )
        dataset_summaries.append(
            {
                "dataset_name": dataset_result["dataset_name"],
                "data_path": dataset_result["data_path"],
                "num_samples": dataset_result["num_samples"],
                "metrics": dataset_result["metrics"],
                "best_sample": dataset_result["best_sample"],
            }
        )
        all_rows.extend(dataset_result["rows"])

    metrics = summarize_rows(all_rows)
    diagnostics = {
        "num_datasets": len(dataset_summaries),
        "dataset_summaries": dataset_summaries,
        "best_sample": max(all_rows, key=lambda row: row["improvement_ratio"]) if all_rows else None,
        "worst_sample": min(all_rows, key=lambda row: row["improvement_ratio"]) if all_rows else None,
    }
    return {"metrics": metrics, "diagnostics": diagnostics}
